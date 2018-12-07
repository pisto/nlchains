#include <iostream>
#include <vector>
#include "../common/utilities.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "../common/loop_control.hpp"
#include "kg_fpu_toda.hpp"

using namespace std;

/*
 * Yoshida symplectic integration for the nonlinear Klein Gordon, alpha/beta FPUT and Toda chains.
 * For a description of the implementation strategies on GPU see kg_fpu_toda.cu .
 */

namespace kg_fpu_toda {

	template<Model model>
	int main(int argc, char *argv[]) {

		bool split_kernel = false;
		double m = 0., alpha = 0., beta;
		{
			using namespace boost::program_options;
			parse_cmdline parser("Options for "s + argv[0]);
			if (model == KG) parser.options.add_options()(",m", value(&m)->required(), "linear parameter m");
			else parser.options.add_options()("alpha", value(&alpha)->required(), "third order nonlinearity");
			parser.options.add_options()
					("beta", value(&beta)->required(), "fourth order nonlinearity")
					("split_kernel", "force use of split kernel");
			parser(argc, argv);
			split_kernel = parser.vm.count("split_kernel");
		}

		cudalist<double> omega(gconf.chain_length);
		{
			vector<double> omega_host(gconf.chain_length);
			for (int k = 0; k <= int(gconf.chain_length) / 2; k++) {
				auto s2 = 2 * sin(k * M_PI / gconf.chain_length);
				omega_host[k] = sqrt(m + s2 * s2);
			}
			for (int k = -1; k >= -int(gconf.chain_length - 1) / 2; k--)
				omega_host[gconf.chain_length + k] = omega_host[-k];
			cudaMemcpy(omega, omega_host.data(), gconf.sizeof_linenergies, cudaMemcpyHostToDevice) && assertcu;
		}

		results res(m == 0.);

		enum {
			s_move = 0, s_dump, s_entropy, s_entropy_aux, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		/*
		 * FFT for the linear energies is performed on the data that is first converted to complex format
		 * (real = phi, imaginary = pi), and transposed into an array with indices { real/img, copy, chain_index }.
		 * This allows to simply double the batch size of the FFT (gconf.shard_copies * 2) to transform both phi and pi,
		 * and it also allows to run the next symplectic steps right after the copy to the auxiliary fft buffer.
		 */
		cudalist<double2> fft_phi(gconf.shard_elements * 2);
		auto fft_pi = &fft_phi[gconf.shard_elements];
		cufftHandle fft_plan = 0;
		destructor([&] { if (fft_plan) cufftDestroy(fft_plan); });
		cufftPlan1d(&fft_plan, gconf.chain_length, CUFFT_Z2Z, 2 * int(gconf.shard_copies)) && assertcufft;
		cufftSetStream(fft_plan, streams[s_entropy]) && assertcufft;

		//constants
		double dt_c_host[8], dt_d_host[8], alpha2 = 2 * alpha, alpha2_inv = 1 / alpha2;
		loopi(8) dt_c_host[i] = symplectic_c[i] * gconf.dt, dt_d_host[i] = symplectic_d[i] * gconf.dt;
		set_device_object(dt_c_host, dt_c);
		set_device_object(dt_d_host, dt_d);
		set_device_object(alpha, kg_fpu_toda::alpha);
		set_device_object(m, kg_fpu_toda::m);
		set_device_object(alpha2, kg_fpu_toda::alpha2);
		set_device_object(alpha2_inv, kg_fpu_toda::alpha2_inv);
		set_device_object(beta, kg_fpu_toda::beta);

		/*
		 * The code in kg_fpu_toda.cu might not find an optimized kernel that uses the planar format. If that is the
		 * or if the user specifies --split_kernel, then we use the split format (see utilities_cuda.cuh).
		 */
		plane2split *splitter = 0;
		if (split_kernel) {
			splitter = new plane2split(gconf.chain_length, gconf.shard_copies);
			splitter->split(gres.shard, streams[s_move]);
		}
		destructor([&] { delete splitter; });

		loop_control loop_ctl(streams[s_move]);
		auto dumper = [&] {
			cudaMemcpyAsync(gres.shard_host, gres.shard, gconf.sizeof_shard, cudaMemcpyDeviceToHost, streams[s_dump]) &&
			assertcu;
			if (!splitter) completion(streams[s_dump]).blocks(streams[s_move]);
			add_cuda_callback(streams[s_dump], loop_ctl.callback_err, [&, t = *loop_ctl](cudaError_t status) {
				if (loop_ctl.callback_err) return;
				status && assertcu;
				res.write_shard(t, gres.shard_host);
			});
		};
		destructor(cudaDeviceSynchronize);
		while (1) {
			if (splitter) splitter->plane(gres.shard, streams[s_move], streams[s_dump]).blocks(streams[s_entropy]);
			if (loop_ctl % gconf.dump_interval == 0) dumper();
			//entropy stream is already synced to move or dump stream, that is the readiness of the planar representation
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			//interleave phi and pi with zeroes, since we do a complex FFT
			cudaMemcpy2DAsync(fft_phi, sizeof(double2), gres.shard, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy]) && assertcu;
			cudaMemcpy2DAsync(fft_pi, sizeof(double2), &gres.shard[0].y, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy_aux]) && assertcu;
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);
			if (!splitter) completion(streams[s_entropy]).blocks(streams[s_move]);
			cufftExecZ2Z(fft_plan, fft_phi, fft_phi, CUFFT_FORWARD) && assertcufft;
			make_linenergies(fft_phi, fft_pi, omega, streams[s_entropy]);
			//cleanup buffer for next FFT
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			cudaMemsetAsync(fft_phi, 0, gconf.sizeof_shard * 2, streams[s_entropy_aux]) && assertcu;
			add_cuda_callback(streams[s_entropy], loop_ctl.callback_err, [&, t = *loop_ctl](cudaError_t status) {
				if (loop_ctl.callback_err) return;
				status && assertcu;
				auto entropies = res.entropies(gres.linenergies_host, (0.5 / gconf.shard_copies) / gconf.chain_length);
				res.check_entropy(entropies);
				if (t % gconf.dump_interval == 0) res.write_linenergies(t);
				res.write_entropy(t, entropies);
			});
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);

			if (loop_ctl.break_now()) break;

			completion finish_move = move<model>(splitter, streams[s_move]);
			finish_move.blocks(streams[s_entropy]);
			finish_move.blocks(streams[s_dump]);

			loop_ctl += gconf.kernel_batching;
		}

		if (loop_ctl % gconf.dump_interval != 0) {
			dumper();
			completion(streams[s_entropy]).wait();
			res.write_linenergies(loop_ctl);
		}
		return 0;
	}
}

ginit = [] {
	auto &programs = ::programs();
	programs["KleinGordon"] = kg_fpu_toda::main<kg_fpu_toda::KG>;
	programs["FPU"] = kg_fpu_toda::main<kg_fpu_toda::FPU>;
	programs["Toda"] = kg_fpu_toda::main<kg_fpu_toda::Toda>;
};
