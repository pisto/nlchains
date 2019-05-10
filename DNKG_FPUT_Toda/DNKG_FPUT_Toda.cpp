#include <iostream>
#include <vector>
#include "../common/utilities.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "../common/loop_control.hpp"
#include "DNKG_FPUT_Toda.hpp"

using namespace std;

/*
 * Yoshida symplectic integration for the nonlinear Klein Gordon, alpha/beta FPUT and Toda chains.
 * For a description of the implementation strategies on GPU see DNKG_FPUT_Toda.cu .
 */

namespace DNKG_FPUT_Toda {

	template<Model model>
	int main(int argc, char *argv[]) {

		bool split_kernel = false;
		double m = 0., alpha = 0., beta = 0.;
		{
			using namespace boost::program_options;
			parse_cmdline parser("Options for "s + argv[0]);
			switch (model) {
				case DNKG:
					parser.options.add_options()
							(",m", value(&m)->required(), "linear parameter m")
							("beta", value(&beta)->required(), "fourth order nonlinearity");
					break;
				case FPUT:
					parser.options.add_options()
							("alpha", value(&alpha)->required(), "third order nonlinearity")
							("beta", value(&beta)->required(), "fourth order nonlinearity");
					break;
				case Toda:
					parser.options.add_options()
							("alpha", value(&alpha)->required(),
							 "steepness of exponential, V(dx)=e^(2*alpha*dx)/(4*alpha^2)-dx/(2*alpha)");
					break;
			}
			parser.options.add_options()("split_kernel", "force use of split kernel");
			parser.run(argc, argv);
			split_kernel = parser.vm.count("split_kernel");
		}
		auto ctx = cuda_ctx.activate(mpi_node_coord);

		cudalist<double> omega(gconf.chain_length);
		cudaMemcpy(omega, dispersion(m).data(), gconf.sizeof_linenergies, cudaMemcpyHostToDevice) && assertcu;

		results res(m == 0.);

		enum {
			s_move = 0, s_dump, s_entropy, s_entropy_aux, s_results, s_total
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
		set_device_object(alpha, DNKG_FPUT_Toda::alpha);
		set_device_object(m, DNKG_FPUT_Toda::m);
		set_device_object(alpha2, DNKG_FPUT_Toda::alpha2);
		set_device_object(alpha2_inv, DNKG_FPUT_Toda::alpha2_inv);
		set_device_object(beta, DNKG_FPUT_Toda::beta);

		/*
		 * The code in DNKG_FPUT_Toda.cu might not find an optimized kernel that uses the planar format. If that is the
		 * or if the user specifies --split_kernel, then we use the split format (see utilities_cuda.cuh).
		 */
		plane2split *splitter = 0;
		if (split_kernel) {
			splitter = new plane2split(gconf.chain_length, gconf.shard_copies);
			splitter->split(gres.shard_gpu, streams[s_move]);
		}
		destructor([&] { delete splitter; });

		loop_control_gpu loop_ctl(gconf.time_offset, streams[s_move]);
		auto dumper = [&] {
			cudaMemcpyAsync(gres.shard_host, gres.shard_gpu, gconf.sizeof_shard, cudaMemcpyDeviceToHost, streams[s_dump]) &&
			assertcu;
			completion done_copy(streams[s_dump]);
			if (!splitter) done_copy.blocks(streams[s_move]);
			done_copy.blocks(streams[s_results]);
		};
		destructor(cudaDeviceSynchronize);
		while (1) {
			if (splitter) splitter->plane(gres.shard_gpu, streams[s_move], streams[s_dump]).blocks(streams[s_entropy]);
			bool full_dump = loop_ctl % gconf.dump_interval == 0;
			if (full_dump) dumper();
			//entropy stream is already synced to move or dump stream, that is the readiness of the planar representation
			cudaMemsetAsync(fft_phi, 0, gconf.sizeof_shard * 2, streams[s_entropy]) && assertcu;
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			//interleave phi and pi with zeroes, since we do a complex FFT
			cudaMemcpy2DAsync(fft_phi, sizeof(double2), gres.shard_gpu, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy]) && assertcu;
			cudaMemcpy2DAsync(fft_pi, sizeof(double2), &gres.shard_gpu[0].y, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy_aux]) && assertcu;
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);
			if (!splitter) completion(streams[s_entropy]).blocks(streams[s_move]);
			cufftExecZ2Z(fft_plan, fft_phi, fft_phi, CUFFT_FORWARD) && assertcufft;
			make_linenergies(fft_phi, fft_pi, omega, streams[s_entropy]);
			completion(streams[s_entropy]).blocks(streams[s_results]);
			add_cuda_callback(streams[s_results], loop_ctl.callback_err,
			                  [&, full_dump, t = *loop_ctl](cudaError_t status) {
				                  if (loop_ctl.callback_err) return;
				                  status && assertcu;
				                  res.calc_linenergies((0.5 / gconf.shard_copies) / gconf.chain_length).calc_entropies().check_entropy().write_entropy(t);
				                  if (full_dump) res.write_linenergies(t).write_shard(t);
			                  });
			completion done_results(streams[s_results]);
			done_results.blocks(streams[s_dump]);
			done_results.blocks(streams[s_entropy]);

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
			completion(streams[s_dump]).wait();
			res.write_shard(loop_ctl);
		}
		return 0;
	}
}

ginit = [] {
	using namespace DNKG_FPUT_Toda;
	auto &programs = ::programs();
	programs["DNKG"] = main<DNKG>;
	programs["FPUT"] = main<FPUT>;
	programs["Toda"] = main<Toda>;
};
