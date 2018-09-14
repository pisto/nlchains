#include <iostream>
#include <vector>
#include "../utilities.hpp"
#include "../results.hpp"
#include "../symplectic.hpp"
#include "kg_fpu_toda.hpp"

using namespace std;

namespace kg_fpu_toda {

	template<Model model>
	int main(int argc, char* argv[]){

		bool use_split_kernel = false;
		double m, alpha, beta;
		{
			using namespace boost::program_options;
			parse_cmdline parser(string("Options for ") + argv[0]);
			if (model == KG) parser.options.add_options()(",m", value(&m)->required(), "linear parameter m");
			if (model != KG) parser.options.add_options()("alpha", value(&alpha)->required(), "third order nonlinearity");
			parser.options.add_options()
					("beta", value(&beta)->required(), "fourth order nonlinearity")
					("split-kernel", "force use of split kernel");
			try {
				parser(argc, argv);
				(::configuration&)gconf = ::gconf;
				use_split_kernel = parser.vm.count("split-kernel");
			} catch(const invalid_argument& e) {
				if(!mpi_global_coord) cerr<<"Error in command line: "<<e.what()<<endl<<parser.options<<endl;
				return 1;
			}
		}

		cudalist<double> omega(gconf.chain_length);
		{
			vector<double> omega_host(gconf.chain_length);
			for (int k = 0; k <= int(gconf.chain_length) / 2; k++) {
				auto s2 = 2 * sin(k * M_PI / gconf.chain_length);
				omega_host[k] = sqrt(m + s2 * s2);
			}
			for (int k = -1; k >= -int(gconf.chain_length - 1) / 2; k--) omega_host[gconf.chain_length + k] = omega_host[-k];
			cudaMemcpy(omega, omega_host.data(), gconf.linenergy_size, cudaMemcpyHostToDevice) && assertcu;
		}

		results res(m == 0.);

		enum {
			s_move = 0, s_dump, s_entropy, s_entropy_aux, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		//FFT is performed on the data that is first converted to complex format and transposed into
		//an array with indices { real/img, copy, chain_index }. This allows to simply double the batch
		//size of the FFT (gconf.shard_copies * 2) to transform both phi and pi.
		cudalist<double2> fft_phi(gconf.shard_elements * 2);
		auto fft_pi = &fft_phi[gconf.shard_elements];
		cufftHandle fft_plan = 0;
		destructor([&]{ if(fft_plan) cufftDestroy(fft_plan); });
		cufftPlan1d(&fft_plan, gconf.chain_length, CUFFT_Z2Z, 2 * int(gconf.shard_copies)) && assertcufft;
		cufftSetStream(fft_plan, streams[s_entropy]) && assertcufft;

		//constants
		double dt_c_host[8], dt_d_host[8], alpha2 = 2 * alpha, alpha2_inv = 1 / alpha2;
		loopi(8) dt_c_host[i] = symplectic_c[i] * gconf.dt, dt_d_host[i] = symplectic_d[i] * gconf.dt;
		set_device_object(dt_c_host, dt_c, streams[s_move]);
		set_device_object(dt_d_host, dt_d, streams[s_move]);
		set_device_object(alpha, kg_fpu_toda::alpha, streams[s_move]);
		set_device_object(m, kg_fpu_toda::m, streams[s_move]);
		set_device_object(alpha2, kg_fpu_toda::alpha2, streams[s_move]);
		set_device_object(alpha2_inv, kg_fpu_toda::alpha2_inv, streams[s_move]);
		set_device_object(beta, kg_fpu_toda::beta, streams[s_move]);

		exception_ptr callback_err;
		completion throttle;
		plane2split* splitter = 0;
		if(use_split_kernel){
			splitter = new plane2split(gconf.chain_length, gconf.shard_copies);
			splitter->split(gres.shard, streams[s_move]);
		}
		destructor([&]{ delete splitter; });
		uint64_t t = gconf.timebase;
		auto dumper = [&]{
			cudaMemcpyAsync(gres.shard_host, gres.shard, gconf.shard_size, cudaMemcpyDeviceToHost, streams[s_dump]) && assertcu;
			if(!splitter) completion(streams[s_dump]).blocks(streams[s_move]);
			add_cuda_callback(streams[s_dump], [&, t](cudaStream_t, cudaError_t status) {
				if(callback_err) return;
				try {
					status && assertcu;
					res.write_shard(t, gres.shard_host);
				} catch(...) { callback_err = current_exception(); }
			});
		};
		destructor(cudaDeviceSynchronize);
		for (bool unified_max_step = false, throttleswitch = true;; t += gconf.steps_grouping, throttleswitch ^= true) {
			//throttle enqueuing of kernels
			if (throttleswitch) throttle = completion(streams[s_move]);
			else throttle.wait();

			if(splitter) splitter->plane(gres.shard, streams[s_move], streams[s_dump]).blocks(streams[s_entropy]);
			if(t % gconf.steps_grouping_dump == 0) dumper();
			//entropy stream is already synced to move or dump stream, that is the readiness of the planar representation
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			//interleave phi and pi with zeroes, since we do a complex FFT
			cudaMemcpy2DAsync(fft_phi, sizeof(double2), gres.shard, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy]) && assertcu;
			cudaMemcpy2DAsync(fft_pi, sizeof(double2), &gres.shard[0].y, sizeof(double2), sizeof(double),
			                  gconf.shard_elements, cudaMemcpyDeviceToDevice, streams[s_entropy_aux]) && assertcu;
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);
			if(!splitter) completion(streams[s_entropy]).blocks(streams[s_move]);
			cufftExecZ2Z(fft_plan, fft_phi, fft_phi, CUFFT_FORWARD) && assertcufft;
			make_linenergies(fft_phi, fft_pi, omega, streams[s_entropy]);
			//cleanup buffer for next FFT
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			cudaMemsetAsync(fft_phi, 0, gconf.shard_size * 2, streams[s_entropy_aux]) && assertcu;
			add_cuda_callback(streams[s_entropy], [&, t](cudaStream_t, cudaError_t status) {
				if(callback_err) return;
				try {
					status && assertcu;
					auto entropies = res.entropies(gres.linenergies_host, (0.5 / gconf.shard_copies) / gconf.chain_length);
					res.check_entropy(entropies);
					if(t % gconf.steps_grouping_dump == 0) res.write_linenergies(t);
					res.write_entropy(t, entropies);
				} catch(...) { callback_err = current_exception(); }
			});
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);

			//termination checks
			if(callback_err) rethrow_exception(callback_err);
			if(quit_requested && !unified_max_step){
				boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
				unified_max_step = true;
			}
			if(gconf.steps && gconf.steps == t){
				//make sure all MPI calls are matched, use_split_kernel
				if(!unified_max_step)
					boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
				break;
			}

			completion finish_move = move<model>(splitter, streams[s_move]);
			finish_move.blocks(streams[s_entropy]);
			finish_move.blocks(streams[s_dump]);

		}
		if(t % gconf.steps_grouping_dump != 0){
			dumper();
			completion(streams[s_entropy]).wait();
			res.write_linenergies(t);
		}
		return 0;
	}
}

ginit = []{
	auto& programs = ::programs();
	programs["KleinGordon"] = kg_fpu_toda::main<kg_fpu_toda::KG>;
	programs["FPU"] = kg_fpu_toda::main<kg_fpu_toda::FPU>;
	programs["Toda"] = kg_fpu_toda::main<kg_fpu_toda::Toda>;
};
