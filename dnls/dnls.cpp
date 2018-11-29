#include <vector>
#include <iostream>
#include <complex>
#include <boost/multi_array.hpp>
#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "dnls.hpp"

using namespace std;

namespace dnls {

	int main(int argc, char* argv[]){

		double beta;
		bool no_linear_callback, no_nonlinear_callback;
		{
			using namespace boost::program_options;
			parse_cmdline parser("Options for "s + argv[0]);
			parser.options.add_options()
					("no-linear-callback", "do not use cuFFT callback for linear evolution")
					("no-nonlinear-callback", "do not use cuFFT callback for nonlinear evolution")
					("beta", value(&beta)->required(), "fourth order nonlinearity");
			parser(argc, argv);
			no_linear_callback = parser.vm.count("no-linear-callback");
			no_nonlinear_callback = parser.vm.count("no-nonlinear-callback");
		}

		vector<double> omega_host(gconf.chain_length);
		loopk(gconf.chain_length){
			auto s2 = 2 * sin(k * M_PI / gconf.chain_length);
			omega_host[k] = s2 * s2;
		}
		cudalist<double> omega(gconf.chain_length);
		cudaMemcpy(omega, omega_host.data(), gconf.chain_length * sizeof(double), cudaMemcpyHostToDevice) && assertcu;

		cudalist<cufftDoubleComplex> psis_k(gconf.shard_elements);
		results res(true);

		enum {
			s_move = 0, s_linear_callback_settings, s_nonlinear_callback_settings, s_dump, s_entropy, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		cudalist<cufftDoubleComplex> evolve_linear_tables_all(8 * gconf.chain_length);
		{
			boost::multi_array<complex<double>, 2> evolve_linear_table_host(boost::extents[8][gconf.chain_length]);
			auto normalization = 1. / gconf.chain_length;
			complex<double> complexdt = 1i * gconf.dt;
			loopi(8) loopj(gconf.chain_length)
					evolve_linear_table_host[i][j] = exp(complexdt * symplectic_d[i == 7 ? 0 : i] * omega_host[j]) * normalization;
			cudaMemcpy(evolve_linear_tables_all, evolve_linear_table_host.origin(), 8 * gconf.chain_length * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) && assertcu;
		}

		cufftHandle fft = 0, fft_elvolve_psik = 0, fft_elvolve_psi = 0;
		destructor([&]{ cufftDestroy(fft); cufftDestroy(fft_elvolve_psik); cufftDestroy(fft_elvolve_psi); });

		cufftPlan1d(&fft, gconf.chain_length, CUFFT_Z2Z, gconf.shard_copies) && assertcufft;
		cufftSetStream(fft, streams[s_move]) && assertcufft;
		if(!no_linear_callback){
			using namespace callback;
			cufftPlan1d(&fft_elvolve_psik, gconf.chain_length, CUFFT_Z2Z, gconf.shard_copies) && assertcufft;
			cufftSetStream(fft_elvolve_psik, streams[s_move]) && assertcufft;
			auto evolve_linear_ptr_host = get_device_object(evolve_linear_ptr);
			auto evolve_linear_tables_all_ptr = *evolve_linear_tables_all;
			cufftXtSetCallback(fft_elvolve_psik, (void**)&evolve_linear_ptr_host, CUFFT_CB_LD_COMPLEX_DOUBLE, (void**)&evolve_linear_tables_all_ptr) && assertcufft;
			set_device_object(gconf.chain_length, chainlen);
		}
		cudalist<double, true> beta_dt_symplectic_all;
		auto beta_dt_symplectic_ptr = get_device_address(callback::beta_dt_symplectic);
		if(!no_nonlinear_callback){
			using namespace callback;
			cufftPlan1d(&fft_elvolve_psi, gconf.chain_length, CUFFT_Z2Z, gconf.shard_copies) && assertcufft;
			cufftSetStream(fft_elvolve_psi, streams[s_move]) && assertcufft;
			auto evolve_nonlinear_ptr_host = get_device_object(evolve_nonlinear_ptr);
			cufftXtSetCallback(fft_elvolve_psi, (void**)&evolve_nonlinear_ptr_host, CUFFT_CB_LD_COMPLEX_DOUBLE, 0) && assertcufft;
			beta_dt_symplectic_all = cudalist<double, true>(8, true);
			loopk(8) beta_dt_symplectic_all[k] = beta * gconf.dt * (k == 7 ? 2. : 1.) * symplectic_c[k];
		}

		exception_ptr callback_err;
		completion throttle;
		uint64_t t = gconf.timebase;
		auto dumper = [&]{
			cudaMemcpyAsync(gres.shard_host, gres.shard, gconf.shard_size, cudaMemcpyDeviceToHost, streams[s_dump]) && assertcu;
			completion(streams[s_dump]).blocks(streams[s_move]);
			add_cuda_callback(streams[s_dump], callback_err, [&, t](cudaError_t status) {
				if(callback_err) return;
				status && assertcu;
				res.write_shard(t, gres.shard_host);
			});
		};
		destructor(cudaDeviceSynchronize);
		for (bool unified_max_step = false, throttleswitch = true;; t += gconf.steps_grouping, throttleswitch ^= true) {
			//throttle enqueuing of kernels
			if (throttleswitch) throttle = completion(streams[s_move]);
			else throttle.wait();

			if(t % gconf.steps_grouping_dump == 0) dumper();
			cufftExecZ2Z(fft, gres.shard, psis_k, CUFFT_FORWARD) && assertcufft;
			completion(streams[s_move]).blocks(streams[s_entropy]);
			make_linenergies(psis_k, omega, streams[s_entropy]);
			add_cuda_callback(streams[s_entropy], callback_err, [&, t](cudaError_t status) {
				if(callback_err) return;
				status && assertcu;
				auto entropies = res.entropies(gres.linenergies_host, 1. / gconf.shard_elements);
				res.check_entropy(entropies);
				if(t % gconf.steps_grouping_dump == 0) res.write_linenergies(t);
				res.write_entropy(t, entropies);
			});

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

			for(uint32_t i = 0; i < gconf.steps_grouping; i++) for(int k = 0; k < 7; k++){
				if(fft_elvolve_psi){
					cudaMemcpyAsync(beta_dt_symplectic_ptr, &beta_dt_symplectic_all[!k && i ? 7 : k], sizeof(double), cudaMemcpyHostToDevice, streams[s_nonlinear_callback_settings]);
					completion(streams[s_nonlinear_callback_settings]).blocks(streams[s_move]);
				}
				else evolve_nonlinear(beta * gconf.dt * (!k && i ? 2. : 1.) * symplectic_c[k], streams[s_move]);
				cufftExecZ2Z(fft_elvolve_psi ?: fft, gres.shard, gres.shard, CUFFT_FORWARD) && assertcufft;
				completion(streams[s_move]).blocks(streams[s_nonlinear_callback_settings]);
				if(fft_elvolve_psik){
					memset_device_object(callback::evolve_linear_table_idx, k, streams[s_linear_callback_settings]);
					completion(streams[s_linear_callback_settings]).blocks(streams[s_move]);
				}
				else evolve_linear(&evolve_linear_tables_all[k * gconf.chain_length], streams[s_move]);
				cufftExecZ2Z(fft_elvolve_psik ?: fft, gres.shard, gres.shard, CUFFT_INVERSE) && assertcufft;
				completion(streams[s_move]).blocks(streams[s_linear_callback_settings]);
			}
			completion finish_move = evolve_nonlinear(beta * gconf.dt * symplectic_c[7], streams[s_move]);
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
	::programs()["dnls"] = dnls::main;
};
