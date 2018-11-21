#include <vector>
#include <iostream>
#include <complex>
#include <boost/multi_array.hpp>
#include "../utilities.hpp"
#include "../configuration.hpp"
#include "../results.hpp"
#include "../symplectic.hpp"
#include "dnls.hpp"

using namespace std;

namespace dnls {

	int main(int argc, char* argv[]){

		double beta;
		{
			using namespace boost::program_options;
			parse_cmdline parser("Options for "s + argv[0]);
			parser.options.add_options()
					("beta", value(&beta)->required(), "fourth order nonlinearity");
			parser(argc, argv);
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
			s_move = 0, s_dump, s_entropy, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		cufftHandle fft = 0;
		destructor([&]{ cufftDestroy(fft); });
		cufftPlan1d(&fft, gconf.chain_length, CUFFT_Z2Z, gconf.shard_copies) && assertcufft;
		cufftSetStream(fft, streams[s_move]) && assertcufft;
		cudalist<cufftDoubleComplex> evolve_linear_table(8 * gconf.chain_length);
		double beta_dt_symplectic[8];
		{
			loopi(8) beta_dt_symplectic[i] = beta * gconf.dt * (i == 7 ? 2. : 1.) * symplectic_c[i];

			boost::multi_array<complex<double>, 2> evolve_linear_table_host(boost::extents[8][gconf.chain_length]);
			auto normalization = 1. / gconf.chain_length;
			complex<double> complexdt = 1i * gconf.dt;
			loopi(8) loopj(gconf.chain_length)
				evolve_linear_table_host[i][j] = exp(complexdt * symplectic_d[i == 7 ? 0 : i] * omega_host[j]) * normalization;
			cudaMemcpy(evolve_linear_table, evolve_linear_table_host.origin(), 8 * gconf.chain_length * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) && assertcu;
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
				evolve_nonlinear(beta_dt_symplectic[i && !k ? 7 : k], streams[s_move]);
				cufftExecZ2Z(fft, gres.shard, gres.shard, CUFFT_FORWARD) && assertcufft;
				evolve_linear(&evolve_linear_table[k * gconf.chain_length], streams[s_move]);
				cufftExecZ2Z(fft, gres.shard, gres.shard, CUFFT_INVERSE) && assertcufft;
			}
			completion finish_move = evolve_nonlinear(beta_dt_symplectic[0], streams[s_move]);
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
