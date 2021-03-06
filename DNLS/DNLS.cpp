#include <vector>
#include <iostream>
#include <complex>
#include <boost/multi_array.hpp>
#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "../common/loop_control.hpp"
#include "DNLS.hpp"

using namespace std;

/*
 * The Yoshida 6th order symplectic algorithm, implemented as per arXiv:1012.3242 . It is suggested in the paper
 * that the linear operator evolution can be performed as a FFT, and here we implement it so.
 *
 * Contrary to DNKG_FPUT_Toda or dDNKG, each symplectic step is broken into a number of FFTs, hence different kernel
 * invocations. This makes the algorithm much slower than the others implemented. This is mitigated by the use of
 * cuFFT callbacks, in order to coalesce the linear/nonlinear evolutions in the load callback of the FFTs. In my
 * experience cuFFT callbacks can be of great benefit, but also in some cases (depending on architecture, clocks,
 * number of GPUs used, etc.) they can degrade performance. By default, they are used for both linear and nonlinear
 * evolutions, but the user can disable one or both with the switches --no_linear_callback and --no_nonlinear_callback.
 */
namespace DNLS {

	int main(int argc, char *argv[]) {

		double beta;
		bool no_linear_callback, no_nonlinear_callback;
		{
			using namespace boost::program_options;
			parse_cmdline parser("Options for "s + argv[0]);
			parser.options.add_options()
					("no_linear_callback", "do not use cuFFT callback for linear evolution")
					("no_nonlinear_callback", "do not use cuFFT callback for nonlinear evolution")
					("beta", value(&beta)->required(), "fourth order nonlinearity");
			parser.run(argc, argv);
			no_linear_callback = parser.vm.count("no_linear_callback");
			no_nonlinear_callback = parser.vm.count("no_nonlinear_callback");
		}
		auto ctx = cuda_ctx.activate(mpi_node_coord);

		auto omega_host = dispersion();
		cudalist<double> omega(gconf.chain_length);
		cudaMemcpy(omega, omega_host.data(), gconf.chain_length * sizeof(double), cudaMemcpyHostToDevice) && assertcu;

		cudalist<cufftDoubleComplex> psis_k(gconf.shard_elements);
		results res(true);

		enum {
			s_move = 0, s_cb_linear, s_cb_nonlinear, s_dump, s_linenergies, s_results, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		/*
		 * Precompute the evolution factors in Fourier space (linear operator), including FFT normalization, for all
		 * the symplectic integration steps.
		 */
		cudalist<cufftDoubleComplex> evolve_linear_tables_all(7 * gconf.chain_length);
		cudaMemcpy(evolve_linear_tables_all, evolve_linear_table().origin(),
		           7 * gconf.chain_length * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) && assertcu;

		/*
		 * Three FFT plans are required because of the nonlinear/linear evolution callbacks. The plain fft plan is used
		 * to make the linear energies, and in case that the user disabled callbacks.
		 * The settings for the callbacks (the symplectic time step, or the linear evolution table) are put on global
		 * variables asynchronously with additional streams.
		 */
		cufftHandle fft_plain = 0, fft_plain_entropy = 0, fft_elvolve_psik = 0, fft_elvolve_psi = 0;
		destructor([&] {
			cufftDestroy(fft_plain);
			cufftDestroy(fft_plain_entropy);
			cufftDestroy(fft_elvolve_psik);
			cufftDestroy(fft_elvolve_psi);
		});

		cudalist<double, true> beta_dt_symplectic_all;
		auto beta_dt_symplectic_ptr = get_device_address(callback::beta_dt_symplectic);
		cudalist<void> area;
		{
			auto init_plan = [&](cufftHandle &fft) {
				size_t dummy;
				cufftCreate(&fft) && assertcufft;
				cufftSetAutoAllocation(fft, false) && assertcufft;
				cufftMakePlan1d(fft, gconf.chain_length, CUFFT_Z2Z, gconf.shard_copies, &dummy) && assertcufft;
				cufftSetStream(fft, streams[s_move]) && assertcufft;
			};
			size_t maxsize = 0;
			auto update_max_size = [&](cufftHandle fft) {   //use only one work area
				size_t size;
				cufftGetSize(fft, &size) && assertcufft;
				maxsize = max(maxsize, size);
			};
			init_plan(fft_plain);
			update_max_size(fft_plain);
			init_plan(fft_plain_entropy);
			cufftSetStream(fft_plain_entropy, streams[s_linenergies]) && assertcufft;
			update_max_size(fft_plain_entropy);
			if (!no_linear_callback) {
				init_plan(fft_elvolve_psik);
				auto evolve_linear_ptr_host = get_device_object(callback::evolve_linear_ptr);
				auto evolve_linear_tables_all_ptr = *evolve_linear_tables_all;
				cufftXtSetCallback(fft_elvolve_psik, (void **) &evolve_linear_ptr_host, CUFFT_CB_LD_COMPLEX_DOUBLE,
				                   (void **) &evolve_linear_tables_all_ptr) && assertcufft;
				set_device_object(gconf.chain_length, callback::chainlen);
				update_max_size(fft_elvolve_psik);
			}
			if (!no_nonlinear_callback) {
				init_plan(fft_elvolve_psi);
				auto evolve_nonlinear_ptr_host = get_device_object(callback::evolve_nonlinear_ptr);
				cufftXtSetCallback(fft_elvolve_psi, (void **) &evolve_nonlinear_ptr_host, CUFFT_CB_LD_COMPLEX_DOUBLE,
				                   0) && assertcufft;
				beta_dt_symplectic_all = cudalist<double, true>(8, true);
				loopk(8) beta_dt_symplectic_all[k] = beta * gconf.dt * (k == 7 ? 2. : 1.) * symplectic_c[k];
				update_max_size(fft_elvolve_psi);
			}
			area = maxsize;
			cufftSetWorkArea(fft_plain, area) && assertcufft;
			cufftSetWorkArea(fft_plain_entropy, area) && assertcufft;
			if (!no_linear_callback) cufftSetWorkArea(fft_elvolve_psik, area) && assertcufft;
			if (!no_nonlinear_callback) cufftSetWorkArea(fft_elvolve_psi, area) && assertcufft;
		}

		loop_control_gpu loop_ctl(streams[s_move]);
		auto dumper = [&] {
			cudaMemcpyAsync(gres.shard_host, gres.shard_gpu, gconf.sizeof_shard, cudaMemcpyDeviceToHost, streams[s_dump]) &&
			assertcu;
			completion done_copy(streams[s_dump]);
			done_copy.blocks(streams[s_move]);
			done_copy.blocks(streams[s_results]);
		};

		cudaDeviceSynchronize() && assertcu;
		destructor(cudaDeviceSynchronize);
		while (1) {
			bool full_dump = loop_ctl % gconf.dump_interval == 0;
			if (full_dump) dumper();
			cufftExecZ2Z(fft_plain_entropy, gres.shard_gpu, psis_k, CUFFT_FORWARD) && assertcufft;
			completion(streams[s_linenergies]).blocks(streams[s_move]);
			make_linenergies(psis_k, omega, streams[s_linenergies]);
			completion(streams[s_linenergies]).blocks(streams[s_results]);
			add_cuda_callback(streams[s_results], loop_ctl.callback_err,
			                  [&, full_dump, t = *loop_ctl](cudaError_t status) {
				                  if (loop_ctl.callback_err) return;
				                  status && assertcu;
				                  res.calc_linenergies(1. / gconf.shard_elements).calc_entropies().check_entropy().write_entropy(t);
				                  if (full_dump) res.write_linenergies(t).write_shard(t);
			                  });
			completion done_results(streams[s_results]);
			done_results.blocks(streams[s_dump]);
			done_results.blocks(streams[s_linenergies]);

			if (loop_ctl.break_now()) break;

			for (uint32_t i = 0; i < gconf.kernel_batching; i++)
				for (int k = 0; k < 7; k++) {
					if (fft_elvolve_psi) {
						cudaMemcpyAsync(beta_dt_symplectic_ptr, &beta_dt_symplectic_all[!k && i ? 7 : k],
						                sizeof(double), cudaMemcpyHostToDevice, streams[s_cb_nonlinear]);
						completion(streams[s_cb_nonlinear]).blocks(streams[s_move]);
					} else evolve_nonlinear(beta * gconf.dt * (!k && i ? 2. : 1.) * symplectic_c[k], streams[s_move]);
					cufftExecZ2Z(fft_elvolve_psi ?: fft_plain, gres.shard_gpu, gres.shard_gpu, CUFFT_FORWARD) && assertcufft;
					completion(streams[s_move]).blocks(streams[s_cb_nonlinear]);
					if (fft_elvolve_psik) {
						memset_device_object(callback::evolve_linear_table_idx, k, streams[s_cb_linear]);
						completion(streams[s_cb_linear]).blocks(streams[s_move]);
					} else evolve_linear(&evolve_linear_tables_all[k * gconf.chain_length], streams[s_move]);
					cufftExecZ2Z(fft_elvolve_psik ?: fft_plain, gres.shard_gpu, gres.shard_gpu, CUFFT_INVERSE) && assertcufft;
					completion(streams[s_move]).blocks(streams[s_cb_linear]);
				}
			completion finish_move = evolve_nonlinear(beta * gconf.dt * symplectic_c[7], streams[s_move]);
			finish_move.blocks(streams[s_linenergies]);
			finish_move.blocks(streams[s_dump]);

			loop_ctl += gconf.kernel_batching;
		}

		if (loop_ctl % gconf.dump_interval != 0) {
			dumper();
			completion(streams[s_linenergies]).wait();
			res.write_linenergies(loop_ctl);
			completion(streams[s_dump]).wait();
			res.write_shard(loop_ctl);
		}
		return 0;
	}
}

ginit = [] {
	::programs()["DNLS"] = DNLS::main;
};
