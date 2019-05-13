#include <iostream>
#include <fstream>
#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "../common/loop_control.hpp"
#include "dDNKG.hpp"

using namespace std;

/*
 * Nonlinear discrete Klein-Gordon with a mass parameter specific to every site.
 *
 * The linear eigensystem is solved, and the linear energies are calculated through an explicit projection on the
 * eigenstates: such operation is implemented as a matrix multiplication. This is pretty much the only difference
 * between this solver and DNKG_FPUT_Toda.
 */

namespace dDNKG {

	int main(int argc, char *argv[]) {

		bool split_kernel = false;
		arma::vec mp2_host;
		double beta;
		{
			using namespace boost::program_options;
			string m_fname;
			parse_cmdline parser("Options for "s + argv[0]);
			parser.options.add_options()
					(",m", value(&m_fname)->required(), "linear parameter m filename")
					("beta", value(&beta)->required(), "fourth order nonlinearity")
					("split_kernel", "force use of split kernel");
			parser.run(argc, argv);
			split_kernel = parser.vm.count("split_kernel");

			mp2_host.resize(gconf.chain_length);
			try {
				ifstream ms(m_fname);
				ms.exceptions(ios::failbit | ios::badbit | ios::eofbit);
				ms.read((char *) mp2_host.memptr(), gconf.chain_length * sizeof(double));
				//XXX see comment on catching ios_base::failure in common/main.cpp
			} catch (const ios_base::failure &e) {
				throw ios::failure("could not read m file ("s + e.what() + ")", e.code());
			}
			mp2_host += 2;
		}
		auto ctx = cuda_ctx.activate(mpi_node_coord);

		cudalist<double> mp2(gconf.chain_length),
				eigenvectors(size_t(gconf.chain_length) * gconf.chain_length),
				eigenvectors_times_omega(size_t(gconf.chain_length) * gconf.chain_length),
				projection_phi(2 * size_t(gconf.shard_copies) * gconf.chain_length);
		auto projection_pi = &projection_phi[size_t(gconf.shard_copies) * gconf.chain_length];
		{
			arma::mat eigenvectors_host;
			arma::vec omegas;
			eigensystem(mp2_host, eigenvectors_host, omegas);
			auto vecsize = gconf.chain_length * sizeof(double), matsize =
					size_t(gconf.chain_length) * gconf.chain_length * sizeof(double);
			cudaMemcpy(mp2, mp2_host.memptr(), vecsize, cudaMemcpyHostToDevice) && assertcu;
			cudaMemcpy(get_device_address(dDNKG::mp2), mp2, min(sizeof(dDNKG::mp2), vecsize),
			           cudaMemcpyDeviceToDevice) && assertcu;

			cudaMemcpy(eigenvectors, eigenvectors_host.memptr(), matsize, cudaMemcpyHostToDevice) && assertcu;
			loopi(gconf.chain_length) eigenvectors_host.col(i) *= omegas[i];
			cudaMemcpy(eigenvectors_times_omega, eigenvectors_host.memptr(), matsize, cudaMemcpyHostToDevice) &&
			assertcu;
		}

		/*
		 * The projection over the eigenvectors is implemented as a matrix multiplication (cublasDgemm).
		 * The shard is put into split format (phi/pi, chain, copy), and matrix-multiplied with the eigenvectors
		 * matrix. Note that this works because cuBLAS uses column-major indexing, so it sees the transposed shard
		 * matrices as (phi/pi, copy, chain)-indexed. The eigenvectors matrix used with the phi shard is multiplied
		 * by omega already, that would be otherwise an adjustement to be done after the squaring of the projections.
		 */
		cublasHandle_t cublas = 0;
		destructor([&] { cublasDestroy(cublas); });
		cublasCreate(&cublas) && assertcublas;

		results res(false);

		enum {
			s_move = 0, s_dump, s_entropy, s_entropy_aux, s_results, s_total
		};
		cudaStream_t streams[s_total];
		memset(streams, 0, sizeof(streams));
		destructor([&] { for (auto stream : streams) cudaStreamDestroy(stream); });
		for (auto &stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) && assertcu;

		//constants
		double dt_c_host[8], dt_d_host[8];
		loopi(8) dt_c_host[i] = symplectic_c[i] * gconf.dt, dt_d_host[i] = symplectic_d[i] * gconf.dt;
		set_device_object(dt_c_host, dt_c);
		set_device_object(dt_d_host, dt_d);
		set_device_object(beta, dDNKG::beta);

		plane2split splitter;
		splitter.split(streams[s_move]);

		loop_control_gpu loop_ctl(streams[s_move]);
		auto dumper = [&] {
			if (split_kernel) splitter.plane(streams[s_move], streams[s_dump]);
			cudaMemcpyAsync(gres.shard_host, gres.shard_gpu, gconf.sizeof_shard, cudaMemcpyDeviceToHost, streams[s_dump]) &&
			assertcu;
			completion done_copy(streams[s_dump]);
			if (!split_kernel) done_copy.blocks(streams[s_move]);
			done_copy.blocks(streams[s_results]);
		};
		destructor(cudaDeviceSynchronize);
		while (1) {
			bool full_dump = loop_ctl % gconf.dump_interval == 0;
			if (full_dump) dumper();
			if (!split_kernel) splitter.split(streams[s_move], streams[s_entropy]);
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			double one = 1, zero = 0;
			cublasSetStream(cublas, streams[s_entropy]) && assertcublas;
			cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			            gconf.shard_copies, gconf.chain_length, gconf.chain_length,
			            &one, splitter.coords_transposed, gconf.shard_copies,
			            eigenvectors_times_omega, gconf.chain_length,
			            &zero, projection_phi, gconf.shard_copies) && assertcublas;
			cublasSetStream(cublas, streams[s_entropy_aux]) && assertcublas;
			cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			            gconf.shard_copies, gconf.chain_length, gconf.chain_length,
			            &one, splitter.momenta_transposed, gconf.shard_copies,
			            eigenvectors, gconf.chain_length,
			            &zero, projection_pi, gconf.shard_copies) && assertcublas;
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);
			if (split_kernel) completion(streams[s_entropy]).blocks(streams[s_move]);
			make_linenergies(projection_phi, projection_pi, streams[s_entropy]);
			completion(streams[s_entropy]).blocks(streams[s_results]);
			add_cuda_callback(streams[s_results], loop_ctl.callback_err,
			                  [&, full_dump, t = *loop_ctl](cudaError_t status) {
				                  if (loop_ctl.callback_err) return;
				                  status && assertcu;
				                  res.calc_linenergies(0.5 / gconf.shard_copies).calc_entropies().check_entropy().write_entropy(t);
				                  if (full_dump) res.write_linenergies(t).write_shard(t);
			                  });
			completion done_results(streams[s_results]);
			done_results.blocks(streams[s_dump]);
			done_results.blocks(streams[s_entropy]);

			if (loop_ctl.break_now()) break;

			completion finish_move = move(splitter, split_kernel, mp2, streams[s_move]);
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
	::programs()["dDNKG"] = dDNKG::main;
};
