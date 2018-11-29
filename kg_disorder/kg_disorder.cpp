#include <iostream>
#include <fstream>
#include <armadillo>
#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/results.hpp"
#include "../common/symplectic.hpp"
#include "kg_disorder.hpp"

using namespace std;

namespace kg_disorder {

	int main(int argc, char *argv[]) {

		bool use_split_kernel = false;
		arma::vec mp2_host;
		double beta;
		{
			using namespace boost::program_options;
			string m_fname;
			parse_cmdline parser("Options for "s + argv[0]);
			parser.options.add_options()
					(",m", value(&m_fname)->required(), "linear parameter m filename")
					("beta", value(&beta)->required(), "fourth order nonlinearity")
					("split-kernel", "force use of split kernel");
			parser(argc, argv);
			use_split_kernel = parser.vm.count("split-kernel");

			mp2_host.resize(gconf.chain_length);
			try {
				ifstream ms(m_fname);
				ms.exceptions(ios::failbit | ios::badbit | ios::eofbit);
				ms.read((char *) mp2_host.memptr(), gconf.chain_length * sizeof(double));
			} catch (const ios_base::failure &e) {
				throw ios::failure("could not read m file ("s + e.what() + ")", e.code());
			}
			mp2_host += 2;
		}

		cudalist<double> mp2(gconf.chain_length),
				eigenvectors(size_t(gconf.chain_length) * gconf.chain_length),
				eigenvectors_times_omega(size_t(gconf.chain_length) * gconf.chain_length),
				projection_phi(2 * size_t(gconf.shard_copies) * gconf.chain_length);
		auto projection_pi = &projection_phi[size_t(gconf.shard_copies) * gconf.chain_length];
		{
			auto vecsize = gconf.chain_length * sizeof(double), matsize =
					size_t(gconf.chain_length) * gconf.chain_length * sizeof(double);
			cudaMemcpy(mp2, mp2_host.memptr(), vecsize, cudaMemcpyHostToDevice) && assertcu;
			cudaMemcpy(get_device_address(kg_disorder::mp2), mp2, min(sizeof(kg_disorder::mp2), vecsize),
			           cudaMemcpyDeviceToDevice) && assertcu;

			arma::mat interaction = diagmat(mp2_host), eigenvectors_host;
			interaction.diag(1).fill(-1);
			interaction.diag(-1).fill(-1);
			interaction(0, gconf.chain_length - 1) = interaction(gconf.chain_length - 1, 0) = -1;
			arma::vec omegas;
			if (!arma::eig_sym(omegas, eigenvectors_host, interaction))
				throw runtime_error("Cannot calculate eigensystem!");
			omegas = sqrt(omegas);

			if (!mpi_global_coord) {
				ofstream dump_eigensystem(gconf.dump_prefix + "-omegas");
				dump_eigensystem.exceptions(ios::failbit | ios::badbit | ios::eofbit);
				dump_eigensystem.write((char *) omegas.memptr(), vecsize);
				dump_eigensystem.close();
				dump_eigensystem.open(gconf.dump_prefix + "-eigenvectors");
				dump_eigensystem.write((char *) eigenvectors_host.memptr(), matsize);
			}
			cudaMemcpy(eigenvectors, eigenvectors_host.memptr(), matsize, cudaMemcpyHostToDevice) && assertcu;
			loopi(gconf.chain_length) eigenvectors_host.col(i) *= omegas[i];
			cudaMemcpy(eigenvectors_times_omega, eigenvectors_host.memptr(), matsize, cudaMemcpyHostToDevice) &&
			assertcu;
		}

		cublasHandle_t cublas = 0;
		destructor([&] { cublasDestroy(cublas); });
		cublasCreate(&cublas) && assertcublas;

		results res(false);

		enum {
			s_move = 0, s_dump, s_entropy, s_entropy_aux, s_total
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
		set_device_object(beta, kg_disorder::beta);

		plane2split splitter(gconf.chain_length, gconf.shard_copies);
		splitter.split(gres.shard, streams[s_move]);

		exception_ptr callback_err;
		completion throttle;
		uint64_t t = gconf.timebase;
		auto dumper = [&] {
			if (use_split_kernel) splitter.plane(gres.shard, streams[s_move], streams[s_dump]);
			cudaMemcpyAsync(gres.shard_host, gres.shard, gconf.shard_size, cudaMemcpyDeviceToHost, streams[s_dump]) &&
			assertcu;
			if (!use_split_kernel) completion(streams[s_dump]).blocks(streams[s_move]);
			add_cuda_callback(streams[s_dump], callback_err, [&, t](cudaError_t status) {
				if (callback_err) return;
				status && assertcu;
				res.write_shard(t, gres.shard_host);
			});
		};
		destructor(cudaDeviceSynchronize);
		for (bool unified_max_step = false, throttleswitch = true;; t += gconf.steps_grouping, throttleswitch ^= true) {
			//throttle enqueuing of kernels
			if (throttleswitch) throttle = completion(streams[s_move]);
			else throttle.wait();

			if (t % gconf.steps_grouping_dump == 0) dumper();
			if (!use_split_kernel) splitter.split(gres.shard, streams[s_move], streams[s_entropy]);
			completion(streams[s_entropy]).blocks(streams[s_entropy_aux]);
			double one = 1, zero = 0;
			cublasSetStream(cublas, streams[s_entropy]) && assertcublas;
			cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			            gconf.shard_copies, gconf.chain_length, gconf.chain_length,
			            &one, splitter.real_transposed, gconf.shard_copies,
			            eigenvectors_times_omega, gconf.chain_length,
			            &zero, projection_phi, gconf.shard_copies) && assertcublas;
			cublasSetStream(cublas, streams[s_entropy_aux]) && assertcublas;
			cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			            gconf.shard_copies, gconf.chain_length, gconf.chain_length,
			            &one, splitter.img_transposed, gconf.shard_copies,
			            eigenvectors, gconf.chain_length,
			            &zero, projection_pi, gconf.shard_copies) && assertcublas;
			completion(streams[s_entropy_aux]).blocks(streams[s_entropy]);
			if (use_split_kernel) completion(streams[s_entropy]).blocks(streams[s_move]);
			make_linenergies(projection_phi, projection_pi, streams[s_entropy]);
			add_cuda_callback(streams[s_entropy], callback_err, [&, t](cudaError_t status) {
				if (callback_err) return;
				status && assertcu;
				auto entropies = res.entropies(gres.linenergies_host, 0.5 / gconf.shard_copies);
				res.check_entropy(entropies);
				if (t % gconf.steps_grouping_dump == 0) res.write_linenergies(t);
				res.write_entropy(t, entropies);
			});

			//termination checks
			if (callback_err) rethrow_exception(callback_err);
			if (quit_requested && !unified_max_step) {
				boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
				unified_max_step = true;
			}
			if (gconf.steps && gconf.steps == t) {
				//make sure all MPI calls are matched, use_split_kernel
				if (!unified_max_step)
					boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
				break;
			}

			completion finish_move = move(splitter, use_split_kernel, mp2, streams[s_move]);
			finish_move.blocks(streams[s_entropy]);
			finish_move.blocks(streams[s_dump]);

		}
		if (t % gconf.steps_grouping_dump != 0) {
			dumper();
			completion(streams[s_entropy]).wait();
			res.write_linenergies(t);
		}
		return 0;
	}
}

ginit = [] {
	::programs()["KleinGordon-disorder"] = kg_disorder::main;
};
