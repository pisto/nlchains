#include <vector>
#include "../../common/utilities.hpp"
#include "../../common/configuration.hpp"
#include "../../common/results.hpp"
#include "../../common/symplectic.hpp"
#include "../../common/loop_control.hpp"
#include "../dDNKG.hpp"

using namespace std;

namespace dDNKG {
	namespace cpu {

		namespace {
			double beta;
		}
		[[gnu::always_inline]] double rhs(double left, double center, double right, double mp2) {
			return -(center * (mp2 + beta * center * center) - left - right);
		}

		make_simd_clones("default,avx,avx512f")
		int main(int argc, char *argv[]) {

			arma::vec mp2_arma;
			{
				using namespace boost::program_options;
				string m_fname;
				parse_cmdline parser("Options for "s + argv[0]);
				parser.options.add_options()
						(",m", value(&m_fname)->required(), "linear parameter m filename")
						("beta", value(&beta)->required(), "fourth order nonlinearity");
				parser(argc, argv);

				mp2_arma.resize(gconf.chain_length);
				try {
					ifstream ms(m_fname);
					ms.exceptions(ios::failbit | ios::badbit | ios::eofbit);
					ms.read((char *) mp2_arma.memptr(), gconf.chain_length * sizeof(double));
					//XXX see comment on catching ios_base::failure in common/main.cpp
				} catch (const ios_base::failure &e) {
					throw ios::failure("could not read m file ("s + e.what() + ")", e.code());
				}
				mp2_arma += 2;
			}

			arma::mat eigenvectors;
			arma::vec omegas;
			eigensystem(mp2_arma, eigenvectors, omegas);
			vector<double, simd_allocator<double>> mp2_buff(mp2_arma.memptr(), mp2_arma.memptr() + gconf.chain_length);
			double *__restrict__ mp2 = mp2_buff.data();
			BOOST_ALIGN_ASSUME_ALIGNED(mp2, 64);

			results res(false);

			double dt_c[8], dt_d[8];
			loopi(8) dt_c[i] = symplectic_c[i] * gconf.dt, dt_d[i] = symplectic_d[i] * gconf.dt;

			auto accumulate_linenergies = [&](size_t c) {
				auto chain = &gres.shard_host[c * gconf.chain_length];
				auto eigen = eigenvectors.memptr();
				loopi(gconf.chain_length) {
					double dotphi = 0, dotpi = 0;
					for (uint16_t i = 0; i < gconf.chain_length; i++, eigen++) {
						auto phipi = chain[i];
						dotphi += phipi.x * (*eigen), dotpi += phipi.y * (*eigen);
					}
					auto omega_dotphi = omegas[i] * dotphi;
					gres.linenergies_host[i] += omega_dotphi * omega_dotphi + dotpi * dotpi;
				}
			};
			//loop expects linenergies to be accumulated during move phase
			loopi(gconf.shard_copies) accumulate_linenergies(i);

			vector<double, simd_allocator<double>> phis_buff(gconf.chain_length + 2), pis_buff(gconf.chain_length);
			double *__restrict__ phi = phis_buff.data(), *__restrict__ pi = pis_buff.data();
			BOOST_ALIGN_ASSUME_ALIGNED(phi, 64);
			BOOST_ALIGN_ASSUME_ALIGNED(pi, 64);

			loop_control loop_ctl(gconf.time_offset);
			while (1) {
				auto entropies = res.entropies(gres.linenergies_host, 0.5 / gconf.shard_copies);
				res.check_entropy(entropies);
				res.write_entropy(loop_ctl, entropies);
				if (loop_ctl % gconf.dump_interval == 0) {
					res.write_shard(loop_ctl, gres.shard_host);
					res.write_linenergies(loop_ctl);
				}

				if (loop_ctl.break_now()) break;

				loopi(gconf.chain_length) gres.linenergies_host[i] = 0;
				for (int c = 0; c < gconf.shard_copies; c++) {
					auto planar = &gres.shard_host[c * gconf.chain_length];
					for (int i_0 = 0, i = 1; i_0 < gconf.chain_length; i_0++, i++)
						phi[i] = planar[i_0].x, pi[i_0] = planar[i_0].y;
					for (auto i = 0; i < gconf.kernel_batching; i++) {
						for (int k = 0; k < 7; k++) {
							double dt_c_k = dt_c[k];
							if (i && !k)
								dt_c_k *= 2;
							openmp_simd
							for (uint16_t i = 0; i < gconf.chain_length; i++)
								phi[i + 1] += dt_c_k * pi[i];
							phi[0] = phi[gconf.chain_length], phi[gconf.chain_length + 1] = phi[1];
							openmp_simd
							for (uint16_t i = 0; i < gconf.chain_length; i++)
								pi[i] += dt_d[k] * rhs(phi[i], phi[i + 1], phi[i + 2], mp2[i]);
						}
					}
					for (int i_0 = 0, i = 1; i_0 < gconf.chain_length; i_0++, i++) {
						phi[i] += dt_c[7] * pi[i_0];
						planar[i_0] = {phi[i], pi[i_0]};
					}
					accumulate_linenergies(c);
				}

				loop_ctl += gconf.kernel_batching;
			}

			if (loop_ctl % gconf.dump_interval != 0) {
				res.write_shard(loop_ctl, gres.shard_host);
				res.write_linenergies(loop_ctl);
			}
			return 0;

		}

	}
}

ginit = [] {
	::programs()["dDNKG-cpu"] = dDNKG::cpu::main;
};
