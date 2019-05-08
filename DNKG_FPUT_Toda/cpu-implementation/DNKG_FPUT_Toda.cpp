#include <vector>
#include <fftw3.h>
#include "../../common/utilities.hpp"
#include "../../common/configuration.hpp"
#include "../../common/results.hpp"
#include "../../common/symplectic.hpp"
#include "../../common/loop_control.hpp"
#include "../../common/wisdom_sync.hpp"
#include "../DNKG_FPUT_Toda.hpp"

using namespace std;

namespace DNKG_FPUT_Toda {
	namespace cpu {

		namespace {
			double m, alpha, beta, alpha2, alpha2_inv;
		}

		template<Model m>
		double rhs(double left, double center, double right);

		template<>
		[[gnu::always_inline]] double rhs<DNKG>(double left, double center, double right) {
			return -(center * (m + 2 + beta * center * center) - left - right);
		}

		template<>
		[[gnu::always_inline]] double rhs<FPUT>(double left, double center, double right) {
			auto ret = (left + right - 2 * center);
			ret *= (1 + alpha * (right - left));
			auto dright = right - center, dleft = center - left;
			return ret + beta * (dright * dright * dright - dleft * dleft * dleft);
		}

		template<>
		[[gnu::always_inline]] double rhs<Toda>(double left, double center, double right) {
			//this should generate a call to the SIMD version of exp()
			return alpha2_inv * (exp(alpha2 * (right - center)) - exp(alpha2 * (center - left)));
		}

		template<Model model>
		make_simd_clones("default,avx,avx512f")
		int main(int argc, char *argv[]) {

			wisdom_sync wsync("none");
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
				wsync.add_options(parser.options);
				parser.run(argc, argv);
				if (model == Toda) alpha2 = 2 * alpha, alpha2_inv = 1 / alpha2;
			}

			vector<double> omega = dispersion(m);

			results res(m == 0.);

			destructor(fftw_cleanup);
			wsync.gather();
			auto fft_phis = fftw_alloc_complex(gconf.chain_length), fft_pis = fftw_alloc_complex(gconf.chain_length);
			destructor([=] { fftw_free(fft_phis); fftw_free(fft_pis); });
			if (!fft_phis || !fft_pis) throw bad_alloc();
			auto plan = fftw_plan_dft_1d(gconf.chain_length, fft_phis, fft_phis, FFTW_FORWARD, FFTW_EXHAUSTIVE | wsync.fftw_flags);
			destructor([=] { fftw_destroy_plan(plan); });
			if (!plan) throw runtime_error("Cannot create FFTW3 plan");
			wsync.scatter();

			double dt_c[8], dt_d[8];
			loopi(8) dt_c[i] = symplectic_c[i] * gconf.dt, dt_d[i] = symplectic_d[i] * gconf.dt;

			auto accumulate_linenergies = [&](size_t c) {
				auto chain = &gres.shard_host[c * gconf.chain_length];
				loopi(gconf.chain_length)
					fft_phis[i][0] = chain[i].x, fft_pis[i][0] = chain[i].y, fft_phis[i][1] = fft_pis[i][1] = 0;
				fftw_execute_dft(plan, fft_phis, fft_phis);
				fftw_execute_dft(plan, fft_pis, fft_pis);
				auto square = [](double x) { return x * x; };
				loopi(gconf.chain_length) {
					auto fft_phi = fft_phis[i], fft_pi = fft_pis[i];
					auto energy = (square(fft_phi[0]) + square(fft_phi[1])) * omega[i];
					energy += 2 * (fft_phi[1] * fft_pi[0] - fft_phi[0] * fft_pi[1]);
					energy *= omega[i];
					energy += square(fft_pi[0]) + square(fft_pi[1]);
					gres.linenergies_host[i] += energy;
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
				auto entropies = res.entropies(gres.linenergies_host, (0.5 / gconf.shard_copies) / gconf.chain_length);
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
								pi[i] += dt_d[k] * rhs<model>(phi[i], phi[i + 1], phi[i + 2]);
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
	using namespace DNKG_FPUT_Toda::cpu;
	auto &programs = ::programs();
	programs["DNKG-cpu"] = main<DNKG_FPUT_Toda::DNKG>;
	programs["FPUT-cpu"] = main<DNKG_FPUT_Toda::FPUT>;
	programs["Toda-cpu"] = main<DNKG_FPUT_Toda::Toda>;
};
