#include <cmath>
#include <vector>
#include <fftw3.h>
#include "../../common/utilities.hpp"
#include "../../common/configuration.hpp"
#include "../../common/results.hpp"
#include "../../common/symplectic.hpp"
#include "../../common/loop_control.hpp"
#include "../../common/wisdom_sync.hpp"
#include "../DNLS.hpp"

using namespace std;

namespace DNLS {
	namespace cpu {

		namespace {
			//XXX this is necessary because std::complex doesn't compile to SIMD
			[[gnu::always_inline]] void split_complex_mul(double &r1, double &i1, double r2, double i2) {
				double r3 = r1 * r2 - i1 * i2, i3 = i2 * r1 + i1 * r2;
				r1 = r3, i1 = i3;
			}
		}

		make_simd_clones("default,avx,avx512f")
		int main(int argc, char *argv[]) {

			double beta;
			wisdom_sync wsync;
			{
				using namespace boost::program_options;
				parse_cmdline parser("Options for "s + argv[0]);
				parser.options.add_options()("beta", value(&beta)->required(), "fourth order nonlinearity");
				wsync.add_options(parser.options);
				parser.run(argc, argv);
			}

			auto omega = dispersion();
			vector<complex<double>, simd_allocator<complex<double>>> evolve_linear_tables[7];
			{
				auto linear_tables_unaligned = DNLS::evolve_linear_table();
				loopi(7) evolve_linear_tables[i].assign(linear_tables_unaligned[i].begin(),
				                                        linear_tables_unaligned[i].end());
			}
			vector<double> beta_dt_symplectic(8);
			loopk(8) beta_dt_symplectic[k] = beta * gconf.dt * (k == 7 ? 2. : 1.) * symplectic_c[k];

			results res(true);

			vector<double, simd_allocator<double>> psi_r_buff(gconf.chain_length), psi_i_buff(gconf.chain_length);
			double *__restrict__ psi_r = psi_r_buff.data();
			double *__restrict__ psi_i = psi_i_buff.data();
			BOOST_ALIGN_ASSUME_ALIGNED(psi_r, 64);
			BOOST_ALIGN_ASSUME_ALIGNED(psi_i, 64);

			destructor(fftw_cleanup);
			wsync.gather();
			fftw_iodim fft_dim{ gconf.chain_length, 1, 1 };
			auto fft = fftw_plan_guru_split_dft(1, &fft_dim, 0, 0, psi_r, psi_i, psi_r, psi_i, FFTW_EXHAUSTIVE | wsync.fftw_flags);
			//XXX fftw_execute_split_dft(fft, psi_i, psi_r, psi_i, psi_r) should do an inverse transform but it returns garbage
			auto fft_back = fftw_plan_guru_split_dft(1, &fft_dim, 0, 0, psi_i, psi_r, psi_i, psi_r, FFTW_EXHAUSTIVE | wsync.fftw_flags);
			destructor([=] { fftw_destroy_plan(fft); fftw_destroy_plan(fft_back); });
			if (!fft || !fft_back) throw runtime_error("Cannot create FFTW3 plan");
			wsync.scatter();

			auto accumulate_linenergies = [&](size_t c) {
				auto planar = &gres.shard_host[c * gconf.chain_length];
				loopi(gconf.chain_length) psi_r[i] = planar[i].x, psi_i[i] = planar[i].y;
				fftw_execute(fft);
				loopi(gconf.chain_length) gres.linenergies_host[i] += psi_r[i] * psi_r[i] + psi_i[i] * psi_i[i];
			};
			//loop expects linenergies to be accumulated during move phase
			loopi(gconf.shard_copies) accumulate_linenergies(i);

			auto evolve_nonlinear = [&](double beta_dt_symplectic)[[gnu::always_inline]]{
				openmp_simd
				for (uint16_t i = 0; i < gconf.chain_length; i++) {
					auto norm = psi_r[i] * psi_r[i] + psi_i[i] * psi_i[i];
					double exp_r, exp_i, angle = -beta_dt_symplectic * norm;
					#if 0
					//XXX sincos is not being vectorized by GCC as of version 8.3, I don't know why
					sincos(angle, &exp_i, &exp_r);
					#else
					//best-effort workaround
					exp_r = cos(angle);                                 //vectorized
						#if 0
						//naive
						exp_i = sin(angle);                             //vectorized
						#else
						//This appears faster because of the sqrt opcode
						exp_i = sqrt(1 - exp_r * exp_r);                //vectorized (native VSQRTPD), exp_i = abs(sin(angle))
						//branchless sign fix
						auto invert_sign = int32_t(angle / M_PI) & 1;   //0 -> even half-circles,  1 -> odd half-circles
						exp_i *= (1. - 2. * invert_sign) * copysign(1., angle);
						#endif
					#endif
					split_complex_mul(psi_r[i], psi_i[i], exp_r, exp_i);
				}
			};

			loop_control loop_ctl;
			while (1) {
				loopi(gconf.chain_length) gres.linenergies_host[i] *= omega[i];
				res.calc_linenergies(1. / gconf.shard_elements).calc_entropies().check_entropy().write_entropy(loop_ctl);
				if (loop_ctl % gconf.dump_interval == 0) res.write_linenergies(loop_ctl).write_shard(loop_ctl);

				if (loop_ctl.break_now()) break;

				loopi(gconf.chain_length) gres.linenergies_host[i] = 0;
				for (int c = 0; c < gconf.shard_copies; c++) {
					auto planar = &gres.shard_host[c * gconf.chain_length];
					loopi(gconf.chain_length) psi_r[i] = planar[i].x, psi_i[i] = planar[i].y;
					for (uint32_t i = 0; i < gconf.kernel_batching; i++)
						for (int k = 0; k < 7; k++) {
							evolve_nonlinear(beta * gconf.dt * (!k && i ? 2. : 1.) * symplectic_c[k]);
							fftw_execute(fft);
							complex<double> * __restrict__ table = evolve_linear_tables[k].data();
							BOOST_ALIGN_ASSUME_ALIGNED(table, 64);
							openmp_simd
							for (uint16_t i = 0; i < gconf.chain_length; i++)
								split_complex_mul(psi_r[i], psi_i[i], table[i].real(), table[i].imag());
							fftw_execute(fft_back);
						}
					evolve_nonlinear(beta * gconf.dt * symplectic_c[7]);

					loopi(gconf.chain_length) planar[i] = { psi_r[i], psi_i[i] };
					accumulate_linenergies(c);
				}

				loop_ctl += gconf.kernel_batching;
			}

			if (loop_ctl % gconf.dump_interval != 0) res.write_linenergies(loop_ctl).write_shard(loop_ctl);
			return 0;

		}

	}
}

ginit = [] {
	::programs()["DNLS-cpu"] = DNLS::cpu::main;
};
