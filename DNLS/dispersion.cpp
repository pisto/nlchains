#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/symplectic.hpp"
#include "DNLS.hpp"

using namespace std;

namespace DNLS {

	vector<double> dispersion() {
		vector<double> omega(gconf.chain_length);
		loopk(gconf.chain_length) {
			auto s2 = 2 * sin(k * M_PI / gconf.chain_length);
			omega[k] = s2 * s2;
		}
		return omega;
	}

	boost::multi_array<complex<double>, 2> evolve_linear_table() {
		auto omega = dispersion();
		boost::multi_array<complex<double>, 2> table(boost::extents[7][gconf.chain_length]);
		auto normalization = 1. / gconf.chain_length;
		complex<double> complexdt = -1i * gconf.dt;
		loopi(7) loopj(gconf.chain_length) table[i][j] = exp(complexdt * symplectic_d[i] * omega[j]) * normalization;
		return table;
	}

}
