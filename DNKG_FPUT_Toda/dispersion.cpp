#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "DNKG_FPUT_Toda.hpp"

using namespace std;

namespace DNKG_FPUT_Toda {

	vector<double> dispersion(double m) {
		vector<double> omega(gconf.chain_length);
		loopk(gconf.chain_length) {
			auto s2 = 2 * sin(k * M_PI / gconf.chain_length);
			omega[k] = sqrt(m + s2 * s2);
		}
		return omega;
	}

}
