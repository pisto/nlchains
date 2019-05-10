#include "utilities.hpp"
#include "results.hpp"

using namespace std;

/*
 * This is the only compute-intensive CPU code (at least with large chain_length). With a modern GCC, it's possible to
 * speed this up by using automatic SIMD vectorization targeted to different instruction sets (with function multiversioning).
 */

make_simd_clones("default,avx,avx512f")
results &results::calc_linenergies(double norm_factor) {
	boost::mpi::all_reduce(mpi_global_results, gres.linenergies_host, gconf.chain_length, linenergies.data(), plus<double>());
	if (a0is0) linenergies[0] = 0;
	double linenergy_normalization = norm_factor / mpi_global_size;
	openmp_simd
	for (uint16_t i = 0; i < gconf.chain_length; i++) linenergies[i] *= linenergy_normalization;
	return *this;
}

make_simd_clones("default,avx,avx512f")
results &results::calc_entropies() {
	double totale = 0, totalloge = 0, totaleloge = 0, *modes;
	uint16_t modes_tot;
	if (entropy_modes_indices.empty()) {
		modes = linenergies.data() + a0is0;
		modes_tot = gconf.chain_length - a0is0;
	} else {
		loopi(entropy_modes.size()) entropy_modes[i] = linenergies[entropy_modes_indices[i]];
		modes = entropy_modes.data();
		modes_tot = entropy_modes.size();
	}
	openmp_simd
	for (uint16_t i = 0; i < modes_tot; i++) {
		auto e = modes[i];
		totale += e;
		auto loge = log(e);     //should generate a call to libmvec.so
		totalloge += loge;
		totaleloge += e * loge;
	}
	double normalization = modes_tot / totale, lognormalization = log(normalization);
	//{ sum(log(e'(k))), sum(e'(k)log(e'(k))) }, with e'(k) = total_modes/linear_energy_total*linear_energy(k)
	WTentropy = -(totalloge + modes_tot * lognormalization);
	INFentropy = normalization * (totaleloge + lognormalization * totale);
	return *this;
}
