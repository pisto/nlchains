#include "utilities.hpp"
#include "results.hpp"

using namespace std;

/*
 * This is the only compute-intensive CPU code (at least with large chain_length). With a modern GCC, it's possible to
 * speed this up by using automatic SIMD vectorization targeted to different instruction sets (with function multiversioning).
 */

#if defined(__GNUC__) && __GNUC__ >= 6
[[gnu::target_clones("default,avx,avx512f")]]
#endif
array<double, 2> results::entropies(const double *shard_linenergies, double norm_factor) {
	//XXX the in-place version always generates an assertion?
	boost::mpi::all_reduce(mpi_global_results, shard_linenergies, gconf.chain_length, linenergies.data(), plus<double>());

	double totale = 0, totalloge = 0, totaleloge = 0, linenergy_normalization = norm_factor / mpi_global_size;
	if (a0is0) linenergies[0] = 0;
	//this helps g++ vectorize the following loops, maybe other compilers as well
	#if defined(_OPENMP) && _OPENMP >= 201307
	#pragma omp simd
	#endif
	for (uint16_t i = 0; i < gconf.chain_length; i++) linenergies[i] *= linenergy_normalization;
	double *modes;
	uint16_t modes_tot;
	if (entropy_modes_indices.empty()) {
		modes = linenergies.data() + a0is0;
		modes_tot = gconf.chain_length - a0is0;
	} else {
		if (entropy_modes.empty()) entropy_modes.resize(entropy_modes_indices.size());
		loopi(entropy_modes.size()) entropy_modes[i] = linenergies[entropy_modes_indices[i]];
		modes = entropy_modes.data();
		modes_tot = entropy_modes.size();
	}
	#if defined(_OPENMP) && _OPENMP >= 201307
	#pragma omp simd
	#endif
	for (uint16_t i = 0; i < modes_tot; i++) {
		auto e = modes[i];
		totale += e;
		auto loge = log(e);     //should generate a call to libmvec.so
		totalloge += loge;
		totaleloge += e * loge;
	}
	double normalization = modes_tot / totale, lognormalization = log(normalization);
	//{ sum(log(e'(k))), sum(e'(k)log(e'(k))) }, with e'(k) = total_modes/linear_energy_total*linear_energy(k)
	return {-(totalloge + modes_tot * lognormalization), normalization * (totaleloge + lognormalization * totale)};
}
