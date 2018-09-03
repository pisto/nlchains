#include "results.hpp"

using namespace std;

/*
 * This is the only compute-intensive CPU code (at least with large chain_length). With a modern GCC, it's possible to
 * speed this up by using automatic SIMD vectorization targeted to different instruction sets (with function multiversioning).
 */

#if defined(__GNUC__) && __GNUC__ >= 6
[[gnu::target_clones("default,avx,avx512f")]]
#endif
array<double, 2> results::entropies(const double* shard_linenergies, double norm_factor){
	//XXX the in-place version always generates an assertion?
	boost::mpi::all_reduce(mpi_global, shard_linenergies, gconf.chain_length, linenergies.data(), plus<double>());

	auto entropy_n = gconf.chain_length - a0is0;
	double totale = 0, totalloge = 0, totaleloge = 0, linenergy_normalization = norm_factor / mpi_global.size();
	if(a0is0) linenergies[0] = 0;
	//this helps g++ vectorize the following loop, maybe other compilers as well
	#if defined(_OPENMP) && _OPENMP >= 201307
	#pragma omp simd
	#endif
	for(uint16_t i = 0; i < entropy_n; i++){
		auto e = (linenergies[i + a0is0] *= linenergy_normalization);
		totale += e;
		auto loge = log(e);     //should generate a call to libmvec.so
		totalloge += loge;
		totaleloge += e * loge;
	}
	double normalization = entropy_n / totale, lognormalization = log(normalization);
	return { -(totalloge + entropy_n * lognormalization), normalization * (totaleloge + lognormalization * totale) };
}
