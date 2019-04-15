#pragma once

/*
 * Some utilities to cleanup code.
 */

#include <cstdint>

//looping
#define loop(v, m)		for(size_t v = 0; v < size_t(m); v++)
#define loopi(m)		loop(i, m)
#define loopj(m)		loop(j, m)
#define loopk(m)		loop(k, m)

/*
 * Fake destructor for C resources, to cope with exceptions. E.g.:
 * void* array = malloc(123);
 * destructor([=]{ free(array); });
 */

#define destructor(f) destructor_helper_macro_1(f, __LINE__)

#include <utility>

template<typename F>
struct destructor_helper {
	F f;

	~destructor_helper() { f(); }
};

template<typename F>
destructor_helper<F> make_destructor_helper(F &&f) {
	return destructor_helper<F>{std::move(f)};
}

#define destructor_helper_macro_2(f, l) auto destructor_ ## l = make_destructor_helper(f)
#define destructor_helper_macro_1(f, l) destructor_helper_macro_2(f, l)

/*
 * Global initialization with a function:
 * ginit = []{ ... };
 */

#define ginit ginit_helper_macro_1(__LINE__)

struct ginit_helper {
	template<typename F>
	ginit_helper(F &&f) { f(); }
};

#define ginit_helper_macro_2(l) static ginit_helper ginit_ ## l __attribute__((used))
#define ginit_helper_macro_1(l) ginit_helper_macro_2(l)

/*
 * CPP test flags for SIMD
 */

#if defined(_OPENMP) && _OPENMP >= 201307
#define openmp_simd _Pragma("omp simd")
#else
#define openmp_simd
#endif

#if defined(__GNUC__) && __GNUC__ >= 6
#define make_simd_clones(x) [[gnu::target_clones(x)]]
#else
#define make_simd_clones(x)
#endif

/*
 * Collect writes to an std::ostream and flush them all together. Useful when sending output
 * to stdout/stderr in parallel program where writes should be "atomic".
 */

#include <sstream>
#include <iostream>

struct collect_ostream : std::ostringstream {
	std::ostream &out;

	collect_ostream(std::ostream &out) : out(out) {}

	~collect_ostream() { out << str(); }

	template<typename T>
	std::ostringstream &operator<<(T &&t) const {
		return const_cast<collect_ostream &>(*this) << std::forward<T>(t);
	}

};

/*
 * Alignment stuff.
 */

#include <boost/align/aligned_allocator.hpp>
#include <boost/version.hpp>
#if defined(__GNUC__) && BOOST_VERSION < 106100
//XXX type error in older versions of boost
#undef BOOST_ALIGN_ASSUME_ALIGNED
#define BOOST_ALIGN_ASSUME_ALIGNED(p, n) \
(p) = static_cast<__typeof__(p)>(__builtin_assume_aligned((p), (n)))
#else
#include <boost/align/assume_aligned.hpp>
#endif

template<typename T> using simd_allocator = boost::alignment::aligned_allocator<T, 64>;
