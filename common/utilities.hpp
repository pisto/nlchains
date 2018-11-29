#pragma once

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
