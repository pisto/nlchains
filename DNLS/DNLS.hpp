#pragma once

#include <vector>
#include <complex>
#include <boost/multi_array.hpp>
#include <cufft.h>
#include <cufftXt.h>
#include "../common/utilities_cuda.cuh"

namespace DNLS {

	std::vector<double> dispersion();

	boost::multi_array<std::complex<double>, 2> evolve_linear_table();

	completion evolve_nonlinear(double beta_dt_symplectic, cudaStream_t stream);

	completion evolve_linear(const cufftDoubleComplex *evolve_linear_table, cudaStream_t stream);

	completion make_linenergies(const cufftDoubleComplex *psis_k, const double *omega, cudaStream_t stream);

	namespace callback {
		extern __constant__ uint16_t chainlen;
		extern __constant__ uint8_t evolve_linear_table_idx;
		extern __constant__ double beta_dt_symplectic;
		extern __device__ const cufftCallbackLoadZ evolve_linear_ptr, evolve_nonlinear_ptr;
	}

}
