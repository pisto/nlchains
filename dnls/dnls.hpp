#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include "../utilities_cuda.cuh"

namespace dnls {

	completion evolve_nonlinear(double beta_dt_symplectic, cudaStream_t stream);
	completion evolve_linear(const cufftDoubleComplex* evolve_linear_table, cudaStream_t stream);
	completion make_linenergies(const cufftDoubleComplex* psis_k, const double* omega, cudaStream_t stream);

}
