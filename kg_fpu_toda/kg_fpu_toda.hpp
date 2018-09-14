#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "../utilities_cuda.cuh"

namespace kg_fpu_toda {
	enum Model {
		KG = 0, FPU, Toda
	};
	extern __constant__ double dt_c[8], dt_d[8], m, alpha, beta;
	extern __constant__ double alpha2, alpha2_inv;

	completion make_linenergies(const double2* fft_phis, const double2* fft_pis, const double* omegas, cudaStream_t stream = 0);
	template<Model model>
	completion move(plane2split*& splitter, cudaStream_t stream);

}
