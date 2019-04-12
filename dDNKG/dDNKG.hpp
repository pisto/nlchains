#pragma once

#include <armadillo>
#include "../common/utilities_cuda.cuh"

namespace dDNKG {

	void eigensystem(const arma::vec &mp2, arma::mat &eigenvectors, arma::vec &omegas);

	extern __constant__ double dt_c[8], dt_d[8], mp2[2048], beta;

	completion move(plane2split &splitter, bool &use_split_kernel, const double *mp2_gmem, cudaStream_t stream);

	completion make_linenergies(const double *projection_phi, const double *projection_pi, cudaStream_t stream);

}
