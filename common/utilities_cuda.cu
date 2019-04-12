#include "utilities_cuda.cuh"
#include "configuration.hpp"

cudaError cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, const void *func, size_t dynamicSMemSize,
                                             int blockSizeLimit) {
	return cudaOccupancyMaxPotentialBlockSize <const void *>(minGridSize, blockSize, func, dynamicSMemSize,
															 blockSizeLimit);
}

__global__ void split_kernel(uint32_t elements, const double2 *planar, double *real, double *img) {
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= elements) return;
	auto pair = planar[idx];
	real[idx] = pair.x, img[idx] = pair.y;
}

__global__ void unsplit_kernel(uint32_t elements, const double *real, const double *img, double2 *planar) {
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= elements) return;
	planar[idx] = {real[idx], img[idx]};
}

/*
 * The transposition is implemented with the cuBLAS operation cublasZgeam.
 */

plane2split::plane2split(uint16_t chainlen, uint16_t shard_copies) :
		chainlen(chainlen), shard_copies(shard_copies), elements(uint32_t(shard_copies) * chainlen),
		real_transposed(elements * 2), img_transposed((double *) real_transposed + elements),
		planar_transposed(elements) {
	cublasCreate(&handle) && assertcublas;
}

completion plane2split::split(const double2 *planar, cudaStream_t producer, cudaStream_t consumer) {
	cuDoubleComplex one{1., 0.}, zero{0., 0.};
	completion(producer).blocks(consumer);
	cublasSetStream(handle, consumer) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, shard_copies, chainlen, &one, planar, chainlen, &zero, 0,
	            shard_copies, planar_transposed, shard_copies) && assertcublas;
	completion(consumer).blocks(producer);
	static auto kinfo = make_kernel_info(split_kernel);
	auto linear_config = kinfo.linear_configuration(elements);
	split_kernel <<< linear_config.x, linear_config.y, 0, consumer >>>
	             (elements, planar_transposed, real_transposed, img_transposed);
	cudaGetLastError() && assertcu;
	return completion(consumer);
}

completion plane2split::plane(double2 *planar, cudaStream_t producer, cudaStream_t consumer) const {
	static auto kinfo = make_kernel_info(unsplit_kernel);
	auto linear_config = kinfo.linear_configuration(elements);
	completion(producer).blocks(consumer);
	unsplit_kernel <<< linear_config.x, linear_config.y, 0, consumer >>>
	               (elements, real_transposed, img_transposed, planar_transposed);
	cudaGetLastError() && assertcu;
	completion(consumer).blocks(producer);
	cuDoubleComplex one{1., 0.}, zero{0., 0.};
	cublasSetStream(handle, consumer) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, chainlen, shard_copies, &one, planar_transposed, shard_copies, &zero,
	            0, chainlen, planar, chainlen) && assertcublas;
	return completion(consumer);
}
