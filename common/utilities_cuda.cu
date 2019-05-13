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

plane2split::plane2split() : coords_transposed(gconf.shard_elements * 2),
                             momenta_transposed((double *) coords_transposed + gconf.shard_elements),
                             planar_transposed(gconf.shard_elements) {
	cublasCreate(&handle) && assertcublas;
}

completion plane2split::split(cudaStream_t producer, cudaStream_t consumer) {
	cuDoubleComplex one{1., 0.}, zero{0., 0.};
	completion(producer).blocks(consumer);
	cublasSetStream(handle, consumer) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, gconf.shard_copies, gconf.chain_length, &one, gres.shard_gpu,
	            gconf.chain_length, &zero, 0, gconf.shard_copies, planar_transposed, gconf.shard_copies) &&
	assertcublas;
	completion(consumer).blocks(producer);
	static auto kinfo = make_kernel_info(split_kernel);
	auto launch = kinfo.linear_configuration(gconf.shard_elements);
	kinfo.k <<< launch.blocks, launch.threads, 0, consumer >>>
	        (gconf.shard_elements, planar_transposed, coords_transposed, momenta_transposed);
	cudaGetLastError() && assertcu;
	return completion(consumer);
}

completion plane2split::plane(cudaStream_t producer, cudaStream_t consumer) const {
	static auto kinfo = make_kernel_info(unsplit_kernel);
	auto launch = kinfo.linear_configuration(gconf.shard_elements);
	completion(producer).blocks(consumer);
	kinfo.k <<< launch.blocks, launch.threads, 0, consumer >>>
	        (gconf.shard_elements, coords_transposed, momenta_transposed, planar_transposed);
	cudaGetLastError() && assertcu;
	completion(consumer).blocks(producer);
	cuDoubleComplex one{1., 0.}, zero{0., 0.};
	cublasSetStream(handle, consumer) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, gconf.chain_length, gconf.shard_copies, &one, planar_transposed,
	            gconf.shard_copies, &zero, 0, gconf.chain_length, gres.shard_gpu, gconf.chain_length) && assertcublas;
	return completion(consumer);
}
