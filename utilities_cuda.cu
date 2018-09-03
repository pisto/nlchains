#include "utilities_cuda.cuh"
#include "configuration.hpp"

int cudaOccupancyMaxPotentialBlockSize_void(void* kern, int maxblock){
	int blocks, threads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kern, 0, maxblock) && assertcu;
	return threads;
}

__global__ void split_kernel(size_t elements, const double2* planar, double* real, double* img){
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= elements) return;
	auto pair = planar[idx];
	real[idx] = pair.x, img[idx] = pair.y;
}

__global__ void unsplit_kernel(size_t elements, const double* real, const double* img, double2* planar){
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= elements) return;
	planar[idx] = { real[idx], img[idx] };
}

plane2split::plane2split(uint16_t chainlen, uint16_t shard_copies):
		chainlen(chainlen), shard_copies(shard_copies), elements(size_t(shard_copies) * chainlen),
		real_transposed(elements * 2), img_transposed((double*)real_transposed + elements), planar_transposed(elements) {
	cublasCreate(&handle) && assertcublas;
}

completion plane2split::split(const double2* planar, cudaStream_t stream_read, cudaStream_t stream_write){
	cuDoubleComplex one{ 1., 0. }, zero{ 0., 0. };
	cublasSetStream(handle, stream_read) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, shard_copies, chainlen, &one, planar, chainlen, &zero, 0, shard_copies, planar_transposed, shard_copies) && assertcublas;
	completion(stream_read).blocks(stream_write);
	static auto kinfo = make_kernel_info(split_kernel);
	auto linear_config = kinfo.linear_configuration(elements, gconf.verbose);
	split_kernel<<<linear_config.x, linear_config.y, 0, stream_write>>>
		(elements, planar_transposed, real_transposed, img_transposed);
	cudaGetLastError() && assertcu;
	return completion(stream_write);
}

completion plane2split::plane(double2* planar, cudaStream_t stream_read, cudaStream_t stream_write) const {
	static auto kinfo = make_kernel_info(unsplit_kernel);
	auto linear_config = kinfo.linear_configuration(elements, gconf.verbose);
	unsplit_kernel<<<linear_config.x, linear_config.y, 0, stream_read>>>
		(elements, real_transposed, img_transposed, planar_transposed);
	cudaGetLastError() && assertcu;
	completion(stream_read).blocks(stream_write);
	cuDoubleComplex one{ 1., 0. }, zero{ 0., 0. };
	cublasSetStream(handle, stream_write) && assertcublas;
	cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, chainlen, shard_copies, &one, planar_transposed, shard_copies, &zero, 0, chainlen, planar, chainlen) && assertcublas;
	return completion(stream_write);
}
