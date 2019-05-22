#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include "configuration.hpp"
#include "mpi_environment.hpp"
#include "utilities.hpp"
#include "utilities_cuda.cuh"

using namespace std;

cuda_ctx_t cuda_ctx;

cuda_ctx_t::cuda_ctx_raii cuda_ctx_t::activate(int id) {
	if (this->id >= 0) throw logic_error("CUDA context is already active");
	cudaSetDevice(id) && assertcu;
	cuda_ctx_t::cuda_ctx_raii raii;
	this->id = id;
	cudaHostRegister(gres.linenergies_host, gconf.sizeof_linenergies, cudaHostRegisterDefault) && assertcu;
	cudaHostGetDevicePointer(&gres.linenergies_gpu, gres.linenergies_host, 0) && assertcu;
	cudaHostRegister(gres.shard_host, gconf.sizeof_shard, cudaHostRegisterDefault) && assertcu;
	gres.shard_gpu = raii.shard_buffer_gpu = gconf.shard_elements;
	cudaMemcpy(gres.shard_gpu, gres.shard_host, gconf.sizeof_shard, cudaMemcpyHostToDevice) && assertcu;

	cudaGetDeviceProperties(&dev_props, id) && assertcu;
	collect_ostream(cout) << process_ident << ": GPU id " << id << ", clocks MHz " << dev_props.clockRate / 1000 << '/'
	                      << dev_props.memoryClockRate / 1000 << endl;
	return raii;
}

cuda_ctx_t::cuda_ctx_raii::~cuda_ctx_raii() {
	if (!shard_buffer_gpu) return;
	cudaDeviceSynchronize() && assertcu;
	cudaHostUnregister(gres.linenergies_host);
	gres.linenergies_gpu = 0;
	cudaHostUnregister(gres.shard_host);
	{ auto move_out = move(shard_buffer_gpu); }
	gres.shard_gpu = 0;
	cuda_ctx = cuda_ctx_t();
	cudaDeviceReset();
}

//cudaOccupancyMaxPotentialBlockSize is not defined out of nvcc, no idea why
cudaError cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, const void *func, size_t dynamicSMemSize,
                                             int blockSizeLimit);

kernel_info_base::kernel_config kernel_info_base::linear_configuration(size_t elements) const {
	/*
	 * Try to find the largest block size so that all SMs are busy, assuming that each thread works on an element.
	 * The effective max block size is the maximum number of resident threads per multiprocessor
	 * (cudaDeviceProp::maxThreadsPerMultiProcessor) divided by the maximum number of resident blocks per multiprocessor.
	 * It is 16 for CC <= 3.7 and CC == 7.5, else it is 32. Then, use cudaOccupancyMaxPotentialBlockSize() to get the
	 * CUDA runtime to take in account everything else, such as shared memory size and registry count limits.
	 */
	int cc = cuda_ctx.dev_props.major * 10 + cuda_ctx.dev_props.minor;
	int max_resident_blocks = cc <= 37 || cc == 75 ? 16 : 32;       //XXX why is this not exposed in cudaDeviceProp??
	int max_block_size = cuda_ctx.dev_props.maxThreadsPerMultiProcessor / max_resident_blocks;
	kernel_config config;
	for (; max_block_size; max_block_size -= 32) {
		int blocks;
		cudaOccupancyMaxPotentialBlockSize(&blocks, &config.threads, k_type_erased, 0, max_block_size) && assertcu;
		config.blocks = elements / config.threads;
		if (config.blocks >= cuda_ctx.dev_props.multiProcessorCount) break;
	}
	config.blocks += !!(elements % config.threads);
	if (gconf.verbose && linear_configuration_printed != elements) {
		collect_ostream(cerr) << process_ident << ": kernel \"" << kname << "\":" << endl << '\t'
		                      << kernel_attrs.numRegs << " regs" << endl << '\t' << kernel_attrs.localSizeBytes
		                      << " lmem" << endl << '\t' << config.threads << " linear block size" << endl;
		linear_configuration_printed = elements;
	}
	return config;
}
