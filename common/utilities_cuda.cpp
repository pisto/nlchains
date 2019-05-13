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

kernel_info_base::kernel_config kernel_info_base::linear_configuration(size_t elements) const {
	int grid_size, block_size;
	//all device have max threads = 2048, CC > 3 can have 64 resident blocks instead of 32
	for (int max_block_size = cuda_ctx.dev_props.major > 3 ? 64 : 128; max_block_size; max_block_size -= 32) {
		int blocks;
		//cudaOccupancyMaxPotentialBlockSize is not exposed out of nvcc, no idea why
		cudaError
		cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, const void *func, size_t dynamicSMemSize,
		                                   int blockSizeLimit);
		cudaOccupancyMaxPotentialBlockSize(&blocks, &block_size, k_type_erased, 0, max_block_size) && assertcu;
		grid_size = elements / block_size;
		if (grid_size >= cuda_ctx.dev_props.multiProcessorCount) break;
	}
	grid_size += !!(elements % block_size);
	if (gconf.verbose && !printed_info) {
		collect_ostream(cerr) << process_ident << ": kernel \"" << kname << "\":" << endl << '\t'
		                      << kernel_attrs.numRegs << " regs" << endl << '\t' << kernel_attrs.localSizeBytes
		                      << " lmem" << endl << '\t' << block_size << " linear block size" << endl;
		printed_info = true;
	}
	return {grid_size, block_size};
}
