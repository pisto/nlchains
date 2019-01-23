#include <cub/iterator/cache_modified_input_iterator.cuh>
#include "../common/utilities_cuda.cuh"
#include "../common/configuration.hpp"
#include "DNLS.hpp"

namespace DNLS {

	template<typename T>
	using load_ldg = cub::CacheModifiedInputIterator<cub::LOAD_LDG, T>;

	__global__ void
	evolve_nonlinear_kernel(uint32_t shard_elements, cufftDoubleComplex *psis, double beta_dt_symplectic) {
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= shard_elements) return;
		auto &psi = psis[idx];
		psi *= e_pow_I(-beta_dt_symplectic * (psi.x * psi.x + psi.y * psi.y));
	}

	__global__ void evolve_linear_kernel(uint32_t shard_elements, uint16_t chainlen, cufftDoubleComplex *psis_k,
	                                     load_ldg<cufftDoubleComplex> evolve_linear_table) {
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= shard_elements) return;
		uint16_t chainlen_mask = chainlen - 1;
		psis_k[idx] *= evolve_linear_table[chainlen & chainlen_mask ? idx % chainlen : idx & chainlen_mask];
	}

	completion evolve_nonlinear(double beta_dt_symplectic, cudaStream_t stream) {
		static auto kinfo = make_kernel_info(evolve_nonlinear_kernel);
		auto linear_config = kinfo.linear_configuration(gconf.shard_elements, gconf.verbose);
		kinfo.k <<< linear_config.x, linear_config.y, 0, stream >>>
		                                                 (gconf.shard_elements, gres.shard, beta_dt_symplectic);
		cudaGetLastError() && assertcu;
		return completion(stream);
	}

	completion evolve_linear(const cufftDoubleComplex *evolve_linear_table, cudaStream_t stream) {
		static auto kinfo = make_kernel_info(evolve_linear_kernel);
		auto linear_config = kinfo.linear_configuration(gconf.shard_elements, gconf.verbose);
		kinfo.k <<< linear_config.x, linear_config.y, 0, stream >>>
		                                                 (gconf.shard_elements, gconf.chain_length, gres.shard, evolve_linear_table);
		cudaGetLastError() && assertcu;
		return completion(stream);
	}

	namespace callback {
		__constant__ uint16_t chainlen;
		__constant__ uint8_t evolve_linear_table_idx;

		__device__ cufftDoubleComplex evolve_linear(void *in, size_t offset, void *evolve_linear_tables_all, void *) {
			auto psis_k = static_cast<cufftDoubleComplex *>(in);
			load_ldg<cufftDoubleComplex> evolve_linear_table(
					static_cast<cufftDoubleComplex *>(evolve_linear_tables_all));
			evolve_linear_table += uint32_t(chainlen) * evolve_linear_table_idx;
			uint32_t idx = offset;
			uint16_t chainlen_mask = chainlen - 1;
			return psis_k[idx] * evolve_linear_table[chainlen & chainlen_mask ? idx % chainlen : idx & chainlen_mask];
		}

		__constant__ double beta_dt_symplectic;

		__device__ cufftDoubleComplex evolve_nonlinear(void *in, size_t offset, void *, void *) {
			auto psi = static_cast<cufftDoubleComplex *>(in)[offset];
			return psi * e_pow_I(-beta_dt_symplectic * (psi.x * psi.x + psi.y * psi.y));
		}

		__device__ const cufftCallbackLoadZ evolve_linear_ptr = evolve_linear, evolve_nonlinear_ptr = evolve_nonlinear;
	}


	__global__ void make_linenergies_kernel(uint16_t copies_shard, uint16_t chainlen, const cufftDoubleComplex *psis_k,
	                                        const double *omega, double *linenergies_host) {
		uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
		if (k >= chainlen) return;
		double sum = 0;
		psis_k += k;
		for (uint16_t c = 0; c < copies_shard; c++, psis_k += chainlen) {
			auto psi_k = *psis_k;
			sum += psi_k.x * psi_k.x + psi_k.y * psi_k.y;
		}
		linenergies_host[k] = sum * omega[k];
	}

	completion make_linenergies(const cufftDoubleComplex *psis_k, const double *omega, cudaStream_t stream) {
		static auto kinfo = make_kernel_info(make_linenergies_kernel);
		auto linear_config = kinfo.linear_configuration(gconf.chain_length, gconf.verbose);
		kinfo.k <<< linear_config.x, linear_config.y, 0, stream >>>
		                                                 (gconf.shard_copies, gconf.chain_length, psis_k, omega, gres.linenergies_host);
		cudaGetLastError() && assertcu;
		return completion(stream);
	}

}
