#include <sstream>
#include <iostream>
#include <iterator>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>
#include "../utilities_cuda.cuh"
#include "../configuration.hpp"
#include "kg_disorder.hpp"

namespace kg_disorder {

	__constant__ double dt_c[8], dt_d[8], mp2[2048], beta;

	__device__ double rhs_KG(double left, double center, double right, double mp2){
		return -(center * (mp2 + beta * center * center) - left - right);
	}

	/*
	 * Move entire chain in a warp. Compare with kg_fpu_toda.cu . Only difference is that the mp2 vector is held
	 * in shared memory in order to have a comparable register usage to the non-disorder version.
	 */
	__global__ void move_chain_in_warp(double2* planar, const double* mp2_gmem, uint32_t steps_grouping, uint16_t copies){
		//compile-time
		constexpr int elements_in_thread = optimized_chain_length / 32 + !!(optimized_chain_length % 32),
				full_lanes = optimized_chain_length % 32 ?: 32;

		int idx = blockIdx.x * blockDim.x + threadIdx.x,
				my_copy = idx / 32, lane = idx % 32, lane_left = (lane + 31) % 32, lane_right = (lane + 1) % 32;
		bool full_lane = lane < full_lanes;
		if(my_copy >= copies) return;
		//offset copy
		planar += my_copy * size_t(optimized_chain_length);
		double phi[elements_in_thread + 2], pi[elements_in_thread];
		__shared__ double mp2_shmem[elements_in_thread][32];
		phi[elements_in_thread] = 0, pi[elements_in_thread - 1] = 0;
		if(threadIdx.x < 32) mp2_shmem[elements_in_thread - 1][lane] = 0;
		//offset previous threads
		size_t thread_offset = full_lane ? lane * elements_in_thread : full_lanes * elements_in_thread + (lane - full_lanes) * (elements_in_thread - 1);
		planar += thread_offset;
		mp2_gmem += thread_offset;
		#pragma unroll
		for(int i_0 = 0, i = 1; i_0 < elements_in_thread - 1; i_0++, i++) {
			auto pair = planar[i_0];
			phi[i] = pair.x, pi[i_0] = pair.y;
			if(threadIdx.x < 32) mp2_shmem[i_0][lane] = mp2_gmem[i_0];
		}
		if(full_lane) {
			auto pair = planar[elements_in_thread - 1];
			phi[elements_in_thread] = pair.x, pi[elements_in_thread - 1] = pair.y;
			if(threadIdx.x < 32) mp2_shmem[elements_in_thread - 1][lane] = mp2_gmem[elements_in_thread - 1];
		}
		__syncthreads();


		for(uint32_t i = 0; i < steps_grouping; i++){
			//XXX this unroll may need to be tweaked
			#pragma unroll (elements_in_thread > 4 ? 1 : 7)
			for(int k = 0; k < 7; k++){
				double dt_c_k = dt_c[k];
				if(i && !k) dt_c_k *= 2;    //merge last and first evolution of phi variable, since pi is not update in 8th steps of 6th order Yoshida
				#pragma unroll
				for(int i_0 = 0, i = 1; i_0 < elements_in_thread; i_0++, i++)
					phi[i] += dt_c_k * pi[i_0];
				//communicate nearest neighbours to adjacent threads
				double communicate = full_lane ? phi[elements_in_thread] : phi[elements_in_thread - 1];
				phi[0] = cub::ShuffleIndex<32>(communicate, lane_left, 0xFFFFFFFF);
				phi[elements_in_thread + 1] = cub::ShuffleIndex<32>(phi[1], lane_right, 0xFFFFFFFF);
				if(!full_lane) phi[elements_in_thread] = phi[elements_in_thread + 1];
				#pragma unroll
				for(int i_0 = 0, i = 1; i_0 < elements_in_thread; i_0++, i++)
					pi[i_0] += dt_d[k] * rhs_KG(phi[i - 1], phi[i], phi[i + 1], mp2_shmem[i_0][lane]);
			}
		}
		#pragma unroll
		for(int i_0 = 0, i = 1; i_0 < elements_in_thread - 1; i_0++, i++) {
			phi[i] += dt_c[7] * pi[i_0];
			planar[i_0] = double2{ phi[i], pi[i_0] };
		}
		if(full_lane) {
			phi[elements_in_thread] += dt_c[7] * pi[elements_in_thread - 1];
			planar[elements_in_thread - 1] = double2{ phi[elements_in_thread], pi[elements_in_thread - 1] };
		}
	}

	/*
	 * Move entire chain in a thread. Compare with kg_fpu_toda.cu .
	 */
	template<uint16_t chain_length>
	__global__ void move_chain_in_thread(double2* planar, uint32_t steps_grouping, uint16_t copies){
		uint16_t my_copy = blockIdx.x * blockDim.x + threadIdx.x;
		if(my_copy >= copies) return;
		//offset copy
		planar += my_copy * size_t(chain_length);
		double2 pairs[chain_length];
		#pragma unroll
		for(uint16_t i = 0; i < chain_length; i++) pairs[i] = planar[i];

		for(uint32_t i = 0; i < steps_grouping; i++){
			//XXX this unroll may need to be tweaked
			#pragma unroll (chain_length > 4 ? 1 : chain_length)
			for(int k = 0; k < 7; k++){
				double dt_c_k = dt_c[k];
				if(i && !k) dt_c_k *= 2;    //merge last and first evolution of phi variable, since pi is not update in 8th steps of 6th order Yoshida
				#pragma unroll
				for(uint16_t i = 0; i < chain_length; i++)
					pairs[i].x += dt_c_k * pairs[i].y;
				pairs[0].y += dt_d[k] * rhs_KG(pairs[chain_length - 1].x, pairs[0].x, pairs[1].x, mp2[0]);
				#pragma unroll
				for(uint16_t i = 1; i < chain_length - 1; i++)
					pairs[i].y += dt_d[k] * rhs_KG(pairs[i - 1].x, pairs[i].x, pairs[i + 1].x, mp2[i]);
				pairs[chain_length - 1].y += dt_d[k] * rhs_KG(pairs[chain_length - 2].x, pairs[chain_length - 1].x, pairs[0].x, mp2[chain_length - 1]);
			}
		}
		#pragma unroll
		for(uint16_t i = 0; i < chain_length; i++){
			pairs[i].x += dt_c[7] * pairs[i].y;
			planar[i] = pairs[i];
		}
	}
	//some machinery to get the compiler to create all the versions of move_planar, and to get the right one at runtime
	namespace {
		using thread_kernel_info = kernel_info<decltype(&move_chain_in_thread<0>)>;
		template<int chain_length = 2>
		struct thread_kernel_resolver {
			static const thread_kernel_info& get(int chain_length_required){
				if(chain_length != chain_length_required)
					return thread_kernel_resolver<chain_length + 1>::get(chain_length_required);
				static thread_kernel_info kinfo(move_chain_in_thread<chain_length>, "move_chain_in_thread<" + std::to_string(chain_length) + ">");
				return kinfo;
			}
		};
		template<> struct thread_kernel_resolver<32> {
			static const thread_kernel_info& get(int){ throw std::logic_error("Shouldn't be here"); }
		};
	}

	/*
	 * Move in split format. Compare with kg_fpu_toda.cu . The mp2 argument may alias the constant
	 * memory array if the constant buffer is large enough to hold all the linear parameter values,
	 * otherwise it resides in global memory.
	 */
	__global__ void move_split(double* phi, double* pi, const double* mp2, uint16_t chainlen, uint16_t shard_copies, uint32_t steps_grouping){
		auto chain_idx_0 = blockIdx.x * blockDim.x + threadIdx.x;
		if(chain_idx_0 >= shard_copies) return;
		size_t chain_idx_last = chain_idx_0 + (chainlen - 1) * size_t(shard_copies);
		//these values are handled outside of the inner loop, can be saved in registers to avoid memory access
		double phi0 = phi[chain_idx_0], phi1 = phi[chain_idx_0 + size_t(shard_copies)], pi0 = pi[chain_idx_0];
		for(uint32_t i = 0; i < steps_grouping; i++){
			for(int k = 0; k < 7; k++){
				double dt_c_k = dt_c[k];
				if(i && !k) dt_c_k *= 2;    //merge last and first evolution of phi variable, since pi is not update in 8th steps of 6th order Yoshida
				double previous_pi = pi[chain_idx_0 + size_t(shard_copies)];    //pi[1]
				phi0 += dt_c_k * pi0;
				phi1 += dt_c_k * previous_pi;
				double last_updated_phi[]{ phi0, phi1 };
				//this appears to be already unrolled by the compiler
				auto mp2_i = mp2 + 1;
				for(size_t i = chain_idx_0 + 2 * size_t(shard_copies); i <= chain_idx_last; i += shard_copies, mp2_i++) {
					double current_pi = pi[i];
					double current_updated_phi = (phi[i] += dt_c_k * current_pi);
					pi[i - shard_copies] = previous_pi + dt_d[k] * rhs_KG(last_updated_phi[0],
					                                                       last_updated_phi[1],
					                                                       current_updated_phi,
					                                                       *mp2_i);
					previous_pi = current_pi;
					last_updated_phi[0] = last_updated_phi[1];
					last_updated_phi[1] = current_updated_phi;
				}
				pi[chain_idx_last] = previous_pi + dt_d[k] * rhs_KG(last_updated_phi[0], last_updated_phi[1], phi0, *mp2_i);
				pi0 += dt_d[k] * rhs_KG(last_updated_phi[1], phi0, phi1, *mp2);
			}
		}
		phi0 += dt_c[7] * pi0;
		phi1 += dt_c[7] * pi[chain_idx_0 + size_t(shard_copies)];
		for(size_t i = chain_idx_0 + 2 * size_t(shard_copies); i <= chain_idx_last; i += shard_copies) phi[i] += dt_c[7] * pi[i];

		phi[chain_idx_0] = phi0;
		phi[chain_idx_0 + size_t(shard_copies)] = phi1;
		pi[chain_idx_0] = pi0;
	}


	completion move(plane2split& splitter, bool& use_split_kernel, const double* mp2_gmem, cudaStream_t stream){
		if (use_split_kernel) {
			static auto kinfo = make_kernel_info(move_split);
			auto linear_config = kinfo.linear_configuration(gconf.shard_copies, gconf.verbose);
			static auto mp2_const_ptr = (const double*)get_device_address(mp2);
			kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
				(splitter.real_transposed, splitter.img_transposed,
				gconf.chain_length <= sizeof(mp2) / sizeof(double) ? mp2_const_ptr : mp2_gmem,
				gconf.chain_length, gconf.shard_copies, gconf.steps_grouping);
		} else {
			if (gconf.chain_length < 32) {
				auto& kinfo = thread_kernel_resolver<>::get(gconf.chain_length);
				auto linear_config = kinfo.linear_configuration(gconf.shard_copies, gconf.verbose);
				kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
					(gres.shard, gconf.steps_grouping, gconf.shard_copies);
			} else if (gconf.chain_length == optimized_chain_length) {
				static auto kinfo = make_kernel_info(move_chain_in_warp);
				auto linear_config = kinfo.linear_configuration(size_t(gconf.shard_copies) * 32, gconf.verbose);
				kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
					(gres.shard, mp2_gmem, gconf.steps_grouping, gconf.shard_copies);
			} else {
				static bool warned = false;
				if(!warned && gconf.chain_length < 2048){
					std::ostringstream msg("Could not find optimized version for chain_length ", std::ios::app);
					msg<<gconf.chain_length<<", try to reconfigure with -Doptimized_chain_length="<<gconf.chain_length
					   <<" and recompile."<<std::endl;
					std::cerr<<msg.str();
					warned = true;
				}
				use_split_kernel = true;
				return move(splitter, use_split_kernel, mp2_gmem, stream);
			}
		}
		cudaGetLastError() && assertcu;
		return completion(stream);
	}


	__global__ void make_linenergies_kernel(const double* projection_phi, const double* projection_pi, uint16_t chain_length, uint16_t shard_copies, double* linenergies){
		size_t idx = (size_t(blockDim.x * blockIdx.x) + threadIdx.x) / 32;
		if(idx >= chain_length) return;
		projection_phi += idx * shard_copies;
		projection_pi += idx * shard_copies;
		double sum = 0;
		for(size_t c = threadIdx.x % 32; c < shard_copies; c += 32){
			auto p_phi = projection_phi[c], p_pi = projection_pi[c];
			sum += p_phi * p_phi + p_pi * p_pi;
		}
		cub::WarpReduce<double>::TempStorage dummy;
		#ifdef __CUDA_ARCH__
		static_assert(sizeof(dummy) <= 1, "make_linenergies_kernel assumes you are on CC >= 3.5 to use warp shuffles");
		#endif
		sum = cub::WarpReduce<double>(dummy).Sum(sum);
		if(threadIdx.x % 32) return;
		linenergies[idx] = sum;
	}

	completion make_linenergies(const double* projection_phi, const double* projection_pi, cudaStream_t stream){
		static auto kinfo = make_kernel_info(make_linenergies_kernel);
		auto linear_config = kinfo.linear_configuration(gconf.chain_length * 32, gconf.verbose);
		kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
			(projection_phi, projection_pi, gconf.chain_length, gconf.shard_copies, gres.linenergies_host);
		cudaGetLastError() && assertcu;
		return completion(stream);
	}

}
