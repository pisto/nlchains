#include <sstream>
#include <iostream>
#include <cub/util_ptx.cuh>
#include "../utilities_cuda.cuh"
#include "../configuration.hpp"
#include "kg_fpu_toda.hpp"

namespace kg_fpu_toda {

	__constant__ double dt_c[8], dt_d[8], m, alpha, beta;

	/*
	 * Right-hand side of the equation of motion. Since the interaction is nearest-neighbor, in most cases
	 * it is possible to cache some computations at one chain element for the element to the right.
	 */

	template<Model model> struct RHS;
	template<> struct RHS<KG>{
		__device__ RHS(double, double){}
		__device__ double operator()(double left, double center, double right){
			return -(center * (m + 2 + beta * center * center) - left - right);
		}
	};
	template<> struct RHS<FPU>{
		double cached_dleft3_beta;
		__device__ RHS(double left, double center){
			auto dleft = center - left;
			cached_dleft3_beta = beta * dleft * dleft * dleft;
		}
		__device__ double operator()(double left, double center, double right){
			auto ret = (left + right - 2 * center);
			ret *= (1 + alpha * (right - left));
			auto dright = right - center;
			double dright3_beta = beta * dright * dright * dright;
			ret += dright3_beta - cached_dleft3_beta;
			cached_dleft3_beta = dright3_beta;
			return ret;
		}
	};
	__constant__ double alpha2, alpha2_inv;
	template<> struct RHS<Toda>{
		double cached_exp_dleft_over_alpha2;
		__device__ RHS(double left, double center):
				cached_exp_dleft_over_alpha2(alpha2_inv * exp(alpha2 * (center - left))) {}
		__device__ double operator()(double left, double center, double right){
			double exp_dright = alpha2_inv * exp(alpha2 * (right - center));
			double ret = exp_dright - cached_exp_dleft_over_alpha2;
			cached_exp_dleft_over_alpha2 = exp_dright;
			return ret;
		}
	};

	/*
	 * Optimized version of time march, no memory transfers as all the chain state is loaded in registers and memory is
	 * read/written once. Each warp owns a copy, each thread contains ceil(chain_length / 32) elements of the chain.
	 * This kernel must be recompiled for each target chain_length, because that is to be a compile time constant,
	 * because phi[] and pi[] must be indexed statically, otherwise accesses to local memory are generated.
	 */
	template<Model model>
	__global__ void move_chain_in_warp(double2 *planar, uint32_t steps_grouping, uint16_t copies){
		//compile-time
		constexpr int elements_in_thread = optimized_chain_length / 32 + !!(optimized_chain_length % 32),
				full_lanes = optimized_chain_length % 32 ?: 32;

		int idx = blockIdx.x * blockDim.x + threadIdx.x,
				my_copy = idx / 32, lane = idx % 32, lane_left = (lane + 31) % 32, lane_right = (lane + 1) % 32;
		bool full_lane = lane < full_lanes;
		if(my_copy >= copies) return;
		//offset copy
		planar += my_copy * size_t(optimized_chain_length);
		/*
		 * Arrangement is (e.g. for optimized_chain_length = 35)
		 * lane idx            |0     |1     |2     |3     |4     .... |31
		 * chain element idx   |0  1| |2  3| |4  5| |6  /| |7  /| .... |34 /|
		 *
		 * phi[] is 2 elements wider than chain_length because phi[0] and phi[chain_length + 1] contain
		 * the nearest neighbors after the warp shuffle.
		 */
		double phi[elements_in_thread + 2], pi[elements_in_thread];
		phi[elements_in_thread] = 0, pi[elements_in_thread - 1] = 0;
		//offset previous threads
		planar += full_lane ? lane * elements_in_thread : full_lanes * elements_in_thread + (lane - full_lanes) * (elements_in_thread - 1);
		#pragma unroll
		for(int i_0 = 0, i = 1; i_0 < elements_in_thread - 1; i_0++, i++) {
			auto pair = planar[i_0];
			phi[i] = pair.x, pi[i_0] = pair.y;
		}
		if(full_lane) {
			auto pair = planar[elements_in_thread - 1];
			phi[elements_in_thread] = pair.x, pi[elements_in_thread - 1] = pair.y;
		}

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
				RHS<model> rhs(phi[0], phi[1]);
				#pragma unroll
				for(int i_0 = 0, i = 1; i_0 < elements_in_thread; i_0++, i++)
					pi[i_0] += dt_d[k] * rhs(phi[i - 1], phi[i], phi[i + 1]);
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
	 * Version for chain_lenght < 32: each thread owns an entire copy of the chain. Like move_chain_in_warp,
	 * this kernel must be generated multiple times with a compile time constant chain_length, in order
	 * to avoid accesses to local memory when indexing arrays. All the version with 1 < chain_length < 32
	 * are generated through template metaprogramming, so no need to recompile. The downside is longer
	 * compilation times.
	 */
	template<Model model, uint16_t chain_length>
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
				RHS<model> rhs(pairs[chain_length - 1].x, pairs[0].x);
				pairs[0].y += dt_d[k] * rhs(pairs[chain_length - 1].x, pairs[0].x, pairs[1].x);
				#pragma unroll
				for(uint16_t i = 1; i < chain_length - 1; i++)
					pairs[i].y += dt_d[k] * rhs(pairs[i - 1].x, pairs[i].x, pairs[i + 1].x);
				pairs[chain_length - 1].y += dt_d[k] * rhs(pairs[chain_length - 2].x, pairs[chain_length - 1].x, pairs[0].x);
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
		using thread_kernel_info = kernel_info<decltype(&move_chain_in_thread<FPU, 0>)>;
		template<Model model, int chain_length = 2>
		struct thread_kernel_resolver {
			static const thread_kernel_info& get(int chain_length_required){
				if(chain_length != chain_length_required)
					return thread_kernel_resolver<model, chain_length + 1>::get(chain_length_required);
				static thread_kernel_info kinfo(move_chain_in_thread<model, chain_length>, "move_chain_in_thread<" + std::to_string(int(model)) + "," + std::to_string(chain_length) + ">");
				return kinfo;
			}
		};
		template<Model model> struct thread_kernel_resolver<model, 32> {
			static const thread_kernel_info& get(int){ throw std::logic_error("Shouldn't be here"); }
		};
	}

	/*
	 * Generic version of move with split format. Each thread owns a copy, and loops through the
	 * chain elements to update them. In order to avoid more than one read/write memory accesses for each chain element,
	 * the loop iteration consists in updating a phi[i] value and the pi[i - 1] value using the saved phi[i - 2],
	 * phi[i - 1] and phi[i] values. This comes at the cost of more complicated bookkeeping of previous phi/pi values,
	 * and code that comes right before and after the loop.
	 * Note that since copies are distributed among threads, using a number of copies multiple of 32 is advised. The
	 * block size is actually determined and printed once at runtime, and setting the number of copies
	 */
	template<Model model>
	__global__ void move_split(double* phi, double* pi, uint16_t chainlen, uint16_t shard_copies, uint32_t steps_grouping){
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
				RHS<model> rhs(phi0, phi1);
				double last_updated_phi[]{ phi0, phi1 };
				//this appears to be already unrolled by the compiler
				for(size_t i = chain_idx_0 + 2 * size_t(shard_copies); i <= chain_idx_last; i += shard_copies) {
					double current_pi = pi[i];
					double current_updated_phi = (phi[i] += dt_c_k * current_pi);
					pi[i - shard_copies] = previous_pi + dt_d[k] * rhs(last_updated_phi[0],
					                                                       last_updated_phi[1],
					                                                       current_updated_phi);
					previous_pi = current_pi;
					last_updated_phi[0] = last_updated_phi[1];
					last_updated_phi[1] = current_updated_phi;
				}
				pi[chain_idx_last] = previous_pi + dt_d[k] * rhs(last_updated_phi[0], last_updated_phi[1], phi0);
				pi0 += dt_d[k] * rhs(last_updated_phi[1], phi0, phi1);
			}
		}
		phi0 += dt_c[7] * pi0;
		phi1 += dt_c[7] * pi[chain_idx_0 + size_t(shard_copies)];
		for(size_t i = chain_idx_0 + 2 * size_t(shard_copies); i <= chain_idx_last; i += shard_copies) phi[i] += dt_c[7] * pi[i];

		phi[chain_idx_0] = phi0;
		phi[chain_idx_0 + size_t(shard_copies)] = phi1;
		pi[chain_idx_0] = pi0;
	}


	template<Model model>
	completion move(plane2split*& splitter, cudaStream_t stream){
		if (splitter) {
			static auto kinfo = make_kernel_info_name(move_split<model>, std::string("move_split<") + std::to_string(int(model)) + ">");
			auto linear_config = kinfo.linear_configuration(gconf.shard_copies, gconf.verbose);
			kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
				(splitter->real_transposed, splitter->img_transposed, gconf.chain_length, gconf.shard_copies, gconf.steps_grouping);
		} else {
			if (gconf.chain_length < 32) {
				auto& kinfo = thread_kernel_resolver<model>::get(gconf.chain_length);
				auto linear_config = kinfo.linear_configuration(gconf.shard_copies, gconf.verbose);
				kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
					(gres.shard, gconf.steps_grouping, gconf.shard_copies);
			} else if (gconf.chain_length == optimized_chain_length) {
				static auto kinfo = make_kernel_info_name(move_chain_in_warp<model>, std::string("move_chain_in_warp<") + std::to_string(int(model)) + ">");
				auto linear_config = kinfo.linear_configuration(size_t(gconf.shard_copies) * 32, gconf.verbose);
				kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
					(gres.shard, gconf.steps_grouping, gconf.shard_copies);
			} else {
				static bool warned = false;
				if(!warned && gconf.chain_length < 2048){
					std::ostringstream msg("Could not find optimized version for chain_length ", std::ios::app);
					msg<<gconf.chain_length<<", try to reconfigure with -Doptimized_chain_length="<<gconf.chain_length
					   <<" and recompile."<<std::endl;
					std::cerr<<msg.str();
					warned = true;
				}
				splitter = new plane2split(gconf.chain_length, gconf.shard_copies);
				splitter->split(gres.shard, stream);
				return move<model>(splitter, stream);
			}
		}
		cudaGetLastError() && assertcu;
		return completion(stream);
	}
	template completion move<KG>(plane2split*& splitter, cudaStream_t stream);
	template completion move<FPU>(plane2split*& splitter, cudaStream_t stream);
	template completion move<Toda>(plane2split*& splitter, cudaStream_t stream);


	__global__ void
	make_linenergies_kernel(uint16_t chainlen, uint16_t copies, double* linenergies, const double2* fft_phis,
			const double2* fft_pis, const double* omegas){
		uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= chainlen) return;
		fft_phis += idx, fft_pis += idx;
		double sum = 0, omega = omegas[idx];
		auto square = [](double x){ return x * x; };
		for(uint16_t c = 0; c < copies; c++, fft_phis += chainlen, fft_pis += chainlen){
			auto fft_phi = *fft_phis, fft_pi = *fft_pis;
			double energy = (square(fft_phi.x) + square(fft_phi.y)) * omega;
			energy += 2 * (fft_phi.y * fft_pi.x - fft_phi.x * fft_pi.y);
			energy *= omega;
			energy += square(fft_pi.x) + square(fft_pi.y);
			sum += energy;
		}
		linenergies[idx] = sum;
	}

	completion make_linenergies(const double2* fft_phis, const double2* fft_pis, const double* omegas, cudaStream_t stream){
		static auto kinfo = make_kernel_info(make_linenergies_kernel);
		auto linear_config = kinfo.linear_configuration(gconf.chain_length, gconf.verbose);
		kinfo.k<<<linear_config.x, linear_config.y, 0, stream>>>
			(gconf.chain_length, gconf.shard_copies, gres.linenergies_host, fft_phis, fft_pis, omegas);
		cudaGetLastError() && assertcu;
		return completion(stream);
	}

}
