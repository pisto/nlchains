#pragma once

#include <stdexcept>
#include <string>
#include <map>
#include <utility>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>

/*
 * Cuda errors to exceptions translation, same as assertmpi in utilities.h.
 * Use as
 *      cuda*(...) && assertcu;
 *      cufft*(...) && assertcufft;
 *      cublas*(...) && assertcublas;
 */

#define assertcu assertcu_helper{ __FILE__ ":" + std::to_string(__LINE__) }
#define assertcufft assertcufft_helper{ __FILE__ ":" + std::to_string(__LINE__) }
#define assertcublas assertcublas_helper{ __FILE__ ":" + std::to_string(__LINE__) }

struct cuda_error : std::runtime_error {
	cuda_error(const std::string &place, cudaError err) : std::runtime_error(
			"@ " + place + ": " + cudaGetErrorString(err)) {}

protected:
	cuda_error(std::string &&err) : std::runtime_error(std::move(err)) {}
};

struct assertcu_helper {
	std::string place;
};

inline int operator&&(cudaError ret, assertcu_helper &&p) {
	return ret == cudaSuccess ? cudaSuccess : throw cuda_error(p.place, ret);
}

#define CODE_STRING(x) { x, #x }

struct assertcufft_helper {
	std::string place;

	static const std::map<cufftResult, std::string> &errors() {
		static const std::map<cufftResult, std::string> errors = {
				CODE_STRING(CUFFT_INVALID_PLAN),
				CODE_STRING(CUFFT_ALLOC_FAILED),
				CODE_STRING(CUFFT_INVALID_TYPE),
				CODE_STRING(CUFFT_INVALID_VALUE),
				CODE_STRING(CUFFT_INTERNAL_ERROR),
				CODE_STRING(CUFFT_EXEC_FAILED),
				CODE_STRING(CUFFT_SETUP_FAILED),
				CODE_STRING(CUFFT_INVALID_SIZE),
				CODE_STRING(CUFFT_UNALIGNED_DATA),
				CODE_STRING(CUFFT_INCOMPLETE_PARAMETER_LIST),
				CODE_STRING(CUFFT_INVALID_DEVICE),
				CODE_STRING(CUFFT_PARSE_ERROR),
				CODE_STRING(CUFFT_NO_WORKSPACE),
				CODE_STRING(CUFFT_NOT_IMPLEMENTED),
				CODE_STRING(CUFFT_LICENSE_ERROR),
				CODE_STRING(CUFFT_NOT_SUPPORTED),
		};
		return errors;
	}
};

struct cufft_error : cuda_error {
	cufft_error(const std::string &place, cufftResult err) :
			cuda_error("cuFFT @ " + place + ": " + assertcufft_helper::errors().at(err) + ", cuda last error: " +
			           cudaGetErrorString(cudaGetLastError())) {}
};

inline int operator&&(cufftResult ret, assertcufft_helper &&p) {
	return ret == CUFFT_SUCCESS ? CUFFT_SUCCESS : throw cufft_error(p.place, ret);
}


struct assertcublas_helper {
	std::string place;

	static const std::map<cublasStatus_t, std::string> &errors() {
		static const std::map<cublasStatus_t, std::string> errors = {
				CODE_STRING(CUBLAS_STATUS_NOT_INITIALIZED),
				CODE_STRING(CUBLAS_STATUS_ALLOC_FAILED),
				CODE_STRING(CUBLAS_STATUS_INVALID_VALUE),
				CODE_STRING(CUBLAS_STATUS_ARCH_MISMATCH),
				CODE_STRING(CUBLAS_STATUS_MAPPING_ERROR),
				CODE_STRING(CUBLAS_STATUS_EXECUTION_FAILED),
				CODE_STRING(CUBLAS_STATUS_INTERNAL_ERROR),
				CODE_STRING(CUBLAS_STATUS_NOT_SUPPORTED),
				CODE_STRING(CUBLAS_STATUS_LICENSE_ERROR),
		};
		return errors;
	}
};

#undef CODE_STRING

struct cublas_error : cuda_error {
	cublas_error(const std::string &place, cublasStatus_t err) :
			cuda_error("cuBLAS @ " + place + ": " + assertcublas_helper::errors().at(err) + ", cuda last error: " +
			           cudaGetErrorString(cudaGetLastError())) {}
};

inline int operator&&(cublasStatus_t ret, assertcublas_helper &&p) {
	return ret == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_SUCCESS : throw cublas_error(p.place, ret);
}

/*
 * Routines to get/set global scope variables in device memory
 */

#include <array>

template<typename T>
T *get_device_address(T &on_device) {
	T *address;
	cudaGetSymbolAddress((void **) &address, (const void *) &on_device) && assertcu;
	return address;
}

template<typename T>
T get_device_object(const T &on_device, cudaStream_t stream = 0) {
	T on_host;
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T), 0, cudaMemcpyDeviceToHost,
	                          stream) && assertcu;
	cudaStreamSynchronize(stream) && assertcu;
	return on_host;
}

template<typename T>
void get_device_object(const T &on_device, T &on_host, cudaStream_t stream = 0) {
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T), 0, cudaMemcpyDeviceToHost,
	                          stream) && assertcu;
}

template<typename T, size_t len>
std::array<T, len> get_device_object(const T (&on_device)[len], cudaStream_t stream = 0) {
	std::array<T, len> on_host;
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T[len]), 0, cudaMemcpyDeviceToHost,
	                          stream) && assertcu;
	cudaStreamSynchronize(stream) && assertcu;
	return on_host;
}

template<typename T>
void set_device_object(const T &on_host, T &on_device, cudaStream_t stream = 0) {
	cudaMemcpyToSymbolAsync((const void *) &on_device, (const void *) &on_host, sizeof(T), 0, cudaMemcpyHostToDevice,
	                        stream) && assertcu;
}

template<typename T>
void memset_device_object(T &on_device, int value, cudaStream_t stream = 0) {
	void *addr;
	cudaGetSymbolAddress(&addr, (const void *) &on_device) && assertcu;
	cudaMemsetAsync(addr, value, sizeof(T), stream) && assertcu;
}

/*
 * Device or host pinned memory wrapper with RAII.
 */

template<typename T = void, bool host = false>
struct cudalist {
	cudalist() = default;

	cudalist(size_t len) {
		static_assert(!host, "wrong constructor");
		cudaMalloc(&mem, len * sizeof(T)) && assertcu;
	}

	cudalist(size_t len, bool wc) {
		static_assert(host, "wrong constructor");
		cudaMallocHost(&mem, len * sizeof(T), cudaHostAllocMapped | (wc ? cudaHostAllocWriteCombined : 0));
	}

	cudalist(cudalist &&o) {
		mem = o.mem;
		o.mem = 0;
	}

	cudalist &operator=(cudalist &&o) {
		this->~cudalist();
		mem = o.mem;
		o.mem = 0;
		return *this;
	}

	operator T *() const { return mem; }

	operator void *() const { return mem; }

	T &operator[](size_t i) { return mem[i]; }

	T *operator*() { return mem; }

	const T &operator[](size_t i) const { return mem[i]; }

	operator bool() const { return mem; }

	T *devptr() const {
		static_assert(host, "cannot call devptr() on a device buffer");
		T *ret;
		cudaHostGetDevicePointer((void **) &ret, (void *) mem, 0) && assertcu;
		return ret;
	}

	~cudalist() {
		if (!mem) return;
		host ? cudaFreeHost((void *) mem) : cudaFree((void *) mem);
		mem = 0;
	}

private:
	T *mem = 0;
};

template<bool host>
struct cudalist<void, host> {
	cudalist() = default;

	cudalist(size_t len) {
		static_assert(!host, "wrong constructor");
		cudaMalloc(&mem, len) && assertcu;
	}

	cudalist(size_t len, bool wc) {
		static_assert(host, "wrong constructor");
		cudaMallocHost(&mem, len, cudaHostAllocMapped | (wc ? cudaHostAllocWriteCombined : 0));
	}

	cudalist(cudalist &&o) {
		mem = o.mem;
		o.mem = 0;
	}

	cudalist &operator=(cudalist &&o) {
		this->~cudalist();
		mem = o.mem;
		o.mem = 0;
		return *this;
	}

	operator void *() const { return mem; }

	void *operator*() { return mem; }

	operator bool() const { return mem; }

	void *devptr() const {
		static_assert(host, "cannot call devptr() on a device buffer");
		void *ret;
		cudaHostGetDevicePointer((void **) &ret, (void *) mem, 0) && assertcu;
		return ret;
	}

	~cudalist() {
		if (!mem) return;
		host ? cudaFreeHost((void *) mem) : cudaFree((void *) mem);
		mem = 0;
	}

private:
	void *mem = 0;
};

/*
 * completion represents a time point in a stream. It is used to easily synchronize between streams,
 * and to make the host wait on stream completions. This is essentially a wrapper struct around cudaEvent_t.
 */

struct completion {

	completion() = default;

	explicit completion(cudaStream_t stream) {
		record(stream);
	}

	completion(completion &&o) {
		*this = std::move(o);
	}

	completion &operator=(completion &&o) {
		delevent();
		std::swap(o.event, event);
		return *this;
	}

	~completion() {
		delevent();
	}

	completion &record(cudaStream_t stream) {
		newevent();
		cudaEventRecord(event, stream) && assertcu;
		return *this;
	}

	void blocks(cudaStream_t stream) const {
		if (!event) return;
		cudaStreamWaitEvent(stream, event, 0) && assertcu;
	}

	void wait() const {
		if (!event) return;
		cudaEventSynchronize(event) && assertcu;
	}

private:
	void delevent() {
		if (!event) return;
		cudaEventDestroy(event);
		event = 0;
	}

	void newevent() {
		delevent();
		cudaEventCreateWithFlags(&event, cudaEventDisableTiming) && assertcu;
	}

	cudaEvent_t event = 0;
};

/*
 * Lambda -> cudaCallback transform. All exceptions are caught and set on the provided
 * exception_ptr reference, to be read by your main thread.
 */

#include <iostream>
#include <exception>
#include <memory>

template<typename L>
struct cuda_callback_data {
	std::exception_ptr &callback_err;
	L lambda;
};

template<typename L>
void add_cuda_callback(cudaStream_t stream, std::exception_ptr &callback_err, L &lambda) {
	cudaStreamAddCallback(stream, +[](cudaStream_t stream, cudaError_t status, void *userData) {
		std::unique_ptr<cuda_callback_data<L &>> data(reinterpret_cast<cuda_callback_data<L &> *>(userData));
		try { data.lambda(status); }
		catch (...) { data.callback_err = std::current_exception(); }
	}, new cuda_callback_data<L &>{callback_err, lambda}, 0) && assertcu;
}

template<typename L>
void add_cuda_callback(cudaStream_t stream, std::exception_ptr &callback_err, L &&lambda) {
	cudaStreamAddCallback(stream, +[](cudaStream_t stream, cudaError_t status, void *userData) {
		std::unique_ptr<cuda_callback_data<L>> data(reinterpret_cast<cuda_callback_data<L> *>(userData));
		try { data->lambda(status); }
		catch (...) { data->callback_err = std::current_exception(); }
	}, new cuda_callback_data<L>{callback_err, std::move(lambda)}, 0) && assertcu;
}

/*
 * CUDA resource RAII. Notes that it also modifies global gres to use GPU buffers.
 */

extern struct cuda_ctx_t {
	cudaDeviceProp dev_props;

	struct cuda_ctx_raii {
		cuda_ctx_raii(cuda_ctx_raii &&o) { shard_buffer_gpu = std::move(o.shard_buffer_gpu); }
		~cuda_ctx_raii();

	private:
		friend struct cuda_ctx_t;
		cudalist<double2> shard_buffer_gpu;

		cuda_ctx_raii() = default;
	};

	cuda_ctx_raii activate(int id);

private:
	int id = -1;
} cuda_ctx;

/*
 * Helper class to print runtime optimization hints about kernels.
 * Use as
 *      static auto your_kernel_info = make_kernel_info(your_kernel);
 * If your kernel has a 1D structure, you can call linear_configuration(elements) to get the optimal
 * block/grid size.
 */

struct kernel_info_base {
	std::string kname;
	cudaFuncAttributes kernel_attrs;

	struct kernel_config {
		int blocks, threads;
	};

	kernel_config linear_configuration(size_t elements) const;

protected:
	mutable bool printed_info;
	const void *k_type_erased;

	kernel_info_base(const void *k, std::string kname) : k_type_erased(k), kname(std::move(kname)) {
		cudaFuncGetAttributes(&kernel_attrs, k) && assertcu;
	}

};

template<typename Kernel>
struct kernel_info : kernel_info_base {
	Kernel k;
	static_assert(sizeof(k) == sizeof(void *), "CUDA kernels are not pointers!");

	kernel_info(Kernel k, const std::string &kname) : kernel_info_base((const void *)k, kname), k(k) {}
};

template<typename Kernel>
auto make_kernel_info(Kernel k, const std::string &kname_preprocessor, const std::string &kname = std::string()) {
	return kernel_info<Kernel>(k, kname.empty() ? kname_preprocessor : kname);
}

#define make_kernel_info(k, ...) make_kernel_info(k, #k, ##__VA_ARGS__)

/*
 * Move from planar (copy, chain index, real/img indexes) to split (real/img, chain, copy) representations of the chains.
 */

struct plane2split {
	plane2split();

	~plane2split() { cublasDestroy(handle); }

	/*
	 * Input resources can be made available to other operations as soon as possible if you
	 * provide two streams in the split/plane calls: the first is released as soon as the input
	 * can be modified, the second does the rest of the work, and the returned completion object
	 * synchronizes to the second.
	 */

	completion split(cudaStream_t producer, cudaStream_t consumer);

	completion split(cudaStream_t stream) { return split(stream, stream); }

	completion plane(cudaStream_t producer, cudaStream_t consumer) const;

	completion plane(cudaStream_t stream) const { return plane(stream, stream); }

	const cudalist<double> coords_transposed;
	double *const momenta_transposed;
private:
	cudalist<double2> planar_transposed;
	cublasHandle_t handle = 0;
};

/*
 * Complex math.
 */

__device__ inline cufftDoubleComplex operator*(const cufftDoubleComplex &a, const cufftDoubleComplex &b) {
	return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline cufftDoubleComplex &operator*=(cufftDoubleComplex &a, const cufftDoubleComplex &b) {
	return (a = a * b);
}

//e^(i x), i = imaginary unit
__device__ inline cufftDoubleComplex e_pow_I(double x) {
	double sin, cos;
	sincos(x, &sin, &cos);
	return {cos, sin};
}
