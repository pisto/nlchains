#pragma once

#include <list>
#include <exception>
#include <stdexcept>
#include "configuration.hpp"
#include "mpi_environment.hpp"
#include "utilities_cuda.cuh"

/*
 * Main loop control: coordinate between the MPI processes to quit at the same time step, throttle enqueuing of kernels,
 * check termination conditions.
 * This struct can be converted to uint64_t to get the current time step.
 */

struct loop_control {

	loop_control(uint64_t timebase): t(timebase) {}

	operator uint64_t() const { return t; }
	uint64_t operator*() const { return t; }
	uint64_t operator+=(uint64_t steps) { return t += steps; }

	bool break_now() const {
		if (quit_requested && !synched) synch_max();
		return gconf.steps == t && (synched || gconf.steps);
	}

protected:
	uint64_t t;
	mutable bool synched = false;
	//the max step reduction is performed independently from all other calculations
	const boost::mpi::communicator mpi_global_alt{ mpi_global, boost::mpi::comm_duplicate };
	void synch_max() const {
		if (synched) throw std::logic_error("max time step has already been synchronized");
		boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
		synched = true;
	}
};

struct loop_control_gpu : loop_control {

	/*
	 * Use this when recording exceptions from other threads. Will be rethrown when calling break_now().
	 */
	std::exception_ptr callback_err;

	loop_control_gpu(uint64_t timebase, cudaStream_t throttle_stream, uint32_t throttle_period = 2) : loop_control(
			timebase), throttle_stream(throttle_stream), throttle_period(throttle_period) {}

	uint64_t operator+=(uint64_t steps) {
		if (throttle_period) {
			completions.emplace_back(throttle_stream);
			while (completions.size() >= throttle_period) {
				completions.front().wait();
				completions.pop_front();
			}
		}
		return loop_control::operator+=(steps);
	}

	bool break_now() {
		if (callback_err) std::rethrow_exception(callback_err);
		return loop_control::break_now();
	}

private:
	std::list<completion> completions;
	const cudaStream_t throttle_stream;
	const uint32_t throttle_period;
};
