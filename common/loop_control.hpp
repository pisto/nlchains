#pragma once

#include <list>
#include <exception>
#include <stdexcept>
#include "mpi_environment.hpp"
#include "configuration.hpp"
#include "utilities_cuda.cuh"

/*
 * Main loop control: coordinate between the MPI processes to quit at the same time step, throttle enqueuing of kernels,
 * check termination conditions.
 * This struct can be converted to uint64_t to get the current time step.
 */

struct loop_control {

	/*
	 * Use this when recording exceptions from other threads. Will be rethrown when calling break_now().
	 */
	std::exception_ptr callback_err;

	loop_control(cudaStream_t throttle_stream, uint32_t throttle_period = 2) : throttle_stream(throttle_stream),
	                                                                           throttle_period(throttle_period) {}

	operator uint64_t() const { return t; }

	uint64_t operator*() const { return t; }

	uint64_t operator+=(uint64_t steps);

	bool break_now();

	~loop_control() {
		//make sure all MPI calls are matched
		if (!synched) synch_max();
	}

private:
	uint64_t t = gconf.time_offset;
	std::list<completion> completions;
	const cudaStream_t throttle_stream;
	const uint32_t throttle_period;
	bool synched = false;
	//the max step reduction is performend independently from allo ther calculations
	const boost::mpi::communicator mpi_global_alt{ mpi_global, boost::mpi::comm_duplicate };

	void synch_max();

};
