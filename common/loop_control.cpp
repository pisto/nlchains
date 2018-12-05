#include "loop_control.hpp"

using namespace std;

uint64_t loop_control::operator+=(uint64_t steps) {
	t += steps;
	if (throttle_period) {
		completions.emplace_back(throttle_stream);
		while (completions.size() >= throttle_period) {
			completions.front().wait();
			completions.pop_front();
		}
	}
	return t;
}

bool loop_control::break_now() {
	if (callback_err) rethrow_exception(callback_err);
	if (quit_requested && !synched) synch_max();
	return gconf.steps && gconf.steps == t;
}

void loop_control::synch_max() {
	if (synched) throw std::logic_error("max time step has already been synchronized");
	boost::mpi::all_reduce(mpi_global_alt, t, gconf.steps, boost::mpi::maximum<uint64_t>());
	synched = true;
}
