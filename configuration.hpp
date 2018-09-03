#pragma once

#include <cstdint>
#include <string>

/*
 * Global configuration and resources.
 */

extern struct configuration {

	bool verbose;

	uint16_t chain_length, shard_copies;
	uint32_t steps_grouping, steps_grouping_dump;
	double dt;

	std::string dump_prefix, initial_filename;
	uint64_t steps = 0, timebase = 0;
	uint16_t copies_total;
	double entropy_limit;
	enum { NONE = 0, INFORMATION, WT } entropy_limit_type = NONE;

	size_t linenergy_size, shard_elements, shard_size;

} gconf;

/*
 * Command line parsing.
 */

#include <boost/program_options.hpp>

struct parse_cmdline {
	boost::program_options::options_description options;
	boost::program_options::variables_map vm;

	parse_cmdline(const std::string& name);
	void operator()(int argc, char* argv[]);
};

#include "utilities_cuda.cuh"

extern struct resources {
	cudalist<double2> shard;
	cudalist<double> linenergies;
	cudalist<double2, true> shard_host;
	cudalist<double, true> linenergies_host;
} gres;
