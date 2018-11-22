#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*
 * Global configuration and resources.
 */

extern struct configuration {

	bool verbose;

	uint16_t chain_length, shard_copies;
	uint32_t copies_total, shard_elements;
	uint32_t steps_grouping, steps_grouping_dump;
	double dt;

	std::string dump_prefix;
	uint64_t steps = 0, timebase = 0;
	double entropy_limit;
	enum { NONE = 0, INFORMATION, WT } entropy_limit_type = NONE;

	size_t linenergy_size, shard_size;
	std::vector<uint16_t> entropy_modes_indices;

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

	std::string initial_filename, entropymask_filename;
	struct help_quit {
		const boost::program_options::options_description options;
	};
};

#include "utilities_cuda.cuh"

extern struct resources {
	cudalist<double2> shard;
	cudalist<double> linenergies;
	cudalist<double2, true> shard_host;
	cudalist<double, true> linenergies_host;
} gres;
