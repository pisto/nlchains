#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <map>

/*
 * Associate subprograms to --program argument.
 */

std::map<std::string, std::function<int(int argc, char *argv[])>> &programs();

/*
 * Global configuration and resources.
 * A shard is the portion of ensemble chain replications that the GPU owns.
 */

extern struct configuration {

	bool verbose;

	uint16_t chain_length, shard_copies;
	uint32_t copies_total, shard_elements;
	uint32_t kernel_batching, dump_interval;
	double dt;

	std::string dump_prefix;
	uint64_t steps = 0, time_offset = 0;
	double entropy_limit;
	enum {
		NONE = 0, INFORMATION, WT
	} entropy_limit_type = NONE;

	size_t sizeof_linenergies, sizeof_shard;
	std::vector<uint16_t> entropy_modes_indices;

} gconf;

/*
 * Command line parsing.
 */

#include <boost/program_options.hpp>

struct parse_cmdline {
	boost::program_options::options_description options;
	boost::program_options::variables_map vm;

	parse_cmdline(const std::string &name);

	void operator()(int argc, char *argv[]);

	std::string initial_filename, entropymask_filename;
	/*
	 * Throw this to terminate gracefully and print the boost options.
	 */
	struct help_quit {
		const boost::program_options::options_description options;
	};
};


#include "utilities.hpp"
//for double2
#include <vector_types.h>

extern struct resources {
	double2 *shard_host = 0, *shard_gpu = 0;
	double *linenergies_host = 0, *linenergies_gpu = 0;

private:
	std::vector<double2> shard_buffer;
	std::vector<double, simd_allocator<double>> linenergies_buffer;
	friend struct parse_cmdline;
} gres;
