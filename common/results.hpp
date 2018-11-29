#pragma once

#include <string>
#include <fstream>
#include <array>
#include <vector>
#include "mpi_environment.hpp"
#include "configuration.hpp"

struct results {
	const bool a0is0;   //first mode has 0 frequency
	std::vector<double> linenergies;

	results(bool a0is0);

	std::array<double, 2> entropies(const double *shard_linenergies, double norm_factor = 1.);

	void write_entropy(uint64_t t, const std::array<double, 2> &entropies) const;

	void write_linenergies(uint64_t t) const;

	void write_shard(uint64_t t, const double2 *shard) const;

	void check_entropy(const std::array<double, 2> &entropies) const;

private:
	mutable std::ofstream entropydump;
	const std::string linenergies_template;
	std::vector<uint16_t> entropy_modes_indices;
	std::vector<double> entropy_modes;
};
