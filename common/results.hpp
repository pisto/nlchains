#pragma once

#include <string>
#include <fstream>
#include <array>
#include <vector>
#include "mpi_environment.hpp"
#include "configuration.hpp"

/*
 * A helper struct to collect and write results (linear energies, entropy).
 */

struct results {
	const bool a0is0;   //first mode has 0 frequency
	std::vector<double> linenergies;    //linear energies reduction

	results(bool a0is0);

	/*
	 * Reduce the linear energies for each processes, and return the WT and information entropies.
	 * norm_factor can be used to normalize the linear energies if the calcualtion on the GPU
	 * returns values with a costant multiplicative factor. It should be the same on all MPI processes.
	 */
	std::array<double, 2> entropies(const double *shard_linenergies, double norm_factor = 1.);

	/*
	 * Write the entropies to the dump file. All processes can call it but only the process 0 will actually write it.
	 */
	void write_entropy(uint64_t t, const std::array<double, 2> &entropies) const;

	/*
	 * Write the linearg energies to the dump file. All processes can call it but only the process 0 will actually write it.
	 */
	void write_linenergies(uint64_t t) const;

	/*
	 * Write the full ensemble state shard. This is an MPI collective call.
	 */
	void write_shard(uint64_t t, const double2 *shard) const;

	/*
	 * Check the entropy agains the user limit, and set quit_requested if necessary. This is an MPI collective call.
	 */
	void check_entropy(const std::array<double, 2> &entropies) const;

private:
	mutable std::ofstream entropydump;
	const std::string linenergies_template;
	std::vector<uint16_t> entropy_modes_indices;
	std::vector<double> entropy_modes;
	const boost::mpi::communicator mpi_global_results{ mpi_global, boost::mpi::comm_duplicate };
};
