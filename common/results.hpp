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
	double WTentropy, INFentropy;

	results(bool a0is0);

	/*
	 * Reduce the linear energies for each processes, and store the result in linenergies.
	 * norm_factor can be used to normalize the linear energies if the calculation on the GPU
	 * returns values with a costant multiplicative factor. It should be the same on all MPI processes.
	 */
	results &calc_linenergies(double norm_factor = 1.);

	/*
	 * Calculate the WT and information entropies.
	 */
	results &calc_entropies();

	/*
	 * Write the entropies to the dump file. All processes can call it but only the process 0 will actually write it.
	 */
	const results &write_entropy(uint64_t t) const;

	/*
	 * Write the linearg energies to the dump file. All processes can call it but only the process 0 will actually write it.
	 */
	const results &write_linenergies(uint64_t t) const;

	/*
	 * Write the full ensemble state shard. This is an MPI collective call.
	 */
	const results &write_shard(uint64_t t) const;

	/*
	 * Check the entropy agains the user limit, and set quit_requested if necessary. This is an MPI collective call.
	 */
	const results &check_entropy() const;

private:
	mutable std::ofstream entropydump;
	const std::string linenergies_template;
	std::vector<uint16_t> entropy_modes_indices;
	std::vector<double> entropy_modes;
	const boost::mpi::communicator mpi_global_results{ mpi_global, boost::mpi::comm_duplicate };
};
