#include <fstream>
#include <stdexcept>
#include <iostream>
#include "../common/utilities.hpp"
#include "../common/configuration.hpp"
#include "../common/mpi_environment.hpp"
#include "dDNKG.hpp"

using namespace std;

namespace dDNKG {

	void eigensystem(const arma::vec &mp2, arma::mat &eigenvectors, arma::vec &omegas) {
		if (!mpi_global_coord) {
			arma::mat interaction = diagmat(mp2);
			interaction.diag(1).fill(-1);
			interaction.diag(-1).fill(-1);
			interaction(0, gconf.chain_length - 1) = interaction(gconf.chain_length - 1, 0) = -1;
			if (!arma::eig_sym(omegas, eigenvectors, interaction))
				throw runtime_error("Cannot calculate eigensystem!");
			omegas = sqrt(omegas);

			auto vecsize = gconf.chain_length * sizeof(double), matsize = size_t(gconf.chain_length) * gconf.chain_length * sizeof(double);
			ofstream dump_eigensystem(gconf.dump_prefix + "-omegas");
			dump_eigensystem.exceptions(ios::failbit | ios::badbit | ios::eofbit);
			dump_eigensystem.write((char *) omegas.memptr(), vecsize);
			dump_eigensystem.close();
			dump_eigensystem.open(gconf.dump_prefix + "-eigenvectors");
			dump_eigensystem.write((char *) eigenvectors.memptr(), matsize);
		} else {
			eigenvectors = arma::mat(gconf.chain_length, gconf.chain_length);
			omegas = arma::vec(gconf.chain_length);
		}
		boost::mpi::broadcast(mpi_global, eigenvectors.memptr(), eigenvectors.n_elem, 0);
		boost::mpi::broadcast(mpi_global, omegas.memptr(), omegas.n_elem, 0);
	}

}
