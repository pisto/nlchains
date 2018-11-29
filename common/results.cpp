#include <cmath>
#include <fstream>
#include "utilities.hpp"
#include "results.hpp"

using namespace std;

results::results(bool a0is0) : a0is0(a0is0), linenergies(gconf.chain_length),
                               linenergies_template(
		                               gconf.dump_prefix.empty() ? "" : gconf.dump_prefix + "-linenergies-"),
                               entropy_modes_indices(gconf.entropy_modes_indices) {
	if (a0is0 && !entropy_modes_indices.empty() && entropy_modes_indices[0] == 0) {
		entropy_modes_indices.erase(entropy_modes_indices.begin());
		if (entropy_modes_indices.size()) throw invalid_argument("too few modes for the calculation of entropy");
	}
	if (mpi_global_coord) return;
	entropydump = ofstream((gconf.dump_prefix.empty() ? "" : gconf.dump_prefix + "-") + "entropy");
	entropydump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
}

void results::write_entropy(uint64_t t, const array<double, 2> &entropies) const {
	if (mpi_global_coord) return;
	double entropyinfo[]{t * gconf.dt, entropies[0], entropies[1]};
	entropydump.write((char *) &entropyinfo, sizeof(entropyinfo)).flush();
	if (!gconf.verbose) return;
	ostringstream info("Entropy step ", ios::app);
	info << t << ": " << entropyinfo[1] << '/' << entropyinfo[2] << endl;
	cerr << info.str();
}

void results::write_linenergies(uint64_t t) const {
	if (mpi_global_coord) return;
	ofstream linenergies_dump(linenergies_template + to_string(t));
	linenergies_dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
	linenergies_dump.write((char *) linenergies.data(), gconf.linenergy_size);
}

void results::write_shard(uint64_t t, const double2 *shard) const {
	auto dumpname = (gconf.dump_prefix.empty() ? "" : gconf.dump_prefix + "-");
	dumpname += to_string(t);
	if (mpi_global.size() == 1) {
		//avoid request a file lock
		ofstream dump(dumpname);
		dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
		dump.write((char *) shard, gconf.shard_size);
		return;
	}
	MPI_File dump_mpif;
	MPI_File_open(MPI_COMM_WORLD, dumpname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &dump_mpif) &&
	assertmpi;
	destructor([&] { MPI_File_close(&dump_mpif); });
	MPI_File_set_size(dump_mpif, gconf.shard_size * mpi_global.size()) && assertmpi;
	MPI_File_write_ordered(dump_mpif, shard, gconf.shard_elements * 2, MPI_DOUBLE, MPI_STATUS_IGNORE) && assertmpi;
}

void results::check_entropy(const array<double, 2> &entropies) const {
	if (!isfinite(entropies[0]) || !isfinite(entropies[1])) {
		//this happens when we have blow ups or some energies extremely close to zero
		quit_requested = true;
		return;
	}
	switch (gconf.entropy_limit_type) {
		case configuration::NONE:
			return;
		case configuration::WT:
			if (entropies[0] >= gconf.entropy_limit) return;
		case configuration::INFORMATION:
			if (entropies[1] >= gconf.entropy_limit) return;
	}
	quit_requested = true;
}
