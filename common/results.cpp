#include <cmath>
#include <fstream>
#include "utilities.hpp"
#include "results.hpp"

using namespace std;

results::results(bool a0is0) : a0is0(a0is0), linenergies(gconf.chain_length),
                               linenergies_template(gconf.dump_prefix + "-linenergies-"),
                               entropy_modes_indices(gconf.entropy_modes_indices) {
	if (a0is0 && !entropy_modes_indices.empty() && entropy_modes_indices[0] == 0) {
		entropy_modes_indices.erase(entropy_modes_indices.begin());
		if (entropy_modes_indices.size() < 2) throw invalid_argument("Too few modes for entropy calculation");
	}
	entropy_modes.resize(entropy_modes_indices.size());
	if (mpi_global_coord) return;
	entropydump = ofstream(gconf.dump_prefix + "-entropy");
	entropydump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
}

void results::write_entropy(uint64_t t, const array<double, 2> &entropies) const {
	if (mpi_global_coord) return;
	double entropyinfo[]{t * gconf.dt, entropies[0], entropies[1]};
	entropydump.write((char *) &entropyinfo, sizeof(entropyinfo)).flush();
	if (gconf.verbose)
		collect_ostream(cout) << "Entropy step " << t << ": " << entropyinfo[1] << '/' << entropyinfo[2] << endl;
}

void results::write_linenergies(uint64_t t) const {
	if (mpi_global_coord) return;
	ofstream linenergies_dump(linenergies_template + to_string(t));
	linenergies_dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
	linenergies_dump.write((char *) linenergies.data(), gconf.sizeof_linenergies);
}

void results::write_shard(uint64_t t, const double2 *shard) const {
	auto dumpname = gconf.dump_prefix + "-";
	dumpname += to_string(t);
	if (mpi_global_size == 1) {
		//avoid requesting a file lock
		ofstream dump(dumpname);
		dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
		dump.write((char *) shard, gconf.sizeof_shard);
		return;
	}
	MPI_File dump_mpif;
	MPI_File_open(mpi_global_results, dumpname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &dump_mpif) &&
	assertmpi;
	destructor([&] { MPI_File_close(&dump_mpif); });
	MPI_File_set_size(dump_mpif, gconf.sizeof_shard * mpi_global_size) && assertmpi;
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
