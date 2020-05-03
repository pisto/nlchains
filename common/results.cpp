#include <cmath>
#include <fstream>
#include "utilities.hpp"
#include "results.hpp"

using namespace std;

results::results(bool a0is0) try : a0is0(a0is0), linenergies(gconf.chain_length),
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
} catch (const ios_base::failure &e) {
	throw ios_base::failure("cannot open entropy file ("s + e.what() + ")", e.code());
}

const results &results::write_entropy(uint64_t t) const {
	if (mpi_global_coord) return *this;
	double entropyinfo[]{t * gconf.dt, WTentropy, INFentropy};
	entropydump.write((char *) &entropyinfo, sizeof(entropyinfo)).flush();
	if (gconf.verbose)
		collect_ostream(cout) << "Entropy step " << t << ": " << entropyinfo[1] << '/' << entropyinfo[2] << endl;
	return *this;
}

const results &results::write_linenergies(uint64_t t) const try {
	if (mpi_global_coord) return *this;
	ofstream linenergies_dump(linenergies_template + to_string(t));
	linenergies_dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
	linenergies_dump.write((char *) linenergies.data(), gconf.sizeof_linenergies);
	return *this;
} catch (const ios_base::failure &e) {
	throw ios_base::failure("cannot open linenergies file ("s + e.what() + ")", e.code());
}

const results &results::write_shard(uint64_t t) const {
	auto dumpname = gconf.dump_prefix + "-";
	dumpname += to_string(t);
	if (mpi_global_size == 1) try {
		//avoid requesting a file lock
		ofstream dump(dumpname);
		dump.exceptions(ios::eofbit | ios::failbit | ios::badbit);
		dump.write((char *) gres.shard_host, gconf.sizeof_shard);
		return *this;
	} catch (const ios_base::failure &e) {
		throw ios_base::failure("cannot open dump file ("s + e.what() + ")", e.code());
	}
	MPI_File dump_mpif;
	MPI_File_open(mpi_global_results, dumpname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &dump_mpif) &&
	assertmpi;
	destructor([&] { MPI_File_close(&dump_mpif); });
	MPI_File_set_size(dump_mpif, gconf.sizeof_shard * mpi_global_size) && assertmpi;
	MPI_File_write_ordered(dump_mpif, gres.shard_host, gconf.shard_elements * 2, MPI_DOUBLE, MPI_STATUS_IGNORE) && assertmpi;
	return *this;
}

const results &results::check_entropy() const {
	if (!isfinite(WTentropy) || !isfinite(INFentropy)) {
		//this happens when we have blow ups or some energies extremely close to zero
		quit_requested = true;
		return *this;
	}
	if (gconf.entropy_limit_type == configuration::NONE) return *this;
	auto checked_entropy = gconf.entropy_limit_type == configuration::WT ? WTentropy : INFentropy;
	if (checked_entropy < gconf.entropy_limit) quit_requested = true;
	return *this;
}
