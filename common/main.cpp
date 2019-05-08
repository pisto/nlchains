#include <csignal>
#include <iostream>
#include <fstream>
#include <boost/core/demangle.hpp>
#include <boost/program_options.hpp>
#include "mpi_environment.hpp"
#include "utilities.hpp"

using namespace std;
using namespace boost::mpi;

volatile sig_atomic_t quit_requested = false;

configuration gconf;
resources gres;

const environment mpienv(threading::multiple, false);
const communicator mpi_global;
const int mpi_global_coord = mpi_global.rank(), mpi_node_coord = [] {
	MPI_Comm node;
	/*
	 * Try to detect in a portable way the group of processes that run on the same node,
	 * and as such need to coordinate for the cuda device indexes.
	 */
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node) && assertmpi;
	return communicator(node, comm_take_ownership).rank();
}(), mpi_global_size = mpi_global.size();

const string process_ident(
		"MPI rank/hostname/node_id: " + to_string(mpi_global_coord) + "/" + mpienv.processor_name() + "/" +
		to_string(mpi_node_coord));

map<string, function<int(int argc, char *argv[])>> &programs() {
	static map<string, function<int(int argc, char *argv[])>> progs;
	return progs;
};

/*
 * Get exceptions instead of asserts from boost.
 */
namespace boost {
	void assertion_failed(const char *expr, const char *function, const char *file, long line) {
		throw logic_error("Boost assert failed: "s + expr + ", at " + file + ":" + to_string(line) + " in " + function);
	}

	void assertion_failed_msg(const char *expr, const char *msg, const char *function, const char *file, long line) {
		throw logic_error(
				"Boost assert failed ("s + msg + "): " + "" + expr + ", at " + file + ":" + to_string(line) + " in " +
				function);
	}
}

parse_cmdline::parse_cmdline(const string &name) : options(name) {
	using namespace boost::program_options;
	options.add_options()
			("verbose,v", "print extra informations")
			("initial,i", value(&initial_filename)->required(), "initial state file")
			("chain_length,n", value(&gconf.chain_length)->required(), "length of the chain")
			("copies,c", value(&gconf.copies_total)->required(), "number of realizations of the chain in the ensemble")
			("dt", value(&gconf.dt)->required(), "time step value")
			("kernel_batching,b", value(&gconf.kernel_batching)->required(), "number of steps per kernel invocation")
			("steps,s", value(&gconf.steps)->default_value(0), "number of steps in total (0 for infinite)")
			("time_offset,o", value(&gconf.time_offset)->default_value(0), "time offset in dump files")
			("entropy,e", value(&gconf.entropy_limit), "entropy limit")
			("WTlimit", "limit WT entropy instead of information entropy")
			("entropymask", value(&entropymask_filename), "mask of linenergies to include in entropy calculation")
			("prefix,p", value(&gconf.dump_prefix)->required(), "prefix for dump files")
			("dump_interval", value(&gconf.dump_interval),
			 "number of steps between full state dumps (defaults to same value as --batch, does not affect time granularity of entropy)");
}

void parse_cmdline::run(int argc, char *argv[]) try {
	if (argc == 1) throw help_quit{options};
	try {
		store(boost::program_options::parse_command_line(argc, argv, options), vm);
		notify(vm);
	} catch (const boost::program_options::error &e) {
		throw invalid_argument(e.what());
	}

	//check user arguments
	gconf.verbose = vm.count("verbose");
	if (gconf.copies_total % mpi_global_size)
		throw invalid_argument("copies must be a multiple of the number of devices");
	gconf.shard_copies = gconf.copies_total / mpi_global_size;
	if (gconf.chain_length < 2 || !gconf.shard_copies || gconf.dt <= 0 || !gconf.kernel_batching)
		throw invalid_argument("--chain_length must be >= 2, --copies, --dt and --kernel_batching must be positive numbers");
	if (vm.count("entropy"))
		gconf.entropy_limit_type = vm.count("WTlimit") ? configuration::WT : configuration::INFORMATION;
	if (!vm.count("dump_interval")) gconf.dump_interval = gconf.kernel_batching;
	else if (!gconf.dump_interval) throw invalid_argument("--dump_interval must be > 0");
	if (gconf.dump_prefix.empty())
		throw invalid_argument("--prefix must not be an empty string");
	if (gconf.dump_interval % gconf.kernel_batching)
		throw invalid_argument("--dump_interval must be a multiple of --kernel_batching");
	if (gconf.steps % gconf.kernel_batching)
		throw invalid_argument("--steps must be a multiple of --kernel_batching");
	if (gconf.time_offset && gconf.steps && gconf.time_offset >= gconf.steps)
		throw invalid_argument("--time_offset must be less than --steps");

	gconf.sizeof_linenergies = sizeof(double) * gconf.chain_length;
	if (uint64_t(gconf.chain_length) * gconf.shard_copies >= 0x40000000ULL)
		throw invalid_argument("(--chain_length * --copies) / mpi_processes must be <= 2^30");
	gconf.shard_elements = uint32_t(gconf.chain_length) * gconf.shard_copies;
	gconf.sizeof_shard = sizeof(double2) * gconf.shard_elements;
	gres.shard_buffer.resize(gconf.shard_elements);
	gres.linenergies_buffer.resize(gconf.chain_length);
	gres.shard_host = gres.shard_buffer.data();
	gres.linenergies_host = gres.linenergies_buffer.data();
	//XXX catching ios_base::failure does not work with gcc 5/6, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66145
	//working around it would be difficult and ugly, let's just hope software stacks move on.
	try {
		ifstream initial_state(initial_filename);
		initial_state.exceptions(ios::failbit | ios::badbit | ios::eofbit);
		initial_state.seekg(gconf.sizeof_shard * mpi_global_coord).read((char *) gres.shard_host, gconf.sizeof_shard);
	} catch (const ios_base::failure &e) {
		throw ios_base::failure("could not read initial state ("s + e.what() + ")", e.code());
	}
	if (!entropymask_filename.empty())
		try {
			ifstream entropymask_file(entropymask_filename);
			entropymask_file.exceptions(ios::failbit | ios::badbit | ios::eofbit);
			loopi(gconf.chain_length)
				if (entropymask_file.get()) gconf.entropy_modes_indices.push_back(i);
			if (gconf.entropy_modes_indices.size() < 2) throw invalid_argument("Too few modes for entropy calculation");
		} catch (const ios_base::failure &e) {
			throw ios_base::failure("could not read entropy mask file ("s + e.what() + ")", e.code());
		}

} catch (const invalid_argument &e) {
	ostringstream fmtmsg;
	fmtmsg << e.what() << endl << options;
	throw invalid_argument(fmtmsg.str());
}


static int print_fatal_error(const string &msg) {
	collect_ostream(cerr) << process_ident << ": " << msg << endl;
	return 1;
}

int main(int argc, char **argv) try {
	//whether catching these signal works is MPI-implementation dependent
	for (int s: {SIGINT, SIGTERM}) signal(s, [](int) { quit_requested = true; });

	if (!programs().size()) throw logic_error("nlchains compiled without any subprogram!");
	string program_names;
	for (auto kv: programs()) {
		if (!program_names.empty()) program_names += ", ";
		program_names += kv.first;
	}
	if (argc == 1) {
		if (!mpi_global_coord)
			cout << "Please specify as first argument the subprogram to run: " << program_names << endl;
		return 0;
	}
	auto program = programs().find(argv[1]);
	if (program == programs().end()) throw invalid_argument("please specific a valid program among " + program_names);

	return program->second(argc - 1, argv + 1);

} catch (const parse_cmdline::help_quit &e) {
	if (!mpi_global_coord) cout << e.options;
	return 0;
} catch (const ios_base::failure &e) {
	return print_fatal_error("I/O error, "s + e.what() + " (" + e.code().message() + ")");
} catch (const system_error &e) {
	return print_fatal_error("system error, "s + e.code().message());
} catch (const invalid_argument &e) {
	if (mpi_global_coord) return 1;
	return print_fatal_error("invalid argument, "s + e.what());
} catch (const std::exception &e) {
	return print_fatal_error("exception "s + boost::core::demangled_name(typeid(e)) + ", " + e.what());
} catch (...) {
	return print_fatal_error("unspecified fatal exception");
}
