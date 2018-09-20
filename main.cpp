#include <csignal>
#include <iostream>
#include <sstream>
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
const communicator mpi_global, mpi_global_alt(mpi_global, comm_duplicate);
const int mpi_global_coord = mpi_global.rank(), mpi_node_coord = []{
	MPI_Comm node;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node) && assertmpi;
	return communicator(node, comm_take_ownership).rank();
}();

map<std::string, function<int(int argc, char* argv[])>>& programs(){
	static map<string, function<int(int argc, char* argv[])>> progs;
	return progs;
};

ostringstream errmsg("MPI rank/host name/GPU id: "
                            + to_string(mpi_global_coord) + "/" + mpienv.processor_name() + "/" + to_string(mpi_node_coord)
                            + ": ", ios::ate);
namespace boost {
	void assertion_failed(const char* expr, const char* function, const char* file, long line) {
		throw logic_error(string("Boost assert failed: ") + expr + ", at " + file + ":" + to_string(line) + " in " + function);
	}
	void assertion_failed_msg(const char* expr, const char* msg, const char* function, const char* file, long line) {
		throw logic_error(string("Boost assert failed (") + msg + "): " + "" + expr + ", at " + file + ":" + to_string(line) + " in " + function);
	}
}

parse_cmdline::parse_cmdline(const string& name): options(name){
	using namespace boost::program_options;
	options.add_options()
			("verbose,v", "print extra informations")
			("prefix,p", value(&gconf.dump_prefix)->required(), "prefix for dump files")
			("initial,i", value(&initial_filename)->required(), "initial state file")
			("chain_length,n", value(&gconf.chain_length)->required(), "length of the chain")
			("copies,c", value(&gconf.copies_total)->required(), "number of ensemble realizations")
			("steps,s", value(&gconf.steps)->default_value(0), "number of steps in total (0 for infinite)")
			("entropy,e", value(&gconf.entropy_limit)->default_value(0), "entropy limit")
			("WTlimit", "limit WT entropy")
			("entropymask", value(&entropymask_filename), "mask of linenergies to include in entropy calculation")
			("base,b", value(&gconf.timebase)->default_value(0), "time offset")
			("dt", value(&gconf.dt)->required(), "time delta")
			("grouping,k", value(&gconf.steps_grouping)->required(),
			 "number of steps per kernel invocation, affects granularity of linear energy and entropy dumps")
			("dumpsteps", value(&gconf.steps_grouping_dump),
			 "number of steps between full state dumps (defaults to same value as --grouping)");
}

void parse_cmdline::operator()(int argc, char* argv[]) try {
	store(boost::program_options::parse_command_line(argc, argv, options), vm);
	notify(vm);

	//check user arguments
	gconf.verbose = vm.count("verbose");
	if(gconf.copies_total % mpi_global.size())
		throw invalid_argument("copies must be a multiple of the number of devices");
	gconf.shard_copies = gconf.copies_total / mpi_global.size();
	if(gconf.chain_length < 2 || !gconf.shard_copies || gconf.dt <= 0 || !gconf.steps_grouping)
		throw invalid_argument("chain_length must be >= 2, copies, dt and grouping must be positive numbers");
	if(gconf.entropy_limit < 0)
		throw invalid_argument("entropy limit must be >= 0");
	if(vm.count("entropy"))
		gconf.entropy_limit_type = vm.count("WTlimit") ? configuration::WT : configuration::INFORMATION;
	if(!vm.count("dumpsteps")) gconf.steps_grouping_dump = gconf.steps_grouping;
	if(gconf.steps_grouping_dump % gconf.steps_grouping)
		throw invalid_argument("--dumpsteps must be a multiple of --grouping");
	if(gconf.steps % gconf.steps_grouping_dump || gconf.timebase % gconf.steps_grouping_dump)
		throw invalid_argument("steps and timebase must be a multiple of --dumpsteps");
	if(gconf.timebase && gconf.steps && gconf.timebase >= gconf.steps)
		throw invalid_argument("timebase must be less than steps");

	gconf.linenergy_size = sizeof(double) * gconf.chain_length;
	gconf.shard_elements = size_t(gconf.chain_length) * gconf.shard_copies;
	gconf.shard_size = sizeof(double2) * gconf.shard_elements;
	gres.shard = cudalist<double2>(gconf.shard_elements);
	gres.linenergies = cudalist<double>(gconf.chain_length);
	gres.shard_host = cudalist<double2, true>(gconf.shard_elements, false);
	gres.linenergies_host = cudalist<double, true>(gconf.chain_length, false);
	cudaMemset(gres.linenergies, 0, sizeof(double) * gconf.chain_length) && assertcu;
	memset(gres.linenergies_host, 0, sizeof(double) * gconf.chain_length);
	try {
		ifstream initial_state(initial_filename);
		initial_state.exceptions(ios::failbit | ios::badbit | ios::eofbit);
		initial_state.seekg(gconf.shard_size * mpi_global_coord).read((char*)*gres.shard_host, gconf.shard_size);
	} catch(...) {
		errmsg<<"could not read initial state"<<endl;
		throw;
	}
	if(!entropymask_filename.empty()) try {
		ifstream entropymask_file(entropymask_filename);
		entropymask_file.exceptions(ios::failbit | ios::badbit | ios::eofbit);
		uint16_t mode = 0;
		loopi(gconf.chain_length){
			if(entropymask_file.get()) gconf.entropy_modes_indices.push_back(mode);
			mode++;
		}
	} catch(...) {
		errmsg<<"could not read entropymask file"<<endl;
		throw;
	}
	cudaMemcpy(gres.shard, gres.shard_host, gconf.shard_size, cudaMemcpyHostToDevice) && assertcu;

} catch(const boost::program_options::error& e){
	throw invalid_argument(e.what());
}


int main(int argc, char** argv) {
	//whether catching these signal works is MPI-implementation dependent
	for(int s: { SIGINT, SIGTERM }) signal(s, [](int){ quit_requested = true; });
	try {

		if (!programs().size()) throw logic_error("nlchains compiled without any subprogram!");
		string program_names;
		for (auto kv: programs()) {
			if (!program_names.empty()) program_names += ", ";
			program_names += kv.first;
		}
		if (argc == 1) {
			cerr << "Please specify as first argument the subprogram to run: " << program_names << endl;
			return 0;
		}
		auto program = programs().find(argv[1]);
		if (program == programs().end()) throw invalid_argument("please specific a valid program among " + program_names);

		cudaSetDevice(mpi_node_coord) && assertcu;
		destructor([]{
			destructor(cudaDeviceSynchronize);
			gres = resources();
			cudaDeviceReset();
		});
		{
			cudaDeviceProp dev_props;
			cudaGetDeviceProperties(&dev_props, mpi_node_coord) && assertcu;
			ostringstream info;
			info<<"GPU "<<dev_props.name<<" (id "<<mpi_node_coord<<") on host "<<mpienv.processor_name()
			    <<" clocks core/mem "<<dev_props.clockRate / 1000<<'/'<<dev_props.memoryClockRate / 1000<<endl;
			loopi(mpi_global.size()){
				if(int(i) == mpi_global_coord) cerr<<info.str();
				mpi_global.barrier();
			}
		}

		return program->second(argc - 1, argv + 1);

	} catch(const ios_base::failure& e) {
		errmsg<<"I/O fatal error: "<<e.what()<<" ("<<e.code().message()<<")"<<endl;
	} catch(const system_error& e) {
		errmsg<<"fatal error: "<<e.what()<<" ("<<e.code().message()<<")"<<endl;
	} catch(const std::exception& e) {
		errmsg<<"fatal error ("<<boost::core::demangled_name(typeid(e))<<"): "<<e.what()<<endl;
	} catch(...) {
		errmsg<<"unspecified fatal error"<<endl;
	}
	cerr<<errmsg.str();
	return 1;

}
