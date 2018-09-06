#include <iostream>
#include <boost/multi_array.hpp>
#include "../utilities.hpp"
#include "../mpi_environment.hpp"
#include "../results.hpp"

using namespace std;

namespace tests {

	int tests(int argc, char* argv[]){

		using namespace boost::program_options;
		parse_cmdline parser("Options for tests");
		try { parser(argc, argv); }
		catch(const invalid_argument& e) {
			if(!mpi_global_coord) cerr<<"Error in command line: "<<e.what()<<endl<<parser.options<<endl;
			return 1;
		}

		boost::multi_array<double2, 2> original(boost::extents[gconf.shard_copies][gconf.chain_length]);
		memcpy(original.data(), gres.shard_host, sizeof(double2) * original.num_elements());
		memset(gres.shard_host, 0, sizeof(double2) * original.num_elements());

		plane2split p2s(gconf.chain_length, gconf.shard_copies);
		p2s.split(gres.shard);
		vector<double> real(p2s.elements * 2);
		auto img = real.data() + p2s.elements;
		cudaMemcpy(real.data(), p2s.real_transposed, sizeof(double) * p2s.elements * 2, cudaMemcpyDeviceToHost);
		p2s.plane(gres.shard_host);
		loopi(gconf.shard_copies) loopj(gconf.chain_length) {
				auto correct = original[i][j];
				auto idx = i * gconf.chain_length + j, transposed_idx = j * gconf.shard_copies + i;
				if (correct.x != real[transposed_idx] || correct.y != img[transposed_idx])
					throw logic_error("failed split format");
				if (correct.x != gres.shard_host[idx].x || correct.y != gres.shard_host[idx].y)
					throw logic_error("failed planar format");
			}

		results(0).write_shard(0, gres.shard_host);

		return 0;
	}
}

ginit = []{ programs()["tests"] = tests::tests; };
