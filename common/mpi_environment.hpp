#pragma once

#include <string>
#include <sstream>
#include <functional>
#include <map>
#include <csignal>
#include <boost/mpi.hpp>
#include "configuration.hpp"

/*
 * MPI environment.
 */

extern const boost::mpi::environment mpienv;
extern const boost::mpi::communicator mpi_global, mpi_global_alt;
extern const int mpi_global_coord, mpi_node_coord;
extern volatile sig_atomic_t quit_requested;

/*
 * MPI return code errors to exceptions, use as following:
 *
 * MPI_Statement(...) && assertmpi
 *
 * Non zero return codes will be turned into a boost::mpi::exception with a more useful file:line location message
 * rather than routine name.
 */

#define assertmpi boost::mpi::exception_location::record{ __FILE__ ":" + std::to_string(__LINE__) }

namespace boost {
	namespace mpi {
		struct exception_location : exception {
			using exception::exception;
			struct record {
				std::string file_line;
			};

			virtual const char *what() const noexcept { return msg.c_str(); }

		protected:
			exception_location(const record &r, int result_code) :
					exception("<unknown routine>", result_code), msg("@" + r.file_line + ": " + exception::what()) {}

			friend void operator&&(int result_code, exception_location::record &&p);

			const std::string msg;
		};

		inline void operator&&(int result_code, exception_location::record &&p) {
			if (!result_code) return;
			throw boost::mpi::exception_location(p, result_code);
		}

	}
}

extern const std::string process_ident;

/*
 * Associate subprograms to --program argument.
 */

std::map<std::string, std::function<int(int argc, char *argv[])>> &programs();
