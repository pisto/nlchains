#pragma once

#include <string>
#include <boost/program_options.hpp>
#include <fftw3.h>
#include "mpi_environment.hpp"

struct wisdom_sync {

	bool do_sync = true;

	void add_options(boost::program_options::options_description &options) {
		options.add_options()
				("no_wisdom_sync", boost::program_options::bool_switch()->notifier([=](bool set) { do_sync = !set; }),
				 "do not sync fftw wisdom across MPI processes");
	}

	void gather() const {
		if (!do_sync || !mpi_global_coord) return;
		std::string wisdom;
		boost::mpi::broadcast(mpi_global, wisdom, 0);
		fftw_forget_wisdom();
		fftw_import_wisdom_from_string(wisdom.c_str());
	}

	void scatter() const {
		if (!do_sync || mpi_global_coord) return;
		std::string wisdom = fftw_export_wisdom_to_string();
		boost::mpi::broadcast(mpi_global, wisdom, 0);
	}

};
