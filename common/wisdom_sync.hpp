#pragma once

#include <string>
#include <iostream>
#include <boost/program_options.hpp>
#include <fftw3.h>
#include "utilities.hpp"
#include "mpi_environment.hpp"

struct wisdom_sync {

	const int &fftw_flags = _fftw_flags;

	void add_options(boost::program_options::options_description &options) {
		options.add_options()
				("wisdom_mode", boost::program_options::value(&wisdom_mode)->default_value("sync"),
				 "Mode for wisdom syncing across MPI processes: \"sync\" calculates wisdom once and propagates to all processes, \"node\" does not propagate wisdom, other values are interpreted as filenames to read wisdom from");
	}

	void gather() const {
		fftw_forget_wisdom();
		if (wisdom_mode == "sync") {
			if (!mpi_global_coord) return;
			std::string wisdom;
			boost::mpi::broadcast(mpi_global, wisdom, 0);
			fftw_import_wisdom_from_string(wisdom.c_str());
			_fftw_flags = FFTW_WISDOM_ONLY;
		} else if (wisdom_mode != "none") {
			fftw_import_wisdom_from_filename(wisdom_mode.c_str());
			_fftw_flags = FFTW_WISDOM_ONLY;
		}
	}

	void scatter() const {
		std::string wisdom = fftw_export_wisdom_to_string();
		if (wisdom_mode == "sync" && !mpi_global_coord) boost::mpi::broadcast(mpi_global, wisdom, 0);
		if (gconf.verbose) {
			if (wisdom_mode == "none")
				collect_ostream(std::cerr) << process_ident << ": FFTW wisdom:" << std::endl << wisdom << std::endl;
			else if (!mpi_global_coord)
				collect_ostream(std::cout) << "FFTW wisdom:" << std::endl << wisdom << std::endl;
		}
	}

private:

	mutable int _fftw_flags = 0;
	std::string wisdom_mode;

};
