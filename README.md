# nlchains

An implementation of the 6th order Yoshida symplectic integration algorithm for a number of models. The implementation is suited to run a large ensemble of realizations of these models. The program also calculates the average energy per linear mode, and the associated information entropy to monitor the route to thermalization due to the nonlinearity. Since the integration is symplectic, there is no support for forcing or dissipation.

Supported models:

| `nlchains` subprogram | Hamiltonian density | Description                                      |
|-----------------------|---------------------|--------------------------------------------------|
| DNKG                  | ![DNKG][DNKG]       | discrete nonlinear Klein-Gordon, equal masses    |
| DNLS                  | ![DNLS][DNLS]       | discrete nonlinear Schrödinger                   |
| FPUT                  | ![FPUT][FPUT]       | Fermi-Pasta-Ulam-Tsingou, α and β variants       |
| Toda                  | ![Toda][Toda]       | Toda lattice                                     |
| dDNKG                 | ![dDNKG][dDNKG]     | discrete nonlinear Klein-Gordon, per-site masses |

[DNKG]: https://bit.ly/2QLJGW5 "\frac{p_{x}^2}{2}+\frac{\left(q_{x+1}-q_{x}\right)^2}{2}+m\frac{q_{x}^2}{2}+\beta\frac{q_{x}^4}{4}"
[DNLS]: https://bit.ly/2Lb1e8M "|\psi_{x+1}-\psi_x|^2+\beta\frac{|\psi_x|^4}{2}"
[FPUT]: https://bit.ly/2BbLNZB "\frac{p_{x}^2}{2}+\frac{\left(q_{x+1}-q_{x}\right)^2}{2}+\alpha\frac{\left(q_{x+1}-q_{x}\right)^3}{3}+\beta\frac{\left(q_{x+1}-q_{x}\right)^4}{4}"
[Toda]: https://bit.ly/2BbwgsK "\frac{p_{x}^2}{2}+\frac{1}{4 \alpha ^2}\left(e^{2 \alpha  \left(q_{x+1}-q_{x}\right)}-2 \alpha  \left(q_{x+1}-q_{x}\right)-1\right)"
[dDNKG]: https://bit.ly/2Ee2xlQ "\frac{p_{x}^2}{2}+\frac{\left(q_{x+1}-q_{x}\right)^2}{2}+m_x\frac{q_{x}^2}{2}+\beta\frac{q_{x}^4}{4}"

# Prerequisites and building

This software has been tested primarily on CUDA 9.0 and K40 devices on Linux. It is expected to work on later CUDA toolchains and GPU architectures. Mac has not been tested but it should in theory work. Windows is not supported, because cuFFT callbacks are used and they are not available in Windows.

The build prerequisites are: [CMake](https://cmake.org/) (>=3.9), MPI (minimum MPI-2, Open MPI 2.x has been tested), [Boost](https://www.boost.org/) (Boost.Program_options, Boost.MPI), [Armadillo](http://arma.sourceforge.net/). From the CUDA Toolkit, we use cuFFT and cuBLAS. For an Ubuntu environment, the requires packages should be `libarmadillo-dev libboost-mpi-dev libboost-program-options-dev cmake`, while for a Fedora environment they should be `armadillo-devel boost-openmpi-devel cmake openmpi-devel`.

## Building with Docker

Since the interaction between host operating system, compiler version and CUDA SDK version can be complex, some Dockerfiles are provided in order to build `nlchains` in a Docker environment. Simply run one of the following:
```bash
#CUDA 9.0, Fedora
docker build -f Dockerfile-fedora25-cuda90 --build-arg optimized_chain_length=XX .
#CUDA 10.0, Fedora
docker build -f Dockerfile-fedora27-cuda100 --build-arg optimized_chain_length=XX .
#CUDA 10.0, Ubuntu
docker build -f Dockerfile-ubuntu1804-cuda100 --build-arg optimized_chain_length=XX .
```
Please note that the parameter `--build-arg optimized_chain_length=XX` with XX some number greater than 32 is necessary, because some optimized versions of the kernels need to be generated with a compile-time constant of the intendend value of the chain length. The generated binary can run any other value of the chain length, but without any specific optimization. For more information read section [Performance considerations](#performance-considerations).

## Building manually

To build nchains, run the following
```bash
git clone --recurse https://pisto@bitbucket.org/pisto/nlchains.git
cd nlchains
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70" -Doptimized_chain_length=XX ..
make -j
```
Please note that the parameter `-Doptimized_chain_length=XX` with XX some number greater than 32 is necessary, because some optimized versions of the kernels need to be generated with a compile-time constant of the intendend value of the chain length. The generated binary can run any other value of the chain length, but without any specific optimization. For more information read section [Performance considerations](#performance-considerations).

Other relevant CMake flags (`-DFlag=Value`) are listed in the table below.

| Flag                  | Value                                                                           |
|:--------------------- |:------------------------------------------------------------------------------- |
| `MPI_CXX_COMPILER`    | If not in `$PATH`, path to your MPI C++ compiler                                |
| `MPI_C_COMPILER`      | If not in `$PATH`, path to your MPI C compiler                                  |
| `CMAKE_CUDA_COMPILER` | If not in `$PATH`, path to the CUDA nvcc                                        |
| `CMAKE_CUDA_FLAGS`    | Flags for the GPU code generation, e.g. `-arch=sm_35`                           |

# Launching a simulation

A `nlchains` invokation looks like this
```
[MPI launcher] nlchains <model> <common options> <model-specific options>
```
`nlchains` can be launched stand-alone to use one GPU on the current host, or through your MPI implementation to use multiple GPUs across multiple nodes. When multiple GPUs are present, `nlchains` splits equally the number of copies of the ensemble among the available GPUs. There is no way to divide the work in unequal shards: all GPUs should have essentially the same computational speed (check the clocks with `nvidia-smi`!), otherwise the faster GPUs will run at the pace of the slower ones.

When launching through MPI, you must set one process for each GPU on each host. Within a host, `nlchains` calculates the number of local processes, and uses that many GPUs, starting from the CUDA GPU index 0. If you want to select some specific GPUs you can use the environment variable [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars).

All `nlchains` dumps are in binary format (64bit double, or 128bit complex double). The dump file can have a prefix. This is the list of dumps generated:

| Name                              | Content                                                                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `<prefix>-<timestep>`             | Full dump of the ensemble state at the specified timestep, as a C++ array: `double[copy][chain_index][2]` or ` complex<double>[copy][chain_index]`  |
| `<prefix>-linenergies-<timestep>` | Linear energy per mode at the specified timestep                                                                                                    |
| `<prefix>-entropy`                | List of tuples `{ time, Wave Turbulence entropy, information entropy }`                                                                             |
| `<prefix>-eigenvectors`           | (dDNKG only): eigenvectors                                                                                                                          |
| `<prefix>-omegas`                 | (dDNKG): square root of the eigenvalues (pulsation of the eigenvectors)                                                                             |

The program is divided into subprograms. The first argument must always be the name of the subprogram, that is one of `DNKG`, `DNLS`, `FPUT`, `Toda`, `dDNKG`. The common options are the following:
```
  -v [ --verbose ]              print extra informations
  -i [ --initial ] arg          initial state file
  -n [ --chain_length ] arg     length of the chain
  -c [ --copies ] arg           number of realizations of the chain in the 
                                ensemble
  --dt arg                      time step value
  -b [ --kernel_batching ] arg  number of steps per kernel invocation
  -s [ --steps ] arg (=0)       number of steps in total (0 for infinite)
  -o [ --time_offset ] arg (=0) time offset in dump files
  -e [ --entropy ] arg (=0)     entropy limit
  --WTlimit                     limit WT entropy instead of information entropy
  --entropymask arg             mask of linenergies to include in entropy 
                                calculation
  -p [ --prefix ] arg           prefix for dump files
  --dump_interval arg           number of steps between full state dumps 
                                (defaults to same value as --batch, does not 
                                affect other dump or entropy time granularity)
```
The initial state file has the same format of the full dump. The argument to `--entropymask` is a file that contains a byte for each mode in the system, and if non-zero then the mode is included in the calculation of the entropy.

Each model has a few extra argument:
```
  -m arg                        linear parameter m (DNLG), linear parameters m filename (dDNLG)
  --alpha arg                   third order nonlinearity (FPUT), exponential steepness (Toda)
  --beta arg                    fourth order nonlinearity (DNLG, FPUT, dDNLG, DNLS)
  --split_kernel                force use of split kernel (DNLG, FPUT, Toda, dDNLG)
  --no_linear_callback          do not use cuFFT callback for linear evolution (DNLS)
  --no_nonlinear_callback       do not use cuFFT callback for nonlinear evolution (DNLS)
```
For an explanation of the `--split_kernel`, `--no_linear_callback` and `--no_nonlinear_callback` arguments, read the [Performance considerations](#performance-considerations).

A typical invocation of `nlchains` to run on two GPUs looks like this:
```
mpirun -n 2 nlchains FPUT -v -p alpha-N128-alpha1 -i alpha-N128 -n 128 -c 4096 --WTlimit -e 0.05 --dt 0.1 -k 10000 --dumpsteps 1000000 --alpha 1 --beta 0
```

# Software internals

In order to help the user interested in adding or modifying the models implemented in nlchains, here are a few notes on the software structure. In the folder [common/](common/) there is the logic that is common to all models: MPI environment setup, common argument parsing, GPU setup, symplectic integration coefficients, etc. Subprograms are expected to add themselves to the map returned by `::programs()` `::main` runs (see the macro `::ginit` in [common/utilities.hpp](common/utilities.hpp)). The helper struct `::parse_cmdline` can be used to parse the command line, and by default it contains the definition of the common command line options. The actual values of the options are stored after parsing in the global variable `::gconf`, and `::gres` contains host and device buffers to accomodate the shard of the ensemble that is owned by the current MPI process. The struct `::results` contains facilities to calculate the linear energies per mode through MPI reduction operations, calculate the entropy, and write the dump files. The folders [DNKG_FPUT_Toda/](DNKG_FPUT_Toda/), [dDNKG/](dDNKG/) and [DNLS/](DNLS/) contain the actual implementation of the predefined models. Each of these folders contain a .cpp source file with most of the host-side logic, and a .cu source file with the GPU kernel. The DNKG, FPUT and Toda models differ only by the actual equation of motion, hence they are implemented as different class templates in the same folder.

## Performance considerations

The compile flag `-Doptimized_chain_length=XX` and the command line option `--split_kernel` are relevant for all models except `DNLS`. There are internally three GPU implementations of each of these models: one specialized for `--chain_length` less than 32, one optimized for the value passed to `optimized_chain_length`, and one generic. The generic version is referred to as the "split kernel" version. The split kernel is the slowest one because every nonlinear chain element update causes a read from memory, but it is flexible, and works with very large values of `--chain_length`. The other two are much more optimized since they keep the whole chain state in registers rather than memory. It is advised to recompile `nlchains` with a different value of `optimized_chain_length` for each target `--chain_length` that you plan to use. If `nlchains` cannot find the optimized version, it will fallback to the split kernel version, generating a warning.

The command line options `--no_linear_callback` and `--no_nonlinear_callback` are relevant to the `DNLS` model. Since the integration amounts to a large number of FFTs, it is generally useful to use the cuFFT feature of FFT callbacks, and embed the linear/nonlinear evolution of the chain in the load phase of the FFTs. However, in my experience the cuFFT callbacks can on the contrary lead to a performance hit in some circumstances. By default, `nlchains` uses them, but you should experiment with these flags and see what is best suited for you. Do not forget to set the clocks of your card to the maximum available with `nvidia-smi` first!

# Known bugs

Because of a bug in older versions of CMake, the MPI compile flags (`MPI_CXX_COMPILE_OPTIONS`) are not honored. See `CMakeLists.txt` for more details.

Because of a bug in `gcc` version 5.\* or 6.\*, opening a non-existen file for the initial state, or the mass parameter configuration in the `dDNKG` model may return a rather generic message:
```
terminate called after throwing an instance of 'std::ios_base::failure'
  what():  basic_ios::clear
```

When run interactively, the program may not close gracefully by interrupting it, e.g. with CTRL-C. This is due to a limitation of the MPI implementation, that may or may not propagate termination signals correctly to all the MPI processes.

On OpenMPI 1.x termination through signals (SIGTERM/SIGINT) does not terminate the MPI processes gracefully because SIGKILL is almost immediately sent after SIGTERM, effectively ignoring the value of MCA `odls_base_sigkill_timeout`. When running in single process mode signal hadling works normally.
