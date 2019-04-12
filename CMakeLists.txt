cmake_minimum_required(VERSION 3.9)
project(nlchains LANGUAGES CXX CUDA)

if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    message(WARNING "Only the GCC compiler has been tested under Linux.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


#libraries and paths

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    link_libraries(-Xlinker --as-needed)
endif()

#CUDA cub (maybe here as a sub-module, maybe system-wise)
include_directories(cub/)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_definitions(-DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX)     #try not to reference MPI C++ bindings
string(STRIP "${MPI_CXX_COMPILE_OPTIONS}" MPI_CXX_COMPILE_OPTIONS)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
    add_compile_options("${MPI_CXX_COMPILE_OPTIONS}")
elseif(MPI_CXX_COMPILE_OPTIONS MATCHES "[^ ]")
    #XXX MPI_CXX_COMPILE_OPTIONS may contain -pthread, and that is not consumed by nvcc without -Xcompiler.
    #https://gitlab.kitware.com/cmake/cmake/issues/18008
    #if you need this line just patch it manually
    message(WARNING "Leaving out MPI flags ${MPI_CXX_COMPILE_OPTIONS} because of a CMake bug (https://gitlab.kitware.com/cmake/cmake/issues/18008), add them manually if you really need them")
endif()

add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED mpi program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
set_source_files_properties(dDNKG/dDNKG.cpp PROPERTIES INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIRS})

#we use the host compiler directly so we need to extract the cuda include path
get_filename_component(CUDA_ROOT ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_ROOT ${CUDA_ROOT} DIRECTORY)
include_directories(${CUDA_ROOT}/include)

#XXX this fails hard on distro that don't build FFTW3 with CMake
#find_package(FFTW3)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW3 REQUIRED fftw3)
include_directories(${FFTW3_INCLUDE_DIRS})


#debug/optimization
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()
if(CMAKE_BUILD_TYPE MATCHES Rel)
    #aid SIMD vectorization
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_property(SOURCE common/results_entropy.cpp DNKG_FPUT_Toda/cpu-implementation/DNKG_FPUT_Toda.cpp dDNKG/cpu-implementation/dDNKG.cpp DNLS/cpu-implementation/DNLS.cpp APPEND PROPERTY COMPILE_FLAGS "-Ofast")
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set_property(SOURCE common/results_entropy.cpp DNKG_FPUT_Toda/cpu-implementation/DNKG_FPUT_Toda.cpp dDNKG/cpu-implementation/dDNKG.cpp DNLS/cpu-implementation/DNLS.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${OpenMP_CXX_FLAGS} ")
    endif()
endif()

#targeted optimization for different values of -chain_length
set(optimized_chain_length 0 CACHE STRING "Create optimized kernels for optimized_chain_length")
if(NOT optimized_chain_length MATCHES "^[0-9]+$" OR optimized_chain_length LESS 32)
    message(FATAL_ERROR "please specify -Doptimized_chain_length= as a number >= 32")
endif()
set_source_files_properties(DNKG_FPUT_Toda/DNKG_FPUT_Toda.cu dDNKG/dDNKG.cu PROPERTIES COMPILE_DEFINITIONS optimized_chain_length=${optimized_chain_length})


#link/build flags
add_executable(nlchains
        common/main.cpp common/results.cpp common/utilities_cuda.cu common/utilities_cuda.cpp common/results_entropy.cpp
        DNKG_FPUT_Toda/DNKG_FPUT_Toda.cpp DNKG_FPUT_Toda/DNKG_FPUT_Toda.cu DNKG_FPUT_Toda/cpu-implementation/DNKG_FPUT_Toda.cpp DNKG_FPUT_Toda/dispersion.cpp
        dDNKG/eigensystem.cpp dDNKG/dDNKG.cpp dDNKG/dDNKG.cu dDNKG/cpu-implementation/dDNKG.cpp
        DNLS/DNLS.cpp DNLS/DNLS.cu DNLS/dispersion.cpp DNLS/cpu-implementation/DNLS.cpp)

set_property(TARGET nlchains PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#XXX workaround for bad CMake support for device link libraries, see https://gitlab.kitware.com/cmake/cmake/issues/18599 https://gitlab.kitware.com/cmake/cmake/issues/18614
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> <FLAGS> -Xcompiler=-fPIC -Wno-deprecated-gpu-targets -shared -dlink <OBJECTS> -o <TARGET> -lcufft_static")

target_link_libraries(nlchains
        ${Boost_LIBRARIES}
        ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        -lcublas -lcufft_static -lculibos
        ${FFTW3_LIBRARY_DIRS} ${FFTW3_LIBRARIES})
