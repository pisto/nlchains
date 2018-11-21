cmake_minimum_required(VERSION 3.9)
project(nlchains LANGUAGES CXX CUDA)

if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    message(WARNING "Only the GCC compiler has been tested.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


#libraries and paths
#CUDA cub (maybe here as a sub-module, maybe system-wise)
include_directories(cub/)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED mpi program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
set_source_files_properties(kg_disorder/kg_disorder.cpp PROPERTIES INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIRS})

#we use the host compiler directly so we need to extract the cuda include path
get_filename_component(CUDA_ROOT ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_ROOT ${CUDA_ROOT} DIRECTORY)
include_directories(${CUDA_ROOT}/include)


#debug/optimization
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()
if(CMAKE_BUILD_TYPE MATCHES Rel)
    #aid SIMD vectorization in results_entropy.cpp
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_property(SOURCE common/results_entropy.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -Ofast ")
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set_property(SOURCE common/results_entropy.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${OpenMP_CXX_FLAGS} ")
    endif()
endif()

#targeted optimization for different values of -chain_length
set(optimized_chain_length 0 CACHE STRING
        "Create optimized kernels for optimized_chain_length")
if(NOT optimized_chain_length MATCHES "^[0-9]+$" OR optimized_chain_length LESS 32)
    message(FATAL_ERROR "please specify -Doptimized_chain_length= as a number >= 32")
endif()
set_source_files_properties(kg_fpu_toda/kg_fpu_toda.cu kg_disorder/kg_disorder.cu PROPERTIES COMPILE_DEFINITIONS optimized_chain_length=${optimized_chain_length})


#link/build flags
add_executable(nlchains
        common/main.cpp common/results.cpp common/utilities_cuda.cu common/results_entropy.cpp
        kg_fpu_toda/kg_fpu_toda.cpp kg_fpu_toda/kg_fpu_toda.cu
        kg_disorder/kg_disorder.cpp kg_disorder/kg_disorder.cu
        dnls/dnls.cpp dnls/dnls.cu)
set_property(TARGET nlchains PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nlchains ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${ARMADILLO_LIBRARIES} -lcublas -lcufft_static -lculibos)
#XXX workaround for bad CMake support for device link libraries, see https://gitlab.kitware.com/cmake/cmake/issues/18599 https://gitlab.kitware.com/cmake/cmake/issues/18614
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "${CMAKE_CUDA_DEVICE_LINK_EXECUTABLE} -lcufft_static")

#boost and MPI tend to pull in much useless stuff in on different platforms
add_definitions(-DOMPI_SKIP_MPICXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker --as-needed")
endif()
