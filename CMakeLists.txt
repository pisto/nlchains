cmake_minimum_required(VERSION 3.9)
project(nlchains LANGUAGES CXX CUDA)

if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang")
    message(WARNING "Only GCC and Clang compilers/linkers have been tested.")
endif()

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

add_definitions(-DOMPI_SKIP_MPICXX -DBOOST_ENABLE_ASSERT_HANDLER)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED mpi program_options)
get_filename_component(CUDA_ROOT ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_ROOT ${CUDA_ROOT} DIRECTORY)
include_directories(${CUDA_ROOT}/include cub/ ${MPI_INCLUDE_PATH})

if(CMAKE_BUILD_TYPE MATCHES Rel)
    #aid SIMD vectorization in results_entropy.cpp
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_property(SOURCE results_entropy.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -Ofast ")
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set_property(SOURCE results_entropy.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${OpenMP_CXX_FLAGS} ")
    endif()
endif()

add_subdirectory(kg_fpu_toda)
add_subdirectory(tests)

add_executable(nlchains main.cpp results.cpp utilities_cuda.cu results_entropy.cpp)
set_property(TARGET nlchains PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#these are lld/ld specific options. --whole-archive is needed.
target_link_libraries(nlchains -Xlinker --as-needed ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES}
        -Xlinker --whole-archive nlchains-kg-fpu-toda nlchains-tests -Xlinker --no-whole-archive -lcublas -lcufft)
