cmake_minimum_required(VERSION 3.9)
project(nlchains-kg-disorder LANGUAGES CXX CUDA)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(optimized_chain_length 0 CACHE STRING
        "Create optimized kernels for optimized_chain_length")
if(NOT optimized_chain_length MATCHES "^[0-9]+$" OR optimized_chain_length LESS 32)
    message(FATAL_ERROR "please specify -Doptimized_chain_length= as a number >= 32")
endif()
set_source_files_properties(kg_disorder.cu PROPERTIES COMPILE_DEFINITIONS optimized_chain_length=${optimized_chain_length})

add_library(nlchains-kg-disorder STATIC kg_disorder.cpp kg_disorder.cu)
target_link_libraries(nlchains-kg-disorder ${ARMADILLO_LIBRARIES} -lcublas)

set_property(TARGET nlchains-kg-disorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nlchains-kg-disorder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
