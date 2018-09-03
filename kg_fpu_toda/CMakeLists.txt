cmake_minimum_required(VERSION 3.9)
project(nlchains-kg-fpu-toda LANGUAGES CXX CUDA)

set(optimized_chain_length 0 CACHE STRING
        "Create optimized kernels for optimized_chain_length")
if(NOT optimized_chain_length MATCHES "^[0-9]+$" OR optimized_chain_length LESS 32)
    message(FATAL_ERROR "please specify -Doptimized_chain_length= as a number >= 32")
endif()
set_source_files_properties(kg_fpu_toda.cu PROPERTIES COMPILE_DEFINITIONS optimized_chain_length=${optimized_chain_length})

add_library(nlchains-kg-fpu-toda STATIC kg_fpu_toda.cpp kg_fpu_toda.cu)

set_property(TARGET nlchains-kg-fpu-toda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nlchains-kg-fpu-toda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
