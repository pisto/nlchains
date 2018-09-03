cmake_minimum_required(VERSION 3.9)
project(nlchains-tests LANGUAGES CXX CUDA)

add_library(nlchains-tests STATIC tests.cpp tests.cu)

set_property(TARGET nlchains-tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nlchains-tests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
