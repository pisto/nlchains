cmake_minimum_required(VERSION 3.9)
project(nlchains-dnls LANGUAGES CXX CUDA)

add_library(nlchains-dnls STATIC dnls.cpp dnls.cu)
target_link_libraries(nlchains-dnls -lcufft)

set_property(TARGET nlchains-dnls PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nlchains-dnls PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
