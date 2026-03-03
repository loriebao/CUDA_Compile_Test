# This searches for the offensive flag in the CXX flags and removes it
string(REPLACE "-fdiagnostics-color=always" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fdiagnostics-color=always" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

cmake_minimum_required(VERSION 3.18) # 3.18+ has better CUDA support

# Set the target GPU architecture (e.g., 80 for Ampere, 89 for Ada)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(HelloWorld LANGUAGES CXX CUDA)

add_executable(cuda_app HelloWorld.cu)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuda_app 
    PRIVATE 
    CUDA::cudart 
    CUDA::cublas   # This resolves the cublasCreate_v2 symbol
)

# set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
# set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
#target_include_directories(cuda_app PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 4. Link the libraries
# Use the CUDA:: targets provided by find_package(CUDAToolkit)
# target_link_libraries(cuda_app 
#     PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
# )

