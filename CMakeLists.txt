cmake_minimum_required(VERSION 3.14)
project(llm.cpp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(BUILD_SHARED_LIBS OFF)
# add_compile_options(-Ofast -march=native)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif ()

enable_testing()
include_directories(.)

# Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

# GoogleTest
add_subdirectory(third_party/googletest)

# Eigen
set(EIGEN3_INCLUDE_DIR third_party/eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(llmc)
add_subdirectory(llmcpp)
