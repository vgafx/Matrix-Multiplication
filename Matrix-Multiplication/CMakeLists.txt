cmake_minimum_required(VERSION 2.8)

project(Matrix-Multiplication)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "-O3 -mavx -fopenmp")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DNMC_VECTORIZE_TARGET=AVX2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    main.cpp
    ScopedTimer.cpp)

set(HEADERS
    ScopedTimer.h
    mm_kernels.h)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
