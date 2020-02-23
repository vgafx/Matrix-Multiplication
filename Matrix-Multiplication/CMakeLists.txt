cmake_minimum_required(VERSION 2.8)

project(Matrix-Multiplication)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    main.cpp
    ScopedTimer.cpp)

set(HEADERS
    ScopedTimer.h)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

