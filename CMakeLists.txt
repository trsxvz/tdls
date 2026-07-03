# TDLS is header-only: this file compiles nothing for the library itself.
# It declares the tdls::tdls INTERFACE target consumed through
# add_subdirectory()/FetchContent (include path + C++17 floor), and
# orchestrates the tests. There is deliberately no install step: consume
# the target, or copy include/.
cmake_minimum_required(VERSION 3.21)

project(tdls VERSION 0.1.0 LANGUAGES CXX)

# The version declared here and the one in include/tdls/core/version.hpp
# must never drift apart.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/tdls/core/version.hpp" TDLS_VERSION_HEADER)
string(REGEX MATCH "TDLS_VERSION_STRING \"([0-9.]+)\"" _ "${TDLS_VERSION_HEADER}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR "version mismatch: version.hpp says '${CMAKE_MATCH_1}', "
                        "project() says '${PROJECT_VERSION}'")
endif()

add_library(tdls INTERFACE)
add_library(tdls::tdls ALIAS tdls)
target_include_directories(tdls INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tdls INTERFACE cxx_std_17)

# Tests are built when tdls is the top-level project, and stay out of the
# way when tdls is consumed by another build.
if(PROJECT_IS_TOP_LEVEL)
    option(TDLS_BUILD_TESTS "Build the tdls test suite" ON)
else()
    option(TDLS_BUILD_TESTS "Build the tdls test suite" OFF)
endif()

if(PROJECT_IS_TOP_LEVEL)
    option(TDLS_BUILD_EXAMPLES "Build the tdls examples" ON)
else()
    option(TDLS_BUILD_EXAMPLES "Build the tdls examples" OFF)
endif()

if(TDLS_BUILD_TESTS OR TDLS_BUILD_EXAMPLES)
    enable_testing()
endif()
if(TDLS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(TDLS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
