cmake_minimum_required(VERSION 3.15)
project(vector_db_from_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library: everything the executables share. Pure C++17 standard
# library -- no external numerical/ANN/ML dependencies.
add_library(vectordb_core
    src/metrics.cpp
    src/exact_index.cpp
    src/kmeans.cpp
    src/ivf_index.cpp
    src/dataset.cpp
)
target_include_directories(vectordb_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Some standard library implementations need -lstdc++fs on older GCC for
# std::filesystem; harmless no-op on modern toolchains/MSVC.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(vectordb_core PUBLIC stdc++fs)
endif()

add_executable(generate_data generate_data.cpp)
target_link_libraries(generate_data PRIVATE vectordb_core)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE vectordb_core)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE vectordb_core)

add_executable(tests tests/tests.cpp)
target_link_libraries(tests PRIVATE vectordb_core)

enable_testing()
add_test(NAME unit_tests COMMAND tests)
