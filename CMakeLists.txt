cmake_minimum_required(VERSION 3.14)
project(liftlab_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standalone Catch2 tests for the header-only kernels (no Python needed).
include(FetchContent)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)
add_executable(cpp_tests cpp/tests/test_resample.cpp)
target_include_directories(cpp_tests PRIVATE cpp/include)
target_link_libraries(cpp_tests PRIVATE Catch2::Catch2WithMain Threads::Threads)

enable_testing()
add_test(NAME cpp_tests COMMAND cpp_tests)
