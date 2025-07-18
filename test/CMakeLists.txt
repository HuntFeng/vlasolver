cmake_minimum_required(VERSION 3.22)
project(Vlasolver)
find_package(Kokkos REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HighFive REQUIRED)
find_package(GTest REQUIRED)
enable_testing()
add_executable(
  main
  main.cpp
  grid_test.cpp
  ../src/2d2v/grid.cpp
  poisson_test.cpp
  ../src/2d2v/poisson.cpp
  ../src/2d2v/world.cpp
  ../src/2d2v/writer.cpp)
target_link_libraries(main GTest::gtest_main Kokkos::kokkos HighFive INIReader)
include(GoogleTest)
gtest_discover_tests(main)
