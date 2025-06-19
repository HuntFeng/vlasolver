cmake_minimum_required(VERSION 3.22)

project(Vlasolver)
find_package(Kokkos REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HighFive REQUIRED)
add_subdirectory(src/1d1v)
# add_executable(main main.cpp) target_link_libraries(main Kokkos::kokkos)
