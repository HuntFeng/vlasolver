cmake_minimum_required(VERSION 3.28)

project(Vlasolver)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gridformat REQUIRED)
add_executable(poisson src/poisson.cpp)
target_link_libraries(poisson OpenMP::OpenMP_CXX Kokkos::kokkoskernels
                      Kokkos::kokkos gridformat::gridformat)
# add_executable(poisson_1d poisson_1d.cpp) target_link_libraries(poisson_1d
# OpenMP::OpenMP_CXX Kokkos::kokkoskernels Kokkos::kokkos
# gridformat::gridformat)
