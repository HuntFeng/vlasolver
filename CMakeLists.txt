cmake_minimum_required(VERSION 3.28)

project(Vlasolver)

# Try to find KokkosKernels (this will also find Kokkos) Make sure when you run
# CMake you give it -DKokkosKernels_ROOT=<KokkosKernels_Prefix>
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(poisson poisson.cpp)
target_link_libraries(poisson OpenMP::OpenMP_CXX Kokkos::kokkoskernels
                      Kokkos::kokkos)
