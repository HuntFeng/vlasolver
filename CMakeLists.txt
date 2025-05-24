cmake_minimum_required(VERSION 3.28)

message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
cmake_policy(SET CMP0074 NEW)

project(MyProgram)

# Try to find KokkosKernels (this will also find Kokkos) Make sure when you run
# CMake you give it -DKokkosKernels_ROOT=<KokkosKernels_Prefix>
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(ex_real_A ex_real_A.cpp)
target_link_libraries(ex_real_A OpenMP::OpenMP_CXX Kokkos::kokkoskernels
                      Kokkos::kokkos)
