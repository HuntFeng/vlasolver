cmake_minimum_required(VERSION 3.22)

project(Vlasolver)
find_package(Kokkos REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HighFive REQUIRED)
add_executable(vlasov_2d2v vlasov.cpp)
target_link_libraries(vlasov_2d2v Kokkos::kokkos HighFive)
