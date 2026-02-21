# cmake_minimum_required(VERSION 3.22)
#
# project(Vlasolver) find_package(Kokkos REQUIRED) find_package(OpenMP REQUIRED)
# find_package(HighFive REQUIRED) find_package(GTest REQUIRED) #
# add_subdirectory(src/1d1v) add_subdirectory(src/2d2v) # add_subdirectory(test)

cmake_minimum_required(VERSION 3.22)
project(Vlasolver CXX)

# Find required packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HighFive REQUIRED)

# Create the core library as interface (header-only)
add_library(vlasolver INTERFACE)
target_include_directories(vlasolver
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vlasolver INTERFACE Kokkos::kokkos KokkosKernels::kokkoskernels HighFive INIReader)

# Set common compiler options
target_compile_features(vlasolver INTERFACE cxx_std_20)

# Build provided examples
# add_executable(cylinder examples/cylinder/main.cpp)
# target_link_libraries(cylinder PRIVATE vlasolver)
# add_executable(sheath examples/sheath/main.cpp)
# target_link_libraries(sheath PRIVATE vlasolver)
# add_executable(sheath_rough_wall examples/sheath_rough_wall/main.cpp)
# target_link_libraries(sheath_rough_wall PRIVATE vlasolver)
add_executable(sheath_rough_wall_2nd_order examples/sheath_rough_wall_2nd_order/main.cpp)
target_link_libraries(sheath_rough_wall_2nd_order PRIVATE vlasolver)
# add_executable(poisson examples/poisson/main.cpp)
# target_link_libraries(poisson PRIVATE vlasolver)
# add_executable(poisson_cylinder examples/poisson_cylinder/main.cpp)
# target_link_libraries(poisson_cylinder PRIVATE vlasolver)
add_executable(cylinder_2nd_order examples/cylinder_2nd_order/main.cpp)
target_link_libraries(cylinder_2nd_order PRIVATE vlasolver)

# Print helpful information when configuration is done
message(STATUS "")
message(STATUS "=== Vlasolver Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "To build examples: cmake --build . ")
message(STATUS "To build your own examples:")
message(STATUS "  1. Add to this CMakeLists.txt:")
message(STATUS "     add_executable(your_example path/to/your_example.cpp)")
message(STATUS "     target_link_libraries(your_example PRIVATE vlasolver)")
message(STATUS "  2. Or create your own CMakeLists.txt that finds this package")
message(STATUS "=======================================")
message(STATUS "")
