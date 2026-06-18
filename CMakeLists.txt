cmake_minimum_required(VERSION 3.22)
project(Vlasolver CXX)

# options
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# find required packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HighFive REQUIRED)

# create the core library as interface (header-only)
add_library(vlasolver INTERFACE)
target_include_directories(vlasolver
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vlasolver INTERFACE Kokkos::kokkos KokkosKernels::kokkoskernels HighFive INIReader)

# set common compiler options
target_compile_features(vlasolver INTERFACE cxx_std_20)

# build provided examples
add_executable(advection examples/advection/main.cpp)
target_link_libraries(advection PRIVATE vlasolver)
add_executable(cylinder examples/cylinder/main.cpp)
target_link_libraries(cylinder PRIVATE vlasolver)
add_executable(sheath examples/sheath/main.cpp)
target_link_libraries(sheath PRIVATE vlasolver)
add_executable(sheath_rough_wall examples/sheath_rough_wall/main.cpp)
target_link_libraries(sheath_rough_wall PRIVATE vlasolver)
add_executable(sheath_oml examples/sheath_oml/main.cpp)
target_link_libraries(sheath_oml PRIVATE vlasolver)
add_executable(poisson examples/poisson/main.cpp)
target_link_libraries(poisson PRIVATE vlasolver)
add_executable(star examples/star/main.cpp)
target_link_libraries(star PRIVATE vlasolver)

# print helpful information when configuration is done
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
