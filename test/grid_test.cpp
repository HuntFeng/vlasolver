#include "../include/reduced/grid.hpp"
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(grid_test, spacing) {
    int ngc                                 = 3;
    Kokkos::Array<double, DIM> origin       = {0.0, 0.0, -5.0, -5.0}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1.0, 1.0, 10.0, 10.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {10, 100, 10, 100}; // number of cells in the grid (excluding ghost cells)
    Grid grid(ncells_interior, ngc);
    grid.set_grid(origin, size, 0);

    EXPECT_DOUBLE_EQ(grid.spacing(0, 0)[0], 0.1);
    EXPECT_DOUBLE_EQ(grid.spacing(0, 0)[1], 0.01);
    EXPECT_DOUBLE_EQ(grid.spacing(0)[2], 1.0);
    EXPECT_DOUBLE_EQ(grid.spacing(0)[3], 0.1);
}
