#include "world.hpp"

World::World(Grid& grid, Kokkos::Array<double, 2> q, Kokkos::Array<double, 2> mu)
    : grid(grid),
      q(q),
      mu(mu) {
    // Initialize the views with appropriate dimensions
    auto [nx, nv] = grid.extents();
    f             = Kokkos::View<double***>("f", nx, nv, 2);
    rho           = Kokkos::View<double*>("rho", nx);
    phi           = Kokkos::View<double*>("phi", nx);
    E             = Kokkos::View<double*>("E", nx);
}
