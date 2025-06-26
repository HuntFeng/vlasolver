#include "world.hpp"
#include <Kokkos_Core.hpp>

World::World(Grid& grid, Kokkos::Array<double, 2> q, Kokkos::Array<double, 2> mu)
    : grid(grid),
      dt(dt),
      q(q),
      mu(mu) {
    // Initialize the views with appropriate dimensions
    auto [nx, ny, nvx, nvy] = grid.ncells;
    f                       = Kokkos::View<double*****>("f", nx, ny, nvx, nvy, 2);
    n                       = Kokkos::View<double***>("n", nx, ny, 2);
    rho                     = Kokkos::View<double**>("rho", nx, ny);
    phi                     = Kokkos::View<double**>("phi", nx, ny);
    E                       = Kokkos::View<double***>("E", nx, ny, 2);
    eps                     = Kokkos::View<double**>("eps", nx, ny);
    a                       = Kokkos::View<double**>("a", nx, ny); // jump condition for poisson
    b                       = Kokkos::View<double**>("b", nx, ny); // jump condition for poisson
    Kokkos::deep_copy(f, 0.0);
    Kokkos::deep_copy(rho, 0.0);
    Kokkos::deep_copy(phi, 0.0);
    Kokkos::deep_copy(E, 0.0);
    Kokkos::deep_copy(eps, 1.0);
    Kokkos::deep_copy(a, 0.0);
    Kokkos::deep_copy(b, 0.0);
}
