#include "world.hpp"
#include <Kokkos_Core.hpp>

World::World(Grid& grid, Kokkos::Array<double, 2> q, Kokkos::Array<double, 2> mu)
    : grid(grid),
      q(q),
      mu(mu) {
    // Initialize the views with appropriate dimensions
    auto [nx, nv] = grid.ncells;
    f             = Kokkos::View<double***>("f", nx, nv, 2);
    rho           = Kokkos::View<double*>("rho", nx);
    phi           = Kokkos::View<double*>("phi", nx);
    E             = Kokkos::View<double*>("E", nx);
    eps           = Kokkos::View<double*>("eps", nx);
    a             = Kokkos::View<double*>("a", nx); // jump condition for poisson
    b             = Kokkos::View<double*>("b", nx); // jump condition for poisson
    Kokkos::deep_copy(f, 0.0);
    Kokkos::deep_copy(rho, 0.0);
    Kokkos::deep_copy(phi, 0.0);
    Kokkos::deep_copy(E, 0.0);
    Kokkos::deep_copy(eps, 1.0);
    Kokkos::deep_copy(a, 0.0);
    Kokkos::deep_copy(b, 0.0);
}
