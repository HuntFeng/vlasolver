#include "world.hpp"
#include <Kokkos_Core.hpp>

World::World(Grid& grid)
    : grid(grid) {
    // Initialize the views with appropriate dimensions
    auto [nx, ny, nvx, nvy] = grid.ncells;
    f                       = Kokkos::View<double****>("f", nx, ny, nvx, nvy);
    flux                    = Kokkos::View<double****>("flux", nx, ny, nvx, nvy);
    flux_l                  = Kokkos::View<double****>("flux_l", nx, ny, nvx, nvy);
    flux_r                  = Kokkos::View<double****>("flux_r", nx, ny, nvx, nvy);
    flux_1st_l              = Kokkos::View<double****>("flux_1st_l", nx, ny, nvx, nvy);
    flux_1st_r              = Kokkos::View<double****>("flux_1st_r", nx, ny, nvx, nvy);
    ep_l                    = Kokkos::View<double****>("ep_l", nx, ny, nvx, nvy);
    ep_r                    = Kokkos::View<double****>("ep_r", nx, ny, nvx, nvy);
    n                       = Kokkos::View<double**>("n", nx, ny);
    rho                     = Kokkos::View<double**>("rho", nx, ny);
    phi                     = Kokkos::View<double**>("phi", nx, ny);
    E                       = Kokkos::View<double***>("E", nx, ny, 2);
    eps                     = Kokkos::View<double**>("eps", nx, ny);
    a                       = Kokkos::View<double**>("a", nx, ny); // jump condition for poisson
    b                       = Kokkos::View<double**>("b", nx, ny); // jump condition for poisson
    Kokkos::deep_copy(f, 0.0);
    Kokkos::deep_copy(flux, 0.0);
    Kokkos::deep_copy(flux_l, 0.0);
    Kokkos::deep_copy(flux_r, 0.0);
    Kokkos::deep_copy(flux_1st_l, 0.0);
    Kokkos::deep_copy(flux_1st_r, 0.0);
    Kokkos::deep_copy(ep_l, 0.0);
    Kokkos::deep_copy(ep_r, 0.0);
    Kokkos::deep_copy(rho, 0.0);
    Kokkos::deep_copy(phi, 0.0);
    Kokkos::deep_copy(E, 0.0);
    Kokkos::deep_copy(eps, 1.0);
    Kokkos::deep_copy(a, 0.0);
    Kokkos::deep_copy(b, 0.0);
}
