#pragma once
#include "grid.hpp"
#include <Kokkos_Core.hpp>

class World {
  public:
    Grid& grid;
    Kokkos::Array<double, 2> q;  // charge number of the particle
    Kokkos::Array<double, 2> mu; // mass ratio of the particle (relative to the electron mass)
    Kokkos::View<double***> f;
    Kokkos::View<double*> rho;
    Kokkos::View<double*> phi;
    Kokkos::View<double*> E;
    Kokkos::View<double*> eps;
    Kokkos::View<double*> phi_jump;
    Kokkos::View<double*> E_jump;

    World(Grid& grid, Kokkos::Array<double, 2> q_in = {-1.0, 1.0}, Kokkos::Array<double, 2> mu_in = {1.0, 1e6});
};
