#pragma once
#include "world.hpp"
#include <Kokkos_Core.hpp>

class PoissonSolver {
  private:
    World& world;
    double tol;
    Kokkos::View<double*> phi_old;
    double omega;
    Kokkos::View<double*> a;
    Kokkos::View<double*> b;
    int max_iter = 1e4;

  public:
    PoissonSolver(World& world, double tol = 1e-6);

    /**
     * Update the potential field using the red-black Gauss-Seidel method.
     *
     * @param is_update_red: 1 for red update, 0 for black update.
     */
    void red_black_update(int is_update_red);

    /**
     * Compute infinity norm of the difference between the old and new potential fields.
     */
    double compute_error();

    /**
     * Iteratively solve the potential field until error is less than tolerance or iteration larger than max_iter.
     */
    void solve();
};
