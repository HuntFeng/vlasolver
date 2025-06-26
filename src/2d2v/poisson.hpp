#pragma once
#include "world.hpp"
#include <Kokkos_Core.hpp>

/**
 * PoissonSolver class implements the red-black Gauss-Seidel method to solve Poisson's equation
 *
 *  laplacian phi = -rho
 *
 **/
class PoissonSolver {
  private:
    World& world;
    double tol;
    Kokkos::View<double**> phi_old;
    double omega;
    Kokkos::View<double**> a;
    Kokkos::View<double**> b;
    int max_iter = 1e4;
    bool debug   = false;

  public:
    PoissonSolver(World& world, double tol = 1e-6);

    /**
     * Enable debug mode for additional output.
     */
    void enable_debug() { debug = true; }

    /**
     * Apply boundary conditions to the potential field.
     */
    void apply_potential_boundary_conditions();

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
