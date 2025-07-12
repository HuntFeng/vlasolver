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
    int max_iter; // max iterations for the solver
    int levels;   // default multigrid levels
    bool debug = false;

  public:
    PoissonSolver(World& world, double tol = 1e-6, int levels = 4, int max_iter = 200);

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
     * @param u: Potential field.
     * @param g: Right-hand side of the Poisson equation.
     * @param eps: Permittivity field.
     * @param a: Jump condition [phi]_Gamma.
     * @param b: Jump condition [d(phi)/dn]_Gamma.
     * @param is_update_red: 1 for red update, 0 for black update.
     */
    void red_black_update(Kokkos::View<double**>& u,
                          const Kokkos::View<double**>& g,
                          const Kokkos::View<double**>& eps,
                          const Kokkos::View<double**>& a,
                          const Kokkos::View<double**>& b,
                          int is_update_red);

    /**
     * Compute infinity norm of the difference between the old and new potential fields.
     */
    double compute_error();

    /**
     * Iteratively solve the potential field until error is less than tolerance or iteration larger than max_iter.
     */
    void solve();

    /**
     * Perform a V-cycle of the multigrid method.
     *
     * @param u: Initial guess for the potential field.
     * @param g: Right-hand side of the Poisson equation.
     * @param eps: Permittivity field.
     * @param a: Jump condition [phi]_Gamma.
     * @param b: Jump condition [d(phi)/dn]_Gamma.
     * @param level: Current multigrid level (0 for finest grid).
     */
    void v_cycle(Kokkos::View<double**>& u,
                 const Kokkos::View<double**>& g,
                 const Kokkos::View<double**>& eps,
                 const Kokkos::View<double**>& a,
                 const Kokkos::View<double**>& b,
                 int level);

    /**
     * Perform red-black Gauss-Seidel iteration to smooth / solve the Poisson equation.
     *
     * @param u: Initial guess for the potential field.
     * @param g: Right-hand side of the Poisson equation.
     * @param eps: Permittivity field.
     * @param a: Jump condition [phi]_Gamma.
     * @param b: Jump condition [d(phi)/dn]_Gamma.
     * @param iters: Number of iterations to perform.
     */
    void gauss_seidel(Kokkos::View<double**>& u,
                      const Kokkos::View<double**>& g,
                      const Kokkos::View<double**>& eps,
                      const Kokkos::View<double**>& a,
                      const Kokkos::View<double**>& b,
                      int iters = 3);

    /**
     * Apply boundary conditions to the potential field.
     *
     * @param u: Potential field.
     * @return: None, modifies u in place.
     */
    void apply_boundary(Kokkos::View<double**>& u);

    /**
     * Compute the nonlinear operator for the Poisson equation. laplacian phi + f(phi).
     *
     * @param u: Potential field.
     * @return: Nonlinear operator applied to the potential field.
     */
    Kokkos::View<double**> nonlinear_operator(const Kokkos::View<double**>& u, const Kokkos::View<double**>& eps);

    /**
     * Restrict a fine grid solution to a coarse grid.
     *
     * @param u: Fine grid solution.
     * @return: Coarse grid solution.
     */
    Kokkos::View<double**> restrict(const Kokkos::View<double**>& u, const Kokkos::Array<size_t, 2>& n_coarse);

    /**
     * Prolongate a coarse grid error to a fine grid.
     *
     * @param ec: Coarse grid error.
     * @param n_fine: Dimensions of the fine grid.
     * @return: Prolongated fine grid error.
     */
    Kokkos::View<double**> prolong(const Kokkos::View<double**>& ec, const Kokkos::Array<size_t, 2>& n_fine);
};
