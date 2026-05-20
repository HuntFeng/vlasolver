/*
 *
 * This Poisson solver uses the algorithm described in:
 * A Boundary Condition Capturing Method for Poisson’s Equation on Irregular Domains
 * by Xu-Dong Liu 2000, Jornal of Computational Physics, doi:10.1006/jcph.2000.6444
 **/
#pragma once
#include "poisson.hpp"
#include <Kokkos_Core.hpp>

/**
 * PoissonSolver class implements the red-black Gauss-Seidel method to solve Poisson's equation
 *
 *  laplacian phi = -rho
 *
 **/
template <typename World>
class PoissonSolver1stOrder : PoissonSolver<PoissonSolver1stOrder<World>> {
  private:
    World& world;
    double tol;
    Kokkos::View<double**> phi_old;
    double omega;
    Kokkos::View<double**> a;
    Kokkos::View<double**> b;
    Kokkos::View<double**> eps;
    int max_iter; // max iterations for the solver

  public:
    /**
     * Using omega to control the relaxation rate of Gauss-Seidel iterations
     * For problem with Dirichlet boundaries, optimal omega is given by
     * omega     = 2.0 / (1.0 + (sin(pi / (nx - 2 * ngc))));
     * For mixed boundary conditions, a lower value is needed or it won't converge
     * If using Gauss-Seidel as a smoother for multigrid method, omega needs to be <= 1.0 it does under-relaxation
     */
    PoissonSolver1stOrder(World& world, double tol = 1e-6, int max_iter = 1e5, float omega = 1.9)
        : world(world),
          tol(tol),
          max_iter(max_iter),
          omega(omega) {
        int nx  = world.grid.ncells[0];
        int ny  = world.grid.ncells[1];
        phi_old = Kokkos::View<double**>("phi_old", nx, ny);
        construct_fields();
    }

    void construct_fields() {
        auto& grid = world.grid;
        int nx     = world.grid.ncells[0];
        int ny     = world.grid.ncells[1];
        a          = Kokkos::View<double**>("a", nx, ny);
        b          = Kokkos::View<double**>("b", nx, ny);
        eps        = Kokkos::View<double**>("eps", nx, ny);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                a(i, j)     = world.poisson_jump_condition_a(x, y);
                b(i, j)     = world.poisson_jump_condition_b(x, y);
                eps(i, j)   = world.permittivity(x, y);
            });
    }

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
                          int is_update_red) {
        using Kokkos::abs;
        auto& grid    = world.grid;
        auto& normal  = world.normal;
        int ngc       = grid.ngc;
        int nx        = u.extent(0);
        int ny        = u.extent(1);
        auto [dx, dy] = grid.spacing(0, 0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if ((i + j) % 2 != is_update_red)
                    return; // skip red grid points

                double x = (i - ngc + 0.5) * dx;
                double y = (j - ngc + 0.5) * dy;

                // jump condition term at left, right, top, bottom fluxes
                double F_l = 0.0, F_r = 0.0, F_b = 0.0, F_t = 0.0;

                // eps at left, right, top, bottom fluxes
                double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
                double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
                double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
                double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

                // interior indicator
                double eta   = world.surface(x, y);
                double eta_l = world.surface(x - dx, y);
                double eta_r = world.surface(x + dx, y);
                double eta_b = world.surface(x, y - dy);
                double eta_t = world.surface(x, y + dy);

                // normal vector
                double n1 = normal(i, j, 0);
                double n2 = normal(i, j, 1);
                // modify eps and F if discontinuity is detected
                if (eta * eta_l <= 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_l;
                    double eta_m = (eta <= 0.0) ? eta : eta_l;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
                    eps_l       = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    double n1_l = normal(i - 1, j, 0);
                    double n2_l = normal(i - 1, j, 1);
                    double theta   = abs(eta_l) / (abs(eta) + abs(eta_l));
                    double a_gamma = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
                    if (eta <= 0.0)
                        F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
                    else
                        F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
                }
                if (eta * eta_r <= 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_r;
                    double eta_m = (eta <= 0.0) ? eta : eta_r;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i + 1, j);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i + 1, j);
                    eps_r       = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    double n1_r = normal(i + 1, j, 0);
                    double n2_r = normal(i + 1, j, 1);
                    double theta   = abs(eta_r) / (abs(eta) + abs(eta_r));
                    double a_gamma = (a(i, j) * abs(eta_r) + a(i + 1, j) * abs(eta)) / (abs(eta) + abs(eta_r));
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
                    if (eta <= 0.0)
                        F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
                    else
                        F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
                }
                if (eta * eta_b <= 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_b;
                    double eta_m = (eta <= 0.0) ? eta : eta_b;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j - 1);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j - 1);
                    eps_b = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    double n1_b  = normal(i, j - 1, 0);
                    double n2_b  = normal(i, j - 1, 1);
                    double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
                    double a_gamma    = (a(i, j) * abs(eta_b) + a(i, j - 1) * abs(eta)) / (abs(eta) + abs(eta_b));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
                    if (eta <= 0.0)
                        F_b = eps_b * a_gamma / (dy * dy) - eps_b * b_gamma * theta / (eps_p * dy);
                    else
                        F_b = -eps_b * a_gamma / (dy * dy) + eps_b * b_gamma * theta / (eps_m * dy);
                }
                if (eta * eta_t <= 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_t;
                    double eta_m = (eta <= 0.0) ? eta : eta_t;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j + 1);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j + 1);
                    eps_t       = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    double n1_t = normal(i, j + 1, 0);
                    double n2_t = normal(i, j + 1, 1);
                    double theta   = abs(eta_t) / (abs(eta) + abs(eta_t));
                    double a_gamma = (a(i, j) * abs(eta_t) + a(i, j + 1) * abs(eta)) / (abs(eta) + abs(eta_t));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
                    if (eta <= 0.0)
                        F_t = eps_t * a_gamma / (dy * dy) + eps_t * b_gamma * theta / (eps_p * dy);
                    else
                        F_t = -eps_t * a_gamma / (dy * dy) - eps_t * b_gamma * theta / (eps_m * dy);
                }

                // update potential field
                double denom   = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy);
                double average = (eps_l * u(i - 1, j) + eps_r * u(i + 1, j)) / (dx * dx) +
                                 (eps_b * u(i, j - 1) + eps_t * u(i, j + 1)) / (dy * dy);
                double Fx = F_l + F_r;
                double Fy = F_b + F_t;

                // relaxation update
                u(i, j) = (1 - omega) * u(i, j) + omega * (average - g(i, j) - Fx - Fy) / denom;
            });
    }

    void compute_electric_field() const {
        auto& E      = world.E;
        auto& phi    = world.phi;
        auto& normal = world.normal;
        auto& grid   = world.grid;
        double dx    = grid.spacing(0, 0)[0];
        double dy    = grid.spacing(0, 0)[1];
        int nx       = grid.ncells[0];
        int ny       = grid.ncells[1];
        int ngc      = grid.ngc;
        using Kokkos::abs;

        Kokkos::deep_copy(E, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // for non-boundary cells, compute electric field using central difference
                E(i, j, 0) = -(phi(i + 1, j) - phi(i - 1, j)) / (2.0 * dx);
                E(i, j, 1) = -(phi(i, j + 1) - phi(i, j - 1)) / (2.0 * dy);

                // for boundary cells, compute electric field using jump conditions
                auto [x, y]  = grid.center(i, j);
                double eta   = world.surface(x, y);
                double eta_l = world.surface(x - dx, y);
                double eta_r = world.surface(x + dx, y);
                double eta_b = world.surface(x, y - dy);
                double eta_t = world.surface(x, y + dy);

                double n1    = normal(i, j, 0);
                double n2    = normal(i, j, 1);
                if (eta * eta_l <= 0.0) {
                    double eps_c = eps(i, j);
                    double eps_l = eps(i - 1, j);
                    double theta = abs(eta_l) / (abs(eta) + abs(eta_l));
                    double n1_l  = normal(i - 1, j, 0);
                    double n2_l  = normal(i - 1, j, 1);
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
                    double phi_I = eps_c * theta * phi(i, j) + eps_l * (1 - theta) * phi(i - 1, j);
                    phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dx;
                    phi_I /= eps_c * theta + eps_l * (1 - theta);
                    E(i, j, 0) = -(phi(i, j) - phi_I) / ((1 - theta) * dx);
                }
                if (eta * eta_r <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_r      = eps(i + 1, j);
                    double n1_r  = normal(i + 1, j, 0);
                    double n2_r  = normal(i + 1, j, 1);
                    double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
                    double phi_I = eps_c * theta * phi(i, j) + eps_r * (1 - theta) * phi(i + 1, j);
                    phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dx;
                    phi_I /= eps_c * theta + eps_r * (1 - theta);
                    E(i, j, 0) = -(phi_I - phi(i, j)) / ((1 - theta) * dx);
                }
                if (eta * eta_b <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_b      = eps(i, j - 1);
                    double n1_b  = normal(i, j - 1, 0);
                    double n2_b  = normal(i, j - 1, 1);
                    double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
                    double phi_I = eps_c * theta * phi(i, j) + eps_b * (1 - theta) * phi(i, j - 1);
                    phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dy;
                    phi_I /= eps_c * theta + eps_b * (1 - theta);
                    E(i, j, 1) = -(phi(i, j) - phi_I) / ((1 - theta) * dy);
                }
                if (eta * eta_t <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_t      = eps(i, j + 1);
                    double n1_t  = normal(i, j + 1, 0);
                    double n2_t  = normal(i, j + 1, 1);
                    double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
                    double phi_I = eps_c * theta * phi(i, j) + eps_t * (1 - theta) * phi(i, j + 1);
                    phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dy;
                    phi_I /= eps_c * theta + eps_t * (1 - theta);
                    E(i, j, 1) = -(phi_I - phi(i, j)) / ((1 - theta) * dy);
                }
            });
    }

    /**
     * Compute infinity norm of the difference between the old and new potential fields.
     */
    double compute_error() {
        using Kokkos::abs;
        auto& phi = world.phi;
        int nx    = world.grid.ncells[0];
        int ny    = world.grid.ncells[1];
        int ngc   = world.grid.ngc;
        double err;
        Kokkos::Max<double> max_reducer(err);
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, double& err) {
                max_reducer.join(err, abs(phi(i, j) - phi_old(i, j)));
            },
            max_reducer);
        return err;
    }

    /**
     * Iteratively solve the potential field until error is less than tolerance or iteration larger than max_iter.
     */
    void solve() {
        auto& rho = world.rho;
        Kokkos::View<double**> g("g", rho.extent(0), rho.extent(1));

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {rho.extent(0), rho.extent(1)}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) { g(i, j) = -rho(i, j); });
        world.potential_boundary_conditions();
        for (int iter = 0; iter < max_iter; ++iter) {
            Kokkos::deep_copy(phi_old, world.phi);
            gauss_seidel(world.phi, g, eps, a, b, 1);
            double err = compute_error();
            if (iter % 1000 == 0 || iter == max_iter - 1) {
                Kokkos::printf("(PoissonSolver) Iteration = %d, Error(L_inf) = %e\n", iter, err);
            }
            if (err < tol) {
                Kokkos::printf("(PoissonSolver) Converged in %d iterations with error %e\n", iter, err);
                return;
            }
        }
    }

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
                      int iters = 1) {

        for (int iter = 0; iter < iters; ++iter) {
            red_black_update(u, g, eps, a, b, 0); // red update
            red_black_update(u, g, eps, a, b, 1); // black update
            // apply_boundary(u);                    // apply boundary conditions after each iteration
            world.potential_boundary_conditions();
        }
    }
};
