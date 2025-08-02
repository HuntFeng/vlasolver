#pragma once
#include <Kokkos_Core.hpp>

// /**
//  * Normalized electron charge density, f(phi) = -exp(phi).
//  *
//  * @param phi: Potential field value.
//  */
// KOKKOS_INLINE_FUNCTION
// double f(double u) {
//     double u0       = 0.3;                                          // reference potential value, eV
//     double Te       = 1.5;                                          // electron temperature, eV
//     double Ld       = 0.288;                                        // Debye length, m
//     double Lx       = 0.4;                                          // domain length, m
//     double lambda_D = Ld / Lx;                                      // normalized Debye length
//     return -Kokkos::exp((u - u0) / Te) / (2 * lambda_D * lambda_D); // normalized electron charge density
//     // return 0;
// };

/**
 * PoissonSolver class implements the red-black Gauss-Seidel method to solve Poisson's equation
 *
 *  laplacian phi = -rho
 *
 **/
template <typename World>
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
    // PoissonSolver(World& world, double tol = 1e-6, int max_iter = 1e5, int levels = 4);
    PoissonSolver(World& world, double tol = 1e-6, int max_iter = 1e5, int levels = 4)
        : world(world),
          tol(tol),
          levels(levels),
          max_iter(max_iter) {
        int nx  = world.grid.ncells[0];
        int ny  = world.grid.ncells[1];
        phi_old = Kokkos::View<double**>("phi_old", nx, ny);
        // Using omega to control the relaxation rate of Gauss-Seidel iterations
        // For problem with Dirichlet boundaries, optimal omega is given by
        // omega     = 2.0 / (1.0 + (sin(pi / (nx - 2 * ngc))));
        // For mixed boundary conditions, a lower value is needed or it won't converge
        // If using Gauss-Seidel as a smoother for multigrid method, omega needs to be <= 1.0 it does under-relaxation
        omega = 1.9;
    }

    /**
     * Enable debug mode for additional output.
     */
    void enable_debug() { debug = true; }

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
        auto& grid = world.grid;
        int ngc    = grid.ngc;
        int nx     = u.extent(0);
        int ny     = u.extent(1);
        double dx  = grid.size[0] / (nx - 2 * ngc);
        double dy  = grid.size[1] / (ny - 2 * ngc);
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

                // modify eps and F if discontinuity is detected
                if (eta * eta_l <= 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_l;
                    double eta_m = (eta <= 0.0) ? eta : eta_l;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
                    eps_l = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_l, n2_l] = world.normal(x - dx, y, dx, dy);
                    double theta      = abs(eta_l) / (abs(eta) + abs(eta_l));
                    double a_gamma    = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));
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
                    eps_r = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_r, n2_r] = world.normal(x + dx, y, dx, dy);
                    double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
                    double a_gamma    = (a(i, j) * abs(eta_r) + a(i + 1, j) * abs(eta)) / (abs(eta) + abs(eta_r));
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

                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_b, n2_b] = world.normal(x, y - dy, dx, dy);
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
                    eps_t = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_t, n2_t] = world.normal(x, y + dy, dx, dy);
                    double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
                    double a_gamma    = (a(i, j) * abs(eta_t) + a(i, j + 1) * abs(eta)) / (abs(eta) + abs(eta_t));
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
                // u(i, j) = (1 - omega) * u(i, j) + omega * (average - g(i, j) + f(u(i, j)) - Fx - Fy) / denom;
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

        double Ld       = 0.288;   // Debye length, m
        double Lx       = 0.4;     // domain length, m
        double lambda_D = Ld / Lx; // normalized Debye length
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {rho.extent(0), rho.extent(1)}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) { g(i, j) = -rho(i, j) / (2 * lambda_D * lambda_D); });
        apply_boundary(world.phi);
        for (int iter = 0; iter < max_iter; ++iter) {
            Kokkos::deep_copy(phi_old, world.phi);
            // v_cycle(world.phi, g, world.eps, world.a, world.b, 0);
            gauss_seidel(world.phi, g, world.eps, world.a, world.b, 1);
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
                      int iters = 1) {

        for (int iter = 0; iter < iters; ++iter) {
            red_black_update(u, g, eps, a, b, 0); // red update
            red_black_update(u, g, eps, a, b, 1); // black update
            apply_boundary(u);                    // apply boundary conditions after each iteration
        }
    }

    /**
     * Apply boundary conditions to the potential field.
     *
     * @param u: Potential field.
     * @return: None, modifies u in place.
     */
    void apply_boundary(Kokkos::View<double**>& u) {
        using Kokkos::abs;
        auto& grid   = world.grid;
        int ngc      = grid.ngc;
        int nx       = u.extent(0);
        int ny       = u.extent(1);
        double dx    = grid.size[0] / (nx - 2 * ngc);
        double dy    = grid.size[1] / (ny - 2 * ngc);
        double phi_w = -66.67; // normalized potential at the wall of the charged cylinder
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double x   = (i - ngc + 0.5) * dx;
                double y   = (j - ngc + 0.5) * dy;
                double eta = world.surface(x, y);
                if (eta < 0.0) {
                    u(i, j) = phi_w; // inside the immersed object, set potential to a constant value
                }
            });

        for (int k = 0; k < ngc; ++k) {
            // left boundary, dirichlet
            Kokkos::deep_copy(Kokkos::subview(u, k, Kokkos::ALL), 0.0);
            // right boundary, neumann
            // Kokkos::deep_copy(Kokkos::subview(u, nx - k - 1, Kokkos::ALL),
            //                   Kokkos::subview(u, nx - ngc - 2, Kokkos::ALL));
            Kokkos::deep_copy(Kokkos::subview(u, nx - k - 1, Kokkos::ALL),
                              Kokkos::subview(u, nx - 2 * ngc + k, Kokkos::ALL));
            // bottom boundary, neumann
            // Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, k), Kokkos::subview(u, Kokkos::ALL, ngc + 1));
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, k), Kokkos::subview(u, Kokkos::ALL, 2 * ngc - k - 1));
            // top boundary, neumann
            // Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, ny - k - 1),
            //                   Kokkos::subview(u, Kokkos::ALL, ny - ngc - 2));
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, ny - k - 1),
                              Kokkos::subview(u, Kokkos::ALL, ny - 2 * ngc + k));
        }
    }

    /**
     * Construct permittivity field
     */
    Kokkos::View<double**> construct_permittivity(const Kokkos::View<double**>& u);

    /**
     * Construct jump condition [phi]_Gamma
     */
    Kokkos::View<double**> construct_jump_condition_a(const Kokkos::View<double**>& u);

    /**
     * Construct jump condition [eps*dphi/dn]_Gamma
     */
    Kokkos::View<double**> construct_jump_condition_b(const Kokkos::View<double**>& u);

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
