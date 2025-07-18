/**
 * Poisson solver for the nonlinear Poisson equation
 * Multigrid V-cycle solver with red-black Gauss-Seidel smoothing
 * The solver is modified to match our cell-centered finite difference
 *
 * u'' + f(u) = g
 *
 */
#include "poisson.hpp"
#include "world.hpp"
#include <Kokkos_Core.hpp>

/**
 * Normalized electron charge density, f(phi) = -exp(phi).
 *
 * @param phi: Potential field value.
 */
KOKKOS_INLINE_FUNCTION
double f(double u) {
    double u0 = 0.3; // reference potential value, eV
    double Te = 1.5; // electron temperature, eV
    // double lambda_D = 0.2275;                                       // Debye length, normalized
    return -Kokkos::exp((u - u0) / Te); // normalized electron charge density
    // return 0;
};

PoissonSolver::PoissonSolver(World& world, double tol, int max_iter, int levels)
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

void PoissonSolver::apply_boundary(Kokkos::View<double**>& u) {
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
        Kokkos::deep_copy(Kokkos::subview(u, nx - k - 1, Kokkos::ALL), Kokkos::subview(u, nx - ngc - 2, Kokkos::ALL));
        // bottom boundary, neumann
        Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, k), Kokkos::subview(u, Kokkos::ALL, ngc + 1));
        // top boundary, neumann
        Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, ny - k - 1), Kokkos::subview(u, Kokkos::ALL, ny - ngc - 2));
    }
}

Kokkos::View<double**> PoissonSolver::construct_permittivity(const Kokkos::View<double**>& u) {
    int nx = u.extent(0);
    int ny = u.extent(1);
    Kokkos::View<double**> eps("eps", nx, ny);
    Kokkos::deep_copy(eps, 1.0);
    return eps;
}

Kokkos::View<double**> PoissonSolver::construct_jump_condition_a(const Kokkos::View<double**>& u) {
    int nx = u.extent(0);
    int ny = u.extent(1);
    Kokkos::View<double**> a("a", nx, ny);
    Kokkos::deep_copy(a, 0.0);
    return a;
}

Kokkos::View<double**> PoissonSolver::construct_jump_condition_b(const Kokkos::View<double**>& u) {
    int nx = u.extent(0);
    int ny = u.extent(1);
    Kokkos::View<double**> b("b", nx, ny);
    Kokkos::deep_copy(b, 0.0);
    return b;
}

Kokkos::View<double**> PoissonSolver::nonlinear_operator(const Kokkos::View<double**>& u,
                                                         const Kokkos::View<double**>& eps) {
    auto& grid = world.grid;
    int ngc    = grid.ngc;
    int nx     = u.extent(0);
    int ny     = u.extent(1);
    double dx  = grid.size[0] / (nx - 2 * ngc);
    double dy  = grid.size[1] / (ny - 2 * ngc);
    Kokkos::View<double**> lhs("lhs", nx, ny);
    Kokkos::deep_copy(lhs, 0.0);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            // eps at left, right, top, bottom fluxes
            double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
            double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
            double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
            double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

            // interior indicator
            double x     = (i - ngc + 0.5) * dx;
            double y     = (j - ngc + 0.5) * dy;
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
                eps_l        = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
            }
            if (eta * eta_r <= 0.0) {
                double eta_p = (eta > 0.0) ? eta : eta_r;
                double eta_m = (eta <= 0.0) ? eta : eta_r;
                double eps_p = (eta > 0.0) ? eps(i, j) : eps(i + 1, j);
                double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i + 1, j);
                eps_r        = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
            }
            if (eta * eta_b <= 0.0) {
                double eta_p = (eta > 0.0) ? eta : eta_b;
                double eta_m = (eta <= 0.0) ? eta : eta_b;
                double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j - 1);
                double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j - 1);
                eps_b        = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
            }
            if (eta * eta_t <= 0.0) {
                double eta_p = (eta > 0.0) ? eta : eta_t;
                double eta_m = (eta <= 0.0) ? eta : eta_t;
                double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j + 1);
                double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j + 1);
                eps_t        = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
            }
            lhs(i, j) = (eps_r * (u(i + 1, j) - u(i, j)) - eps_l * (u(i, j) - u(i - 1, j))) / (dx * dx) +
                        (eps_t * (u(i, j + 1) - u(i, j)) - eps_b * (u(i, j) - u(i, j - 1))) / (dy * dy) + f(u(i, j));
        });
    return lhs;
}

void PoissonSolver::gauss_seidel(Kokkos::View<double**>& u,
                                 const Kokkos::View<double**>& g,
                                 const Kokkos::View<double**>& eps,
                                 const Kokkos::View<double**>& a,
                                 const Kokkos::View<double**>& b,
                                 int iters) {

    for (int iter = 0; iter < iters; ++iter) {
        red_black_update(u, g, eps, a, b, 0); // red update
        red_black_update(u, g, eps, a, b, 1); // black update
        apply_boundary(u);                    // apply boundary conditions after each iteration
    }
}

void PoissonSolver::red_black_update(Kokkos::View<double**>& u,
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
                double eta_p  = (eta > 0.0) ? eta : eta_l;
                double eta_m  = (eta <= 0.0) ? eta : eta_l;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
                eps_l         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_l, n2_l] = world.normal(x - dx, y, dx, dy);
                double theta      = abs(eta_l) / (abs(eta) + abs(eta_l));
                double a_gamma    = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));
                double b_gamma = (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
                if (eta <= 0.0)
                    F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
                else
                    F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
            }
            if (eta * eta_r <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_r;
                double eta_m  = (eta <= 0.0) ? eta : eta_r;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i + 1, j);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i + 1, j);
                eps_r         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_r, n2_r] = world.normal(x + dx, y, dx, dy);
                double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
                double a_gamma    = (a(i, j) * abs(eta_r) + a(i + 1, j) * abs(eta)) / (abs(eta) + abs(eta_r));
                double b_gamma = (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
                if (eta <= 0.0)
                    F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
                else
                    F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
            }
            if (eta * eta_b <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_b;
                double eta_m  = (eta <= 0.0) ? eta : eta_b;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i, j - 1);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i, j - 1);
                eps_b         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_b, n2_b] = world.normal(x, y - dy, dx, dy);
                double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
                double a_gamma    = (a(i, j) * abs(eta_b) + a(i, j - 1) * abs(eta)) / (abs(eta) + abs(eta_b));
                double b_gamma = (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
                if (eta <= 0.0)
                    F_b = eps_b * a_gamma / (dy * dy) - eps_b * b_gamma * theta / (eps_p * dy);
                else
                    F_b = -eps_b * a_gamma / (dy * dy) + eps_b * b_gamma * theta / (eps_m * dy);
            }
            if (eta * eta_t <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_t;
                double eta_m  = (eta <= 0.0) ? eta : eta_t;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i, j + 1);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i, j + 1);
                eps_t         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_t, n2_t] = world.normal(x, y + dy, dx, dy);
                double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
                double a_gamma    = (a(i, j) * abs(eta_t) + a(i, j + 1) * abs(eta)) / (abs(eta) + abs(eta_t));
                double b_gamma = (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
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
            u(i, j) = (1 - omega) * u(i, j) + omega * (average - g(i, j) + f(u(i, j)) - Fx - Fy) / denom;
        });
}

Kokkos::View<double**> PoissonSolver::restrict(const Kokkos::View<double**>& u,
                                               const Kokkos::Array<size_t, 2>& n_coarse) {
    int ngc           = world.grid.ngc;
    auto [nx_c, ny_c] = n_coarse;
    Kokkos::View<double**> u_c("u_c", nx_c, ny_c);
    Kokkos::deep_copy(u_c, 0.0);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx_c, ny_c}), KOKKOS_CLASS_LAMBDA(const int i_c, const int j_c) {
            if (i_c < ngc || j_c < ngc || i_c >= nx_c - ngc || j_c >= ny_c - ngc) {
                // u_c(i_c, j_c) = u(i_c, j_c);
            } else {
                int i_f       = 2 * (i_c - ngc) + ngc;
                int j_f       = 2 * (j_c - ngc) + ngc;
                u_c(i_c, j_c) = (u(i_f, j_f) + u(i_f + 1, j_f) + u(i_f, j_f + 1) + u(i_f + 1, j_f + 1)) / 4.0;
            }
        });

    return u_c;
}

Kokkos::View<double**> PoissonSolver::prolong(const Kokkos::View<double**>& e_c,
                                              const Kokkos::Array<size_t, 2>& n_fine) {
    auto [nx_f, ny_f] = n_fine;
    int ngc           = world.grid.ngc;
    Kokkos::View<double**> e_f("e_f", nx_f, ny_f);
    Kokkos::deep_copy(e_f, 0.0);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx_f, ny_f}), KOKKOS_CLASS_LAMBDA(const int i_f, const int j_f) {
            if (i_f < ngc || j_f < ngc || i_f >= nx_f - ngc || j_f >= ny_f - ngc) {
                // e_f(i_f, j_f) = e_c(i_f, j_f);
            } else {
                int i_c       = (i_f - ngc) / 2 + ngc;
                int j_c       = (j_f - ngc) / 2 + ngc;
                e_f(i_f, j_f) = e_c(i_c, j_c);
                // if ((i_f - ngc) % 2 == 0 && (j_f - ngc) % 2 == 0)
                //     e_f(i_f, j_f) = e_c(i_c, j_c);
                // else if ((i_f - ngc) % 2 == 1 && (j_f - ngc) % 2 == 0)
                //     e_f(i_f, j_f) = 0.5 * (e_c(i_c, j_c) + e_c(i_c + 1, j_c));
                // else if ((i_f - ngc) % 2 == 0 && (j_f - ngc) % 2 == 1)
                //     e_f(i_f, j_f) = 0.5 * (e_c(i_c, j_c) + e_c(i_c, j_c + 1));
                // else
                //     e_f(i_f, j_f) =
                //         0.25 * (e_c(i_c, j_c) + e_c(i_c + 1, j_c) + e_c(i_c, j_c + 1) + e_c(i_c + 1, j_c + 1));
            }
        });
    return e_f;
}

void PoissonSolver::v_cycle(Kokkos::View<double**>& u,
                            const Kokkos::View<double**>& g,
                            const Kokkos::View<double**>& eps,
                            const Kokkos::View<double**>& a,
                            const Kokkos::View<double**>& b,
                            int level) {
    gauss_seidel(u, g, eps, a, b, 10);
    if (level == levels - 1) {
        // on coarsest grid, do more smoothing (solving using gauss-seidel)
        gauss_seidel(u, g, eps, a, b, 30);
        return;
    }

    // compute N^h(u^h) = (eps u')' + f(u)
    Kokkos::View<double**> lhs = nonlinear_operator(u, eps);

    // restrict quantities to coarse grid
    int ngc                           = world.grid.ngc;
    Kokkos::Array<size_t, 2> n_fine   = {u.extent(0), u.extent(1)};
    Kokkos::Array<size_t, 2> n_coarse = {(u.extent(0) - 2 * ngc) / 2 + 2 * ngc, (u.extent(1) - 2 * ngc) / 2 + 2 * ngc};
    Kokkos::View<double**> lhs_c      = restrict(lhs, n_coarse);
    Kokkos::View<double**> u_c        = restrict(u, n_coarse);
    Kokkos::View<double**> g_c        = restrict(g, n_coarse);
    // Kokkos::View<double**> eps_c      = restrict(eps, n_coarse);
    // Kokkos::View<double**> a_c        = restrict(a, n_coarse);
    // Kokkos::View<double**> b_c        = restrict(b, n_coarse);
    Kokkos::View<double**> eps_c = construct_permittivity(u_c);
    Kokkos::View<double**> a_c   = construct_jump_condition_a(u_c);
    Kokkos::View<double**> b_c   = construct_jump_condition_b(u_c);

    // FAS (Full approximation Scheme) correction:
    // tau_c = N^H(I^H_h u^h) - I^H_h N^h(u^h)
    // On the coarse grid, the equation is:
    // u_c'' + f(u_c) = g_c + tau_c
    apply_boundary(u_c);
    Kokkos::View<double**> lhs_uc = nonlinear_operator(u_c, eps_c);
    Kokkos::View<double**> g_fas("g_fas", n_coarse[0], n_coarse[1]);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {n_coarse[0], n_coarse[1]}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double tau_c = lhs_c(i, j) - lhs_uc(i, j);
            g_fas(i, j)  = g_c(i, j) + tau_c;
        });

    // save the initial coarse approximation for later computing the correction
    Kokkos::View<double**> u_c_old("u_c_old", n_coarse[0], n_coarse[1]);
    Kokkos::deep_copy(u_c_old, u_c);

    // recursively call v_cycle to coarse grid
    v_cycle(u_c, g_fas, eps_c, a_c, b_c, level + 1);

    // prolongate correction (coarse grid solution minus restricted fine grid approximation)
    Kokkos::View<double**> corr("corr", n_coarse[0], n_coarse[1]);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {n_coarse[0], n_coarse[1]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) { corr(i, j) = u_c(i, j) - u_c_old(i, j); });
    Kokkos::View<double**> corr_fine = prolong(corr, n_fine);

    // correction
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {n_fine[0], n_fine[1]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) { u(i, j) += corr_fine(i, j); });
    apply_boundary(u);

    // post-smoothing
    gauss_seidel(u, g, eps, a, b, 10);
}

double PoissonSolver::compute_error() {
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

void PoissonSolver::solve() {
    auto& rho = world.rho;
    Kokkos::View<double**> g("g", rho.extent(0), rho.extent(1));
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {rho.extent(0), rho.extent(1)}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) { g(i, j) = -rho(i, j); });
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
