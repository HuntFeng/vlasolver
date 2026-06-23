#include "grid.hpp"
#include "poisson_2nd_order.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

/**
 * Example 4.2 in Cho et al. (2019): irregular interface Gamma.
 *
 * Domain: [-1, 1]^2.
 * Interface: phi(x,y) = r - (0.5 + 0.15*sin(5*ang)) centered at (x0,y0).
 *
 * Solution:
 *   u^- = x^2 + y^2                                     if phi < 0
 *   u^+ = 0.1*(x^2+y^2)^2 - 0.01*log(2*sqrt(x^2+y^2))  if phi >= 0
 *
 * Source f = 4 inside, 16(x^2+y^2) outside.
 * Coefficients: beta^- = 1, beta^+ = 10.
 *
 * Jump conditions:
 *   [u]        = u^+ - u^-
 *   [beta u_n] = (4(x^2+y^2) - 0.1/(x^2+y^2) - 2)*(x*n_x + y*n_y)
 */
struct ImmersedWorld : World<ImmersedWorld> {
    static constexpr double x0 = 0.02 * 2.23606797749979; // 0.02 * sqrt(5)
    static constexpr double y0 = 0.02 * 1.73205080756888; // 0.02 * sqrt(3)

    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {

        construct_surface();
        construct_permittivity();
        construct_normal_field();
    }

    // fill the level set field `eta` over the full domain
    void construct_surface() {
        auto& grid              = this->grid;
        auto& eta               = this->eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y]   = grid.center(i, j);
                double dx     = x - x0;
                double dy     = y - y0;
                double rr     = Kokkos::sqrt(dx * dx + dy * dy);
                double ang    = Kokkos::atan2(dy, dx);
                double radius = 0.5 + 0.15 * Kokkos::sin(5.0 * ang);
                eta(i, j)     = rr - radius;
            });
    }

    // fill the region permittivity fields over the full domain (beta^- = 1, beta^+ = 10)
    void construct_permittivity() {
        auto& grid              = this->grid;
        auto& eps_p             = this->eps_p;
        auto& eps_m             = this->eps_m;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                eps_p(i, j) = 10.0; // permittivity in the eta>0 region (beta^+)
                eps_m(i, j) = 1.0;  // permittivity in the eta<0 region (beta^-)
            });
    }

    // fill the Poisson jump condition fields over the full domain
    void poisson_jump_conditions() {
        auto& grid              = this->grid;
        auto& jump_a            = this->jump_a;
        auto& jump_b            = this->jump_b;
        auto& eta               = this->eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        double dx               = grid.spacing(0, 0)[0];
        double dy               = grid.spacing(0, 0)[1];
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y]    = grid.center(i, j);
                double R2      = x * x + y * y;
                double R2_safe = R2 < 1e-30 ? 1e-30 : R2;

                // [u] = u^+ - u^-
                double u_plus  = 0.1 * R2 * R2 - 0.01 * Kokkos::log(2.0 * Kokkos::sqrt(R2_safe));
                double u_minus = R2;
                jump_a(i, j)   = u_plus - u_minus;

                // [beta u_n] = (4 R^2 - 0.1/R^2 - 2)(x n_x + y n_y)
                double factor = 4.0 * R2_safe - 0.1 / R2_safe - 2.0;
                double dx_s =
                    (-eta(i + 2, j) + 8.0 * eta(i + 1, j) - 8.0 * eta(i - 1, j) + eta(i - 2, j)) / (12.0 * dx);
                double dy_s =
                    (-eta(i, j + 2) + 8.0 * eta(i, j + 1) - 8.0 * eta(i, j - 1) + eta(i, j - 2)) / (12.0 * dy);
                double norm = Kokkos::sqrt(dx_s * dx_s + dy_s * dy_s);
                if (norm < 1e-15) {
                    jump_b(i, j) = 0.0;
                } else {
                    double n_x   = dx_s / norm;
                    double n_y   = dy_s / norm;
                    jump_b(i, j) = factor * (x * n_x + y * n_y);
                }
            });
    }

    void potential_boundary_conditions() {
        auto& grid = this->grid;
        int ngc    = grid.ngc;
        int nx     = grid.ncells[0];
        int ny     = grid.ncells[1];
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc) {
                    auto [x, y]          = grid.center(i, j);
                    double R2            = x * x + y * y;
                    double R2_safe       = R2 < 1e-30 ? 1e-30 : R2;
                    double u_plus        = 0.1 * R2 * R2 - 0.01 * Kokkos::log(2.0 * Kokkos::sqrt(R2_safe));
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, u_plus);
                } else {
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
                }
            });
    }
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);

    const int n = (argc == 2) ? std::stoi(argv[1]) : 64;

    // Domain [-1, 1]^2
    Kokkos::Array<double, DIM> origin   = {-1.0, -1.0, 0.0, 0.0};
    Kokkos::Array<double, DIM> size     = {2.0, 2.0, 1.0, 1.0};
    Kokkos::Array<int, DIM> ncells_intr = {n, n, 1, 1};
    const int ngc                       = 3;

    Grid grid(ncells_intr, ngc);
    grid.set_grid(origin, size, 0);
    ImmersedWorld world(grid);
    PoissonSolver2ndOrder poisson_solver(world);

    // Compute exact solution for error reporting
    auto& rho    = world.rho;
    const int nx = grid.ncells[0];
    const int ny = grid.ncells[1];

    Kokkos::View<double**, Kokkos::HostSpace> u_exact_h("u_exact", nx, ny);
    Kokkos::View<double**, Kokkos::HostSpace> dudx_exact_h("dudx_exact", nx, ny);
    Kokkos::View<double**, Kokkos::HostSpace> dudy_exact_h("dudy_exact", nx, ny);

    auto& eta = world.eta;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            auto [x, y] = grid.center(i, j);
            double R2   = x * x + y * y;

            if (eta(i, j) < 0.0) {
                rho(i, j) = -4.0;
            } else {
                rho(i, j) = -16.0 * R2;
            }
        });

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    poisson_solver.compute_electric_field();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);

    // Build exact solution on host for error checking
    auto phi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
    auto E_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
    auto eta_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.eta);

    for (int i = ngc; i < nx - ngc; ++i) {
        for (int j = ngc; j < ny - ngc; ++j) {
            auto [x, y]    = grid.center(i, j);
            double R2      = x * x + y * y;
            double R2_safe = R2 < 1e-30 ? 1e-30 : R2;

            if (eta_h(i, j) < 0.0) {
                u_exact_h(i, j)    = R2;
                dudx_exact_h(i, j) = 2.0 * x;
                dudy_exact_h(i, j) = 2.0 * y;
            } else {
                u_exact_h(i, j)    = 0.1 * R2 * R2 - 0.01 * std::log(2.0 * std::sqrt(R2_safe));
                double coeff       = 0.4 * R2 - 0.01 / R2_safe;
                dudx_exact_h(i, j) = x * coeff;
                dudy_exact_h(i, j) = y * coeff;
            }
        }
    }

    double max_err_u  = 0.0;
    double max_err_du = 0.0;
    for (int i = ngc; i < nx - ngc; ++i) {
        for (int j = ngc; j < ny - ngc; ++j) {
            double err_u  = std::abs(phi_h(i, j) - u_exact_h(i, j));
            double err_du = std::abs(E_h(i, j, 0) + dudx_exact_h(i, j)) + std::abs(E_h(i, j, 1) + dudy_exact_h(i, j));
            if (err_u > max_err_u)
                max_err_u = err_u;
            if (err_du > max_err_du)
                max_err_du = err_du;
        }
    }
    Kokkos::printf("n=%d  max_err_u=%.3e  max_err_du=%.3e\n", n, max_err_u, max_err_du);

    Writer writer(world, "data/poisson", "poisson_" + std::to_string(n), {"phi", "Ex", "Ey"});
    writer.write(0.0);

    return 0;
}
