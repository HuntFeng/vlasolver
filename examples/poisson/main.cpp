#include "reduced/grid.hpp"
#include "reduced/poisson_2nd_order.hpp"
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

        int ngc = grid.ngc;
        int nx  = grid.ncells[0];
        int ny  = grid.ncells[1];
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc) {
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0);
                } else {
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        double dx     = x - x0;
        double dy     = y - y0;
        double rr     = Kokkos::sqrt(dx * dx + dy * dy);
        double ang    = Kokkos::atan2(dy, dx);
        double radius = 0.5 + 0.15 * Kokkos::sin(5.0 * ang);
        return rr - radius;
    }

    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return surface(x, y) < 0.0 ? 1.0 : 10.0; }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const {
        double R2      = x * x + y * y;
        double u_plus  = 0.1 * R2 * R2 - 0.01 * Kokkos::log(2.0 * Kokkos::sqrt(R2));
        double u_minus = R2;
        return u_plus - u_minus;
    }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_b(double x, double y) const {
        double R2 = x * x + y * y;
        if (R2 < 1e-30)
            R2 = 1e-30;
        double factor = 4.0 * R2 - 0.1 / R2 - 2.0;

        // Compute normal using the same 4th-order formula as construct_fields
        double dx = grid.spacing[0];
        double dy = grid.spacing[1];
        double dx_s =
            (-surface(x + 2 * dx, y) + 8.0 * surface(x + dx, y) - 8.0 * surface(x - dx, y) + surface(x - 2 * dx, y)) /
            (12.0 * dx);
        double dy_s =
            (-surface(x, y + 2 * dy) + 8.0 * surface(x, y + dy) - 8.0 * surface(x, y - dy) + surface(x, y - 2 * dy)) /
            (12.0 * dy);
        double norm = Kokkos::sqrt(dx_s * dx_s + dy_s * dy_s);
        if (norm < 1e-15)
            return 0.0;
        double nx = dx_s / norm;
        double ny = dy_s / norm;

        return factor * (x * nx + y * ny);
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

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);
    PoissonSolver2ndOrder poisson_solver(world);

    // Compute exact solution for error reporting
    auto& rho    = world.rho;
    const int nx = grid.ncells[0];
    const int ny = grid.ncells[1];

    Kokkos::View<double**, Kokkos::HostSpace> u_exact_h("u_exact", nx, ny);
    Kokkos::View<double**, Kokkos::HostSpace> dudx_exact_h("dudx_exact", nx, ny);
    Kokkos::View<double**, Kokkos::HostSpace> dudy_exact_h("dudy_exact", nx, ny);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            double R2           = x * x + y * y;
            double phi          = world.surface(x, y);
            bool inside         = phi < 0.0;

            if (inside) {
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

    for (int i = ngc; i < nx - ngc; ++i) {
        for (int j = ngc; j < ny - ngc; ++j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            double R2           = x * x + y * y;
            double R2_safe      = R2 < 1e-30 ? 1e-30 : R2;
            double phi          = world.surface(x, y);
            bool inside         = phi < 0.0;

            if (inside) {
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
