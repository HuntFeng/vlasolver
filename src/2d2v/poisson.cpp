#include "poisson.hpp"
#include "world.hpp"
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Profiling.hpp>

PoissonSolver::PoissonSolver(World& world, double tol)
    : world(world),
      tol(tol) {
    int nx  = world.grid.ncells[0];
    int ny  = world.grid.ncells[1];
    phi_old = Kokkos::View<double**>("phi", nx, ny);
    omega   = 2 / (1 + Kokkos::sin(Kokkos::numbers::pi / (ny + 1)));
    Kokkos::deep_copy(phi_old, 0.0);
}

void PoissonSolver::red_black_update(int is_update_red) {
    using Kokkos::abs;
    auto& grid = world.grid;
    auto& phi  = world.phi;
    auto& rho  = world.rho;
    auto& eps  = world.eps;
    auto& a    = world.a;
    auto& b    = world.b;

    int nx     = grid.ncells[0];
    int ny     = grid.ncells[1];
    double dx  = grid.spacing[0];
    double dy  = grid.spacing[1];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            if (i < grid.ngc || i >= nx - grid.ngc || j < grid.ngc || j >= ny - grid.ngc)
                return; // skip boundary cells
            if (i % 2 == is_update_red)
                return; // skip red grid points

            auto [x, y, vx, ny] = grid.center({i, j, 0, 0});

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
            double average = (eps_l * phi(i - 1, j) + eps_r * phi(i + 1, j)) / (dx * dx) +
                             (eps_b * phi(i, j - 1) + eps_t * phi(i, j + 1)) / (dy * dy);
            double Fx = F_l + F_r;
            double Fy = F_b + F_t;

            // sor update
            phi(i, j) += omega * (average - rho(i, j) - Fx - Fy - denom * phi(i, j)) / denom;
        });
}

double PoissonSolver::compute_error() {
    using Kokkos::abs;
    auto& phi = world.phi;
    int nx    = world.grid.ncells[0];
    int ny    = world.grid.ncells[1];
    double err;
    Kokkos::Max<double> max_reducer(err);
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double& err) {
            max_reducer.join(err, abs(phi(i, j) - phi_old(i, j)));
        },
        max_reducer);
    return err;
}

void PoissonSolver::solve() {
    int iter = 0;
    Kokkos::deep_copy(phi_old, world.phi);
    red_black_update(0);
    red_black_update(1);
    double err = compute_error();
    iter++;
    while (err > tol && iter < max_iter) {
        Kokkos::deep_copy(phi_old, world.phi);
        red_black_update(0);
        red_black_update(1);
        err = compute_error();
        iter++;
    }
    Kokkos::printf("(PoissonSolver) Iteration = %d, Error(L_inf) = %e\n", iter, err);
};
