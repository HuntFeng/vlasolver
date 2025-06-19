#include "poisson.hpp"
#include "world.hpp"
#include <Kokkos_Core.hpp>

PoissonSolver::PoissonSolver(World& world, double tol)
    : world(world),
      tol(tol) {
    auto [nx, _] = world.grid.extents();
    a            = Kokkos::View<double*>("a", nx);
    b            = Kokkos::View<double*>("b", nx);
    phi_old      = Kokkos::View<double*>("phi", nx);
    omega        = 2 / (1 + Kokkos::sin(Kokkos::numbers::pi / (nx + 1)));
}

// KOKKOS_FUNCTION
// void PoissonSolver::update_potential(int i) const {
//     using Kokkos::abs;
//     auto [dx, dv] = world.grid.spacing();
//     auto [x, v]   = world.grid.center({i, 0});
//
//     // eps at left, right, top, bottom fluxes
//     double eps_l = 0.5 * (world.eps(i - 1) + world.eps(i));
//     double eps_r = 0.5 * (world.eps(i + 1) + world.eps(i));
//
//     // jump condition term at left, right, top, bottom fluxes
//     double F_l = 0.0, F_r = 0.0;
//
//     // interior indicator
//     double eta   = world.surface(x);
//     double eta_l = world.surface(x - dx);
//     double eta_r = world.surface(x + dx);
//
//     // modify world.eps and F if discontinuity is detected
//     if (eta * eta_l <= 0.0) {
//         double eta_p   = (eta > 0.0) ? eta : eta_l;
//         double eta_m   = (eta <= 0.0) ? eta : eta_l;
//         double eps_p   = (eta > 0.0) ? world.eps(i) : world.eps(i - 1);
//         double eps_m   = (eta <= 0.0) ? world.eps(i) : world.eps(i - 1);
//         eps_l          = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
//
//         auto [nx]      = world.normal(x);
//         auto [nx_l]    = world.normal(x - dx);
//         double theta   = abs(eta_l) / (abs(eta) + abs(eta_l));
//         double a_gamma = (a(i) * abs(eta_l) + a(i - 1) * abs(eta)) / (abs(eta) + abs(eta_l));
//         double b_gamma = (b(i) * nx * abs(eta_l) + b(i - 1) * nx_l * abs(eta)) / (abs(eta) + abs(eta_l));
//         if (eta <= 0.0)
//             F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
//         else
//             F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
//     }
//     if (eta * eta_r <= 0.0) {
//         double eta_p   = (eta > 0.0) ? eta : eta_r;
//         double eta_m   = (eta <= 0.0) ? eta : eta_r;
//         double eps_p   = (eta > 0.0) ? world.eps(i) : world.eps(i + 1);
//         double eps_m   = (eta <= 0.0) ? world.eps(i) : world.eps(i + 1);
//         eps_r          = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));
//
//         auto [nx]      = world.normal(x);
//         auto [nx_r]    = world.normal(x + dx);
//         double theta   = abs(eta_r) / (abs(eta) + abs(eta_r));
//         double a_gamma = (a(i) * abs(eta_r) + a(i + 1) * abs(eta)) / (abs(eta) + abs(eta_r));
//         double b_gamma = (b(i) * nx * abs(eta_r) + b(i + 1) * nx_r * abs(eta)) / (abs(eta) + abs(eta_r));
//         if (eta <= 0.0)
//             F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
//         else
//             F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
//     }
//
//     // update potential field
//     double denom   = (eps_l + eps_r) / (dx * dx);
//     double average = (eps_l * world.phi(i - 1) + eps_r * world.phi(i + 1)) / (dx * dx);
//     double Fx      = F_l + F_r;
//
//     // sor update
//     world.phi(i) += omega * (average - world.rho(i) - Fx - denom * world.phi(i)) / denom;
// }

// void PoissonSolver::red_black_update() {
//     auto [nx, _] = world.grid.extents();
//     Kokkos::parallel_for(
//         nx, KOKKOS_CLASS_LAMBDA(int i) {
//             if (i < world.grid.ngc || i >= nx - world.grid.ngc)
//                 return; // skip boundary cells
//             if (i % 2 != 0)
//                 return; // skip red grid points
//             update_potential(i);
//         });
//     // update black grid points using already updated red grid points
//     Kokkos::parallel_for(
//         nx, KOKKOS_CLASS_LAMBDA(int i) {
//             if (i < world.grid.ngc || i >= nx - world.grid.ngc)
//                 return; // skip boundary cells
//             if (i % 2 != 1)
//                 return; // skip red grid points
//             update_potential(i);
//         });
// }

void PoissonSolver::red_black_update(int is_update_red) {
    using Kokkos::abs;
    auto& grid    = world.grid;
    auto& phi     = world.phi;
    auto& rho     = world.rho;
    auto& eps     = world.eps;

    auto [nx, _]  = grid.extents();
    auto [dx, dv] = grid.spacing();
    Kokkos::parallel_for(
        nx, KOKKOS_CLASS_LAMBDA(int i) {
            if (i < grid.ngc || i >= nx - grid.ngc)
                return; // skip boundary cells
            if (i % 2 == is_update_red)
                return; // skip red grid points
                        //

            auto [x, v] = grid.center({i, 0});

            // jump condition term at left, right, top, bottom fluxes
            double F_l = 0.0, F_r = 0.0;

            // eps at left, right, top, bottom fluxes
            double eps_l = 0.5 * (eps(i - 1) + eps(i));
            double eps_r = 0.5 * (eps(i + 1) + eps(i));

            // interior indicator
            double eta   = world.surface(x);
            double eta_l = world.surface(x - dx);
            double eta_r = world.surface(x + dx);

            // modify world.eps and F if discontinuity is detected
            if (eta * eta_l <= 0.0) {
                double eta_p   = (eta > 0.0) ? eta : eta_l;
                double eta_m   = (eta <= 0.0) ? eta : eta_l;
                double eps_p   = (eta > 0.0) ? eps(i) : eps(i - 1);
                double eps_m   = (eta <= 0.0) ? eps(i) : eps(i - 1);
                eps_l          = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [nx]      = world.normal(x, dx);
                auto [nx_l]    = world.normal(x - dx, dx);
                double theta   = abs(eta_l) / (abs(eta) + abs(eta_l));
                double a_gamma = (a(i) * abs(eta_l) + a(i - 1) * abs(eta)) / (abs(eta) + abs(eta_l));
                double b_gamma = (b(i) * nx * abs(eta_l) + b(i - 1) * nx_l * abs(eta)) / (abs(eta) + abs(eta_l));
                if (eta <= 0.0)
                    F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
                else
                    F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
            }
            if (eta * eta_r <= 0.0) {
                double eta_p   = (eta > 0.0) ? eta : eta_r;
                double eta_m   = (eta <= 0.0) ? eta : eta_r;
                double eps_p   = (eta > 0.0) ? eps(i) : eps(i + 1);
                double eps_m   = (eta <= 0.0) ? eps(i) : eps(i + 1);
                eps_r          = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [nx]      = world.normal(x, dx);
                auto [nx_r]    = world.normal(x + dx, dx);
                double theta   = abs(eta_r) / (abs(eta) + abs(eta_r));
                double a_gamma = (a(i) * abs(eta_r) + a(i + 1) * abs(eta)) / (abs(eta) + abs(eta_r));
                double b_gamma = (b(i) * nx * abs(eta_r) + b(i + 1) * nx_r * abs(eta)) / (abs(eta) + abs(eta_r));
                if (eta <= 0.0)
                    F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
                else
                    F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
            }

            // update potential field
            double denom   = (eps_l + eps_r) / (dx * dx);
            double average = (eps_l * phi(i - 1) + eps_r * phi(i + 1)) / (dx * dx);
            double Fx      = F_l + F_r;

            // sor update
            phi(i) += omega * (average - rho(i) - Fx - denom * phi(i)) / denom;
        });
}

double PoissonSolver::compute_error() {
    using Kokkos::abs;
    auto& phi    = world.phi;
    auto [nx, _] = world.grid.extents();
    double err;
    Kokkos::Max<double> max_reducer(err);
    Kokkos::parallel_reduce(
        nx, KOKKOS_CLASS_LAMBDA(int i, double& err) { max_reducer.join(err, abs(phi(i) - phi_old(i))); }, max_reducer);
    return err;
}

void PoissonSolver::solve() {
    int iter = 0;
    Kokkos::deep_copy(phi_old, world.phi);
    red_black_update(0);
    red_black_update(1);
    double err = compute_error();
    iter++;
    // Kokkos::printf("(PoissonSolver) Iteration %d: L2 error = %f\n", iter, err);
    while (err > tol && iter < max_iter) {
        Kokkos::deep_copy(phi_old, world.phi);
        red_black_update(0);
        red_black_update(1);
        err = compute_error();
        iter++;
        // if (iter % 1000 == 0) {
        //     Kokkos::printf("(PoissonSolver) Iteration %d: L2 error = %f\n", iter, err);
        // }
    }
    Kokkos::printf("(PoissonSolver) Iteration %d: L2 error = %f\n", iter, err);
};
