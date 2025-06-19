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
     * Update the potential at a given cell.
     * @param i Index of the cell to update.
     */
    // KOKKOS_FUNCTION
    // void update_potential(int i) const;

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
     * Iteratively solve the potential field.
     */
    void solve();

    // KOKKOS_FUNCTION
    // void update_potential(int i) const {
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
};
