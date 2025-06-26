#include "vlasov.hpp"

Vlasolver::Vlasolver(World& world, PoissonSolver& poisson_solver, Writer& writer)
    : world(world),
      poisson_solver(poisson_solver),
      writer(writer) {}

void Vlasolver::initialize_distribution() const {
    using Kokkos::cos;
    using Kokkos::exp;
    using Kokkos::pow;
    using Kokkos::sqrt;
    using Kokkos::numbers::pi;

    // must assign grid and f here, otherwise, using world.grid.xxx in device region causes illegal memory access
    auto& grid              = world.grid;
    auto& f                 = world.f;

    auto [nx, ny, nvx, nvy] = grid.ncells;
    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [Lx, Ly, Lvx, Lvy] = grid.size;
    int ngc                 = grid.ngc;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            if (iv < ngc || iv >= nvx - ngc || jv < ngc || jv >= nvy - ngc)
                return;

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
            double eta          = world.surface(x, y);
            // example 4 plasma past charged cylinder from IFE-CSL
            f(i, j, iv, jv, 0) = 0.0;
            f(i, j, iv, jv, 1) = 0.0;
        });
}

void Vlasolver::apply_particle_boundary_conditions() const {
    using Kokkos::exp;
    using Kokkos::pow;
    auto& grid              = world.grid;
    auto& f                 = world.f;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
            auto [vx_min, vy_min, vx_max, vy_max] = grid.velocity_ranges;
            auto [dx, dy, dvx, dvy]               = grid.spacing;
            auto [x, y, vx, vy]                   = grid.center({i, j, iv, jv});
            if (i < ngc) {
                f(i, j, iv, jv, s) = (vx > 0.0) ? 2 * vx * exp(-pow(vx, 2)) : 0.0; // left boundary, injection
                // f(i, j, iv, jv, s) = 0.0; // left boundary, injection
            } else if (i >= nx - ngc) {
                if (vx < 0.0)
                    f(i, j, iv, jv, s) = 0.0; // right boundary, zero-inflow
            } else if (j < ngc) {
                // bottom boundary, reflective
                // find the grid point with reversed normal velocity (vy)
                double reflected_vy = -vy; // Reflect normal velocity component
                int jv_reflected    = ngc + static_cast<int>((reflected_vy - vy_min) / dvy + 0.5);

                // check if the reflected index is valid
                if (jv_reflected >= 0 && jv_reflected < nvy) {
                    f(i, j, iv, jv, s) = f(i, 2 * ngc - j - 1, iv, jv_reflected, s);
                }
            } else if (j >= ny - ngc) {
                // top boundary, reflective
                // find the grid point with reversed normal velocity (vy)
                double reflected_vy = -vy; // Reflect normal velocity component
                int jv_reflected    = ngc + static_cast<int>((reflected_vy - vy_min) / dvy + 0.5);

                // check if the reflected index is valid
                if (jv_reflected >= 0 && jv_reflected < nvy) {
                    f(i, j, iv, jv, s) = f(i, 2 * (ny - ngc) - j - 1, iv, jv_reflected, s);
                }
            }
        });
}

void Vlasolver::extrapolate_distribution_function() const {
    auto& grid              = world.grid;
    auto& f                 = world.f;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;
    double dx               = grid.spacing[0];
    double dy               = grid.spacing[1];

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                return;

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});

            // always extrapolate dist function from the interior of the immersed object
            double eta = world.surface(x, y);
            if (eta >= 0.0)
                return;

            // now (x,y) is the interior of the immersed object
            double eta_l  = world.surface(x - dx, y);
            double eta_r  = world.surface(x + dx, y);
            double eta_b  = world.surface(x, y - dy);
            double eta_t  = world.surface(x, y + dy);
            auto [n1, n2] = world.normal(x, y, dx, dy);
            // extrapolate outflow (v.n < 0), zero-inflow(v.n >= 0)
            int Ng                    = 0;
            double extrapolated_value = 0.0;
            if (eta * eta_l <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i - 1, j, iv, jv, s);
                double f_F2    = f(i - 2, j, iv, jv, s);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_r <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i + 1, j, iv, jv, s);
                double f_F2    = f(i + 2, j, iv, jv, s);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_b <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i, j - 1, iv, jv, s);
                double f_F2    = f(i, j - 2, iv, jv, s);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_t <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i, j + 1, iv, jv, s);
                double f_F2    = f(i, j + 2, iv, jv, s);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }

            if (Ng > 0)
                f(i, j, iv, jv, s) = extrapolated_value / Ng; // average the extrapolated values
        });
}

void Vlasolver::compute_charge_density() const {
    auto& rho  = world.rho;
    auto& q    = world.q;
    auto& f    = world.f;
    auto& n    = world.n;
    auto& grid = world.grid;

    Kokkos::deep_copy(n, 0.0);
    Kokkos::deep_copy(rho, 0.0);
    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0}, {nx, ny, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
            if (i < ngc || i >= nx - ngc)
                return;

            double number_density = 0.0;
            for (int iv = ngc; iv < nvx - ngc; ++iv)
                for (int jv = ngc; jv < nvy - ngc; ++jv)
                    number_density += f(i, j, iv, jv, s) * dvx * dvy;
            n(i, j, s) = number_density;
            rho(i, j) += q[s] * number_density;
        });
}

void Vlasolver::compute_poisson_jump_conditions() const {
    auto& a = world.a;
    auto& b = world.b;
    // auto& rho = world.rho;
    int nx = world.grid.ncells[0];
    int ny = world.grid.ncells[1];

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            a(i, j) = 0.0;
            // b(i, j) = -rho(i, j);
            b(i, j) = 0.0;
        });
}

void Vlasolver::compute_electric_field() const {
    auto& E    = world.E;
    auto& phi  = world.phi;
    auto& grid = world.grid;

    Kokkos::deep_copy(E, 0.0);
    double dx = grid.spacing[0];
    double dy = grid.spacing[1];
    int nx    = grid.ncells[0];
    int ny    = grid.ncells[1];
    int ngc   = grid.ngc;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                return;

            E(i, j, 0) = -(phi(i + 1, j) - phi(i - 1, j)) / (2.0 * dx);
            E(i, j, 1) = -(phi(i, j + 1) - phi(i, j - 1)) / (2.0 * dy);
        });
}

void Vlasolver::pfc_update_along_space(double dt) const {
    auto& f                 = world.f;
    auto& grid              = world.grid;

    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                return;

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
            double eta          = world.surface(x, y);
            if (eta < 0.0)
                return; // skip interior of immersed object
            double flux_in = 0.0, flux_out = 0.0;
            // double nu_x = -vx * dt / dx;
            // flux_out += compute_flux(i, j, iv, jv, s, 0, nu_x, f);
            // flux_in += compute_flux(i - 1, j, iv, jv, s, 0, nu_x, f);
            // double nu_y = -vy * dt / dy;
            // flux_out += compute_flux(i, j, iv, jv, s, 1, nu_y, f);
            // flux_in += compute_flux(i, j - 1, iv, jv, s, 1, nu_y, f);
            double advection_vx = vx * dt / dx;
            double advection_vy = vy * dt / dy;
            flux_in += compute_flux(i - 1, j, iv, jv, s, 0, advection_vx, f);
            flux_in += compute_flux(i, j - 1, iv, jv, s, 1, advection_vy, f);
            flux_out += compute_flux(i, j, iv, jv, s, 0, advection_vx, f);
            flux_out += compute_flux(i, j, iv, jv, s, 1, advection_vy, f);
            f(i, j, iv, jv, s) += flux_in - flux_out;
        });
}

void Vlasolver::pfc_update_along_velocity(double dt) const {
    using Kokkos::max;
    using Kokkos::min;
    auto& f                 = world.f;
    auto& E                 = world.E;
    auto& q                 = world.q;
    auto& mu                = world.mu;
    auto& grid              = world.grid;

    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
            if (iv < ngc || iv >= nvx - ngc || jv < ngc || jv >= nvy - ngc)
                return;
            // third order upwind-biased interpolation
            // double ax           = q[s] / mu[s] * E(i, j, 0); // acceleration
            // double nu_vx        = -ax * dt / dvx;
            // double ay           = q[s] / mu[s] * E(i, j, 1); // acceleration
            // double nu_vy        = -ay * dt / dvy;
            double advection_ax = q[s] / mu[s] * E(i, j, 0) * dt / dvx;
            double advection_ay = q[s] / mu[s] * E(i, j, 1) * dt / dvy;
            if (Kokkos::abs(advection_ax) > 5)
                Kokkos::printf("q[s] = %f mu[s] = %f, E(i, j, 0) = %f, dt = %f, dvx = %f\n", q[s], mu[s], E(i, j, 0),
                               dt, dvx);
            if (Kokkos::abs(advection_ay) > 5)
                Kokkos::printf("q[s] = %f mu[s] = %f, E(i, j, 1) = %f, dt = %f, dvy = %f\n", q[s], mu[s], E(i, j, 1),
                               dt, dvy);

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
            double eta          = world.surface(x, y);
            if (eta < 0.0)
                return; // skip interior of immersed object

            // open boundary condition in the v-direction
            double flux_in = 0.0, flux_out = 0.0;
            // flux_in += compute_flux(i, j, iv - 1, jv, s, 2, nu_vx, f);
            // flux_in += compute_flux(i, j, iv, jv - 1, s, 3, nu_vy, f);
            // flux_out += compute_flux(i, j, iv, jv, s, 2, nu_vx, f);
            // flux_out += compute_flux(i, j, iv, jv, s, 3, nu_vy, f);
            flux_in += compute_flux(i, j, iv - 1, jv, s, 2, advection_ax, f);
            flux_in += compute_flux(i, j, iv, jv - 1, s, 3, advection_ay, f);
            flux_out += compute_flux(i, j, iv, jv, s, 2, advection_ax, f);
            flux_out += compute_flux(i, j, iv, jv, s, 3, advection_ay, f);
            f(i, j, iv, jv, s) += flux_in - flux_out;
        });
}

void Vlasolver::advance(double dt) {
    // perform half time step shift along the x-axis
    pfc_update_along_space(dt / 2.0);
    apply_particle_boundary_conditions();
    // extrapolate the distribution function for the ghost cells
    extrapolate_distribution_function();
    // compute electric field
    compute_charge_density();
    // compute potential field();
    compute_poisson_jump_conditions();
    poisson_solver.solve();
    compute_electric_field();
    // perform shift along the v-axis
    pfc_update_along_velocity(dt);
    apply_particle_boundary_conditions();
    // perform second half time step shift along the x-axis
    pfc_update_along_space(dt / 2.0);
    apply_particle_boundary_conditions();
}

void Vlasolver::solve() {
    Kokkos::printf("Step %zu:\n", 0);
    initialize_distribution();
    apply_particle_boundary_conditions();
    extrapolate_distribution_function();
    compute_charge_density();
    compute_poisson_jump_conditions();
    poisson_solver.solve();
    compute_electric_field();
    writer.write(0);

    for (size_t step = 1; step < world.total_steps; ++step) {
        Kokkos::printf("Step %zu:\n", step);
        advance(world.dt);
        if (step % world.diag_steps == 0)
            writer.write(step * world.dt);
    }
    writer.write(world.total_time);
}
