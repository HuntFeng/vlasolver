#include "vlasov.hpp"
#include <impl/Kokkos_HostThreadTeam.hpp>

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
            // example 4 plasma past charged cylinder from IFE-CSL, no ions and electrons initially
            f(i, j, iv, jv) = 0.0;
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
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            auto [vx_min, vy_min, vx_max, vy_max] = grid.velocity_ranges;
            auto [dx, dy, dvx, dvy]               = grid.spacing;
            auto [x, y, vx, vy]                   = grid.center({i, j, iv, jv});
            if (i < ngc) {
                f(i, j, iv, jv) = (vx > 0.0) ? exp(-pow((vx - 1.4), 2)) / 30 : 0.0; // left boundary, injection
            } else if (i >= nx - ngc) {
                if (vx < 0.0)
                    f(i, j, iv, jv) = 0.0; // right boundary, zero-inflow
            } else if (j < ngc) {
                // bottom boundary, reflective
                // find the grid point with reversed normal velocity (vy)
                double reflected_vy = -vy; // Reflect normal velocity component
                int jv_reflected    = ngc + static_cast<int>((reflected_vy - vy_min) / dvy + 0.5);

                // check if the reflected index is valid
                if (jv_reflected >= 0 && jv_reflected < nvy) {
                    f(i, j, iv, jv) = f(i, 2 * ngc - j - 1, iv, jv_reflected);
                }
            } else if (j >= ny - ngc) {
                // top boundary, reflective
                // find the grid point with reversed normal velocity (vy)
                double reflected_vy = -vy; // Reflect normal velocity component
                int jv_reflected    = ngc + static_cast<int>((reflected_vy - vy_min) / dvy + 0.5);

                // check if the reflected index is valid
                if (jv_reflected >= 0 && jv_reflected < nvy) {
                    f(i, j, iv, jv) = f(i, 2 * (ny - ngc) - j - 1, iv, jv_reflected);
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
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
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
                double f_F1    = f(i - 1, j, iv, jv);
                double f_F2    = f(i - 2, j, iv, jv);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_r <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i + 1, j, iv, jv);
                double f_F2    = f(i + 2, j, iv, jv);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_b <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i, j - 1, iv, jv);
                double f_F2    = f(i, j - 2, iv, jv);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }
            if (eta * eta_t <= 0.0) {
                double v_dot_n = vx * n1 + vy * n2;
                double f_F1    = f(i, j + 1, iv, jv);
                double f_F2    = f(i, j + 2, iv, jv);
                double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                extrapolated_value += 2 * f_I - f_F1;
                Ng++;
            }

            if (Ng > 0)
                // set lower bound to 0 to preserve positivity
                // then divide by Ng to get the average value
                f(i, j, iv, jv) = Kokkos::max(extrapolated_value, 0.0) / Ng;
        });
}

void Vlasolver::compute_charge_density() const {
    auto& rho  = world.rho;
    auto& f    = world.f;
    auto& n    = world.n;
    auto& grid = world.grid;

    Kokkos::deep_copy(n, 0.0);
    Kokkos::deep_copy(rho, 0.0);
    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                return;

            double number_density = 0.0;
            for (int iv = ngc; iv < nvx - ngc; ++iv)
                for (int jv = ngc; jv < nvy - ngc; ++jv)
                    number_density += f(i, j, iv, jv) * dvx * dvy;
            n(i, j)   = number_density;
            rho(i, j) = number_density; // only count ions, electrons follow Boltzmann distribution
        });
}

void Vlasolver::compute_poisson_jump_conditions() const {
    auto& a = world.a;
    auto& b = world.b;
    int nx  = world.grid.ncells[0];
    int ny  = world.grid.ncells[1];

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            a(i, j) = 0.0;
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
    auto& flux              = world.flux;
    auto& grid              = world.grid;

    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc || iv < ngc || iv >= nvx - ngc || jv < ngc ||
                jv >= nvy - ngc)
                return;

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
            double eta          = world.surface(x, y);
            if (eta < 0.0)
                return; // skip interior of immersed object
            double flux_in = 0.0, flux_out = 0.0;
            double advection_vx = vx * dt / dx;
            double advection_vy = vy * dt / dy;

            flux_in += compute_flux({i - 1, j, iv, jv}, grid.spacing, 0, advection_vx, f);
            flux_in += compute_flux({i, j - 1, iv, jv}, grid.spacing, 1, advection_vy, f);
            flux_out += compute_flux({i, j, iv, jv}, grid.spacing, 0, advection_vx, f);
            flux_out += compute_flux({i, j, iv, jv}, grid.spacing, 1, advection_vy, f);
            if ((i == 4 && j == 40 && iv == 67 && jv == 130) || (i == 4 && j == 41 && iv == 67 && jv == 130))
                Kokkos::printf("(PFC space) flux_in = %e, flux_out = %e, f(%d, %d, %d, %d) = %e\n", flux_in, flux_out,
                               i, j, iv, jv, f(i, j, iv, jv));
            flux(i, j, iv, jv) = flux_in - flux_out;
        });

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc || iv < ngc || iv >= nvx - ngc || jv < ngc ||
                jv >= nvy - ngc)
                return;
            f(i, j, iv, jv) += flux(i, j, iv, jv);

            if (f(i, j, iv, jv) < 0.0)
                Kokkos::printf("(PFC space) f(%d, %d, %d, %d) = %e\n", i, j, iv, jv, f(i, j, iv, jv));
        });
}

void Vlasolver::pfc_update_along_velocity(double dt) const {
    using Kokkos::max;
    using Kokkos::min;
    auto& f                 = world.f;
    auto& flux              = world.flux;
    auto& E                 = world.E;
    auto& grid              = world.grid;

    auto [dx, dy, dvx, dvy] = grid.spacing;
    auto [nx, ny, nvx, nvy] = grid.ncells;
    int ngc                 = grid.ngc;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc || iv < ngc || iv >= nvx - ngc || jv < ngc ||
                jv >= nvy - ngc)
                return;
            // third order upwind-biased interpolation
            double advection_ax = E(i, j, 0) * dt / dvx;
            double advection_ay = E(i, j, 1) * dt / dvy;

            auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
            double eta          = world.surface(x, y);
            if (eta < 0.0)
                return; // skip interior of immersed object

            // open boundary condition in the v-direction
            double flux_in = 0.0, flux_out = 0.0;
            flux_in += compute_flux({i, j, iv - 1, jv}, grid.spacing, 2, advection_ax, f);
            flux_in += compute_flux({i, j, iv, jv - 1}, grid.spacing, 3, advection_ay, f);
            flux_out += compute_flux({i, j, iv, jv}, grid.spacing, 2, advection_ax, f);
            flux_out += compute_flux({i, j, iv, jv}, grid.spacing, 3, advection_ay, f);
            if ((i == 3 && j == 40 && iv == 67 && jv == 130) || (i == 3 && j == 41 && iv == 67 && jv == 130))
                Kokkos::printf("(PFC velocity) flux_in = %e, flux_out = %e, f(%d, %d, %d, %d) = %e\n", flux_in,
                               flux_out, i, j, iv, jv, f(i, j, iv, jv));
            flux(i, j, iv, jv) = flux_in - flux_out;
        });

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc || iv < ngc || iv >= nvx - ngc || jv < ngc ||
                jv >= nvy - ngc)
                return;
            f(i, j, iv, jv) += flux(i, j, iv, jv);
            if (f(i, j, iv, jv) < 0.0)
                Kokkos::printf("(PFC velocity) f(%d, %d, %d, %d) = %e\n", i, j, iv, jv, f(i, j, iv, jv));
        });
}

void Vlasolver::advance(double dt) {
    Kokkos::printf("start pfc update along space by dt/2------------------------------------------\n");
    pfc_update_along_space(dt / 2.0);
    Kokkos::fence();
    apply_particle_boundary_conditions();
    Kokkos::fence();
    extrapolate_distribution_function();
    Kokkos::fence();
    compute_charge_density();
    Kokkos::fence();
    compute_poisson_jump_conditions();
    Kokkos::fence();
    poisson_solver.solve();
    Kokkos::fence();
    compute_electric_field();
    Kokkos::fence();
    Kokkos::printf("start pfc update along velocity by dt------------------------------------------\n");
    pfc_update_along_velocity(dt);
    Kokkos::fence();
    apply_particle_boundary_conditions();
    Kokkos::fence();
    Kokkos::printf("start pfc update along space by dt/2------------------------------------------\n");
    pfc_update_along_space(dt / 2.0);
    Kokkos::fence();
    apply_particle_boundary_conditions();
    Kokkos::fence();
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

    debug = true;
    for (world.current_step = 1; world.current_step <= world.total_steps; ++world.current_step) {
        Kokkos::printf("Step %zu:\n", world.current_step);
        advance(world.dt);
        if (world.current_step % world.diag_steps == 0)
            writer.write(world.current_step * world.dt);
    }
    world.current_step--;
    if (world.current_step % world.diag_steps != 0)
        writer.write(world.total_time);
}
