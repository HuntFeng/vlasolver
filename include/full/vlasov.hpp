/* 2d2v vlasov simulation
 *
 * Normalization:
 * x_tilde = x / lamda_D
 * v_tilde = v / v_th
 * t_tilde = t * omega_pe
 * E_tilde = E / (m_e * v_th,e * omega_pe / e)
 * f_tilde = f / (n_0 / v_th,e)
 * The subscript tilde denotes normalized quantities.
 *
 * Normalized Vlasov-Poisson system (subscript n omitted):
 * fe_t + ve fe_x - E fe_v = 0 (electron)
 * fi_t + vi fi_x + E fi_v / mu = 0 (ion)
 * d^2phi/dx^2 = -int (fi - fe) dv
 * where mu = m_i / m_e is the mass ratio of the ion to the electron.
 */
// #include "poisson.hpp"
#include "poisson_2nd_order.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>

template <typename World>
class Vlasolver {
  private:
    World& world;
    PoissonSolver<World>& poisson_solver;
    Writer<World>& writer;

  public:
    Vlasolver(World& world, PoissonSolver<World>& poisson_solver, Writer<World>& writer)
        : world(world),
          poisson_solver(poisson_solver),
          writer(writer) {}

    void extrapolate_distribution_function() const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing[0][0]; // species doesn't matter here
        double dy               = grid.spacing[0][1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                for (int sp = 0; sp < 2; ++sp) {
                    auto [x, y, vx, vy] = grid.center({i, j, iv, jv}, sp);

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
                    if (eta * eta_l < 0.0) {
                        double v_dot_n = vx * n1 + vy * n2;
                        double f_F1    = f(i - 1, j, iv, jv, sp);
                        double f_F2    = f(i - 2, j, iv, jv, sp);
                        double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                        extrapolated_value += 2 * f_I - f_F1;
                        Ng++;
                    }
                    if (eta * eta_r < 0.0) {
                        double v_dot_n = vx * n1 + vy * n2;
                        double f_F1    = f(i + 1, j, iv, jv, sp);
                        double f_F2    = f(i + 2, j, iv, jv, sp);
                        double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                        extrapolated_value += 2 * f_I - f_F1;
                        Ng++;
                    }
                    if (eta * eta_b < 0.0) {
                        double v_dot_n = vx * n1 + vy * n2;
                        double f_F1    = f(i, j - 1, iv, jv, sp);
                        double f_F2    = f(i, j - 2, iv, jv, sp);
                        double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                        extrapolated_value += 2 * f_I - f_F1;
                        Ng++;
                    }
                    if (eta * eta_t < 0.0) {
                        double v_dot_n = vx * n1 + vy * n2;
                        double f_F1    = f(i, j + 1, iv, jv, sp);
                        double f_F2    = f(i, j + 2, iv, jv, sp);
                        double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                        extrapolated_value += 2 * f_I - f_F1;
                        Ng++;
                    }

                    if (Ng > 0)
                        f(i, j, iv, jv, sp) = extrapolated_value / Ng;
                }
            });
    }
    void compute_charge_density() const {
        auto& phi               = world.phi;
        auto& rho               = world.rho;
        auto& f                 = world.f;
        auto& n                 = world.n;
        auto& grid              = world.grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        auto& q                 = world.q;

        Kokkos::deep_copy(n, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                for (int sp = 0; sp < 2; ++sp) {
                    auto [dx, dy, dvx, dvy] = grid.spacing[sp];
                    auto [x, y, vx, vy]     = grid.center({i, j, 0, 0}, sp);
                    if (world.surface(x, y) < 0.0)
                        return;

                    for (int iv = ngc; iv < nvx - ngc; ++iv)
                        for (int jv = ngc; jv < nvy - ngc; ++jv)
                            n(i, j, sp) += f(i, j, iv, jv, sp) * dvx * dvy;

                    rho(i, j) += q[sp] * n(i, j, sp);
                }
            });
    }

    // void compute_electric_field() const {
    //     auto& b    = world.b;
    //     auto& E    = world.E;
    //     auto& phi  = world.phi;
    //     auto& eps  = world.eps;
    //     auto& grid = world.grid;
    //     double dx  = grid.spacing[0][0]; // species does not matter here
    //     double dy  = grid.spacing[0][1];
    //     int nx     = grid.ncells[0];
    //     int ny     = grid.ncells[1];
    //     int ngc    = grid.ngc;
    //
    //     Kokkos::deep_copy(E, 0.0);
    //     Kokkos::parallel_for(
    //         Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //             // for non-boundary cells, compute electric field using central difference
    //             E(i, j, 0) = -(phi(i + 1, j) - phi(i - 1, j)) / (2.0 * dx);
    //             E(i, j, 1) = -(phi(i, j + 1) - phi(i, j - 1)) / (2.0 * dy);
    //
    //             // for boundary cells, compute electric field using jump conditions
    //             auto [x, y, vx, vy] = grid.center({i, j, 0, 0}, 0); // species does not matter here
    //             double eta          = world.surface(x, y);
    //             double eta_l        = world.surface(x - dx, y);
    //             double eta_r        = world.surface(x + dx, y);
    //             double eta_b        = world.surface(x, y - dy);
    //             double eta_t        = world.surface(x, y + dy);
    //             if (eta * eta_l <= 0.0) {
    //                 double eps_c      = eps(i, j);
    //                 double eps_l      = eps(i - 1, j);
    //                 double theta      = abs(eta_l) / (abs(eta) + abs(eta_l));
    //                 auto [n1, n2]     = world.normal(x, y, dx, dy);
    //                 auto [n1_l, n2_l] = world.normal(x - dx, y, dx, dy);
    //                 double b_gamma =
    //                     (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
    //                 double phi_I = eps_c * theta * phi(i, j) + eps_l * (1 - theta) * phi(i - 1, j);
    //                 phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dx;
    //                 phi_I /= eps_c * theta + eps_l * (1 - theta);
    //                 E(i, j, 0) = -(phi(i, j) - phi_I) / ((1 - theta) * dx);
    //             }
    //             if (eta * eta_r <= 0.0) {
    //                 double eps_c      = eps(i, j);
    //                 double eps_r      = eps(i + 1, j);
    //                 auto [n1, n2]     = world.normal(x, y, dx, dy);
    //                 auto [n1_r, n2_r] = world.normal(x + dx, y, dx, dy);
    //                 double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
    //                 double b_gamma =
    //                     (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
    //                 double phi_I = eps_c * theta * phi(i, j) + eps_r * (1 - theta) * phi(i + 1, j);
    //                 phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dx;
    //                 phi_I /= eps_c * theta + eps_r * (1 - theta);
    //                 E(i, j, 0) = -(phi_I - phi(i, j)) / ((1 - theta) * dx);
    //             }
    //             if (eta * eta_b <= 0.0) {
    //                 double eps_c      = eps(i, j);
    //                 double eps_b      = eps(i, j - 1);
    //                 auto [n1, n2]     = world.normal(x, y, dx, dy);
    //                 auto [n1_b, n2_b] = world.normal(x, y - dy, dx, dy);
    //                 double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
    //                 double b_gamma =
    //                     (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
    //                 double phi_I = eps_c * theta * phi(i, j) + eps_b * (1 - theta) * phi(i, j - 1);
    //                 phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dy;
    //                 phi_I /= eps_c * theta + eps_b * (1 - theta);
    //                 E(i, j, 1) = -(phi(i, j) - phi_I) / ((1 - theta) * dy);
    //             }
    //             if (eta * eta_t <= 0.0) {
    //                 double eps_c      = eps(i, j);
    //                 double eps_t      = eps(i, j + 1);
    //                 auto [n1, n2]     = world.normal(x, y, dx, dy);
    //                 auto [n1_t, n2_t] = world.normal(x, y + dy, dx, dy);
    //                 double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
    //                 double b_gamma =
    //                     (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
    //                 double phi_I = eps_c * theta * phi(i, j) + eps_t * (1 - theta) * phi(i, j + 1);
    //                 phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dy;
    //                 phi_I /= eps_c * theta + eps_t * (1 - theta);
    //                 E(i, j, 1) = -(phi_I - phi(i, j)) / ((1 - theta) * dy);
    //             }
    //         });
    // }

    void pfc_update(double dt, int axis, int sp) const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& flux_l            = world.flux_l;
        auto& flux_r            = world.flux_r;
        auto& flux_1st_l        = world.flux_1st_l;
        auto& flux_1st_r        = world.flux_1st_r;
        auto& ep_l              = world.ep_l;
        auto& ep_r              = world.ep_r;
        auto& m                 = world.m;
        auto& q                 = world.q;

        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::deep_copy(flux_l, 0.0);
        Kokkos::deep_copy(flux_r, 0.0);
        Kokkos::deep_copy(flux_1st_l, 0.0);
        Kokkos::deep_copy(flux_1st_r, 0.0);
        Kokkos::deep_copy(ep_l, 1.0);
        Kokkos::deep_copy(ep_r, 1.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc - 1, ngc - 1, ngc - 1, ngc - 1}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy]     = grid.center({i, j, iv, jv}, sp);
                auto [dx, dy, dvx, dvy] = grid.spacing[sp];
                double f0 = 0.0, fp1 = 0.0, fm1 = 0.0;
                double advection_velocity = 0;
                int floor_v               = 0;
                int s                     = 0;
                if (axis == 0) {
                    advection_velocity = vx * dt / dx;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = i - floor_v;
                    f0                 = f(s, j, iv, jv, sp);
                    fp1                = f(s + 1, j, iv, jv, sp);
                    fm1                = f(s - 1, j, iv, jv, sp);
                } else if (axis == 1) {
                    advection_velocity = vy * dt / dy;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = j - floor_v;
                    f0                 = f(i, s, iv, jv, sp);
                    fp1                = f(i, s + 1, iv, jv, sp);
                    fm1                = f(i, s - 1, iv, jv, sp);
                } else if (axis == 2) {
                    advection_velocity = q[sp] / m[sp] * E(i, j, 0) * dt / dvx;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = iv - floor_v;
                    f0                 = f(i, j, s, jv, sp);
                    fp1                = f(i, j, s + 1, jv, sp);
                    fm1                = f(i, j, s - 1, jv, sp);
                } else if (axis == 3) {
                    advection_velocity = q[sp] / m[sp] * E(i, j, 1) * dt / dvy;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = jv - floor_v;
                    f0                 = f(i, j, iv, s, sp);
                    fp1                = f(i, j, iv, s + 1, sp);
                    fm1                = f(i, j, iv, s - 1, sp);
                }

                double plus_diff  = fp1 - f0;
                double minus_diff = f0 - fm1;
                double flux       = 0.0;
                double nu         = 0.0;
                if (advection_velocity >= 0.0) {
                    // downwind
                    nu   = advection_velocity - floor_v;
                    flux = f0;
                    flux += (1 - nu) * (2 - nu) * plus_diff / 6.0;
                    flux += (1 - nu) * (1 + nu) * minus_diff / 6.0;
                    flux *= nu;
                } else {
                    // upwind
                    nu   = advection_velocity - (floor_v + 1);
                    flux = f0;
                    flux += -(1 - nu) * (1 + nu) * plus_diff / 6.0;
                    flux += -(2 + nu) * (1 + nu) * minus_diff / 6.0;
                    flux *= nu;
                }

                if (axis == 0) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= i; ++n)
                            flux += f(n, j, iv, jv, sp);
                    } else {
                        for (int n = i + 1; n <= min(s - 1, f.extent_int(0) - 1); ++n)
                            // flux -= f(n, j, iv, jv, sp);
                            flux += f(n, j, iv, jv, sp);
                    }
                    if (i != nx - ngc - 1) {
                        flux_1st_l(i + 1, j, iv, jv) = nu * f0;
                        flux_l(i + 1, j, iv, jv)     = flux;
                    }
                    if (i != ngc - 1) {
                        flux_1st_r(i, j, iv, jv) = nu * f0;
                        flux_r(i, j, iv, jv)     = flux;
                    }
                } else if (axis == 1) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= j; ++n)
                            flux += f(i, n, iv, jv, sp);
                    } else {
                        for (int n = j + 1; n <= min(s - 1, f.extent_int(1) - 1); ++n)
                            flux += f(i, n, iv, jv, sp);
                    }

                    if (j != ny - ngc - 1) {
                        flux_1st_l(i, j + 1, iv, jv) = nu * f0;
                        flux_l(i, j + 1, iv, jv)     = flux;
                    }
                    if (j != ngc - 1) {
                        flux_1st_r(i, j, iv, jv) = nu * f0;
                        flux_r(i, j, iv, jv)     = flux;
                    }
                } else if (axis == 2) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= iv; ++n)
                            flux += f(i, j, n, jv, sp);
                    } else {
                        for (int n = iv + 1; n <= min(s - 1, f.extent_int(2) - 1); ++n)
                            flux += f(i, j, n, jv, sp);
                    }

                    if (iv != nvx - ngc - 1) {
                        flux_1st_l(i, j, iv + 1, jv) = nu * f0;
                        flux_l(i, j, iv + 1, jv)     = flux;
                    }
                    if (iv != ngc - 1) {
                        flux_1st_r(i, j, iv, jv) = nu * f0;
                        flux_r(i, j, iv, jv)     = flux;
                    }
                } else {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= jv; ++n)
                            flux += f(i, j, iv, n, sp);
                    } else {
                        for (int n = jv + 1; n <= min(s - 1, f.extent_int(3) - 1); ++n)
                            flux += f(i, j, iv, n, sp);
                    }

                    if (jv != nvy - ngc - 1) {
                        flux_1st_l(i, j, iv, jv + 1) = nu * f0;
                        flux_l(i, j, iv, jv + 1)     = flux;
                    }
                    if (jv != ngc - 1) {
                        flux_1st_r(i, j, iv, jv) = nu * f0;
                        flux_r(i, j, iv, jv)     = flux;
                    }
                }
            });

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                double d_r = flux_r(i, j, iv, jv) - flux_1st_r(i, j, iv, jv);
                double d_l = flux_l(i, j, iv, jv) - flux_1st_l(i, j, iv, jv);
                // double delta = -f(i, j, iv, jv) + flux_1st_l(i, j, iv, jv) - flux_1st_r(i, j, iv, jv);
                // should be left - right, I think the paper has a typo
                double delta = -f(i, j, iv, jv, sp) - flux_1st_l(i, j, iv, jv) + flux_1st_r(i, j, iv, jv);
                // if (delta > 0.0) // it should be non-positive since first order monotone fluxes are used
                //     Kokkos::printf("Positive delta(%d, %d, %d, %d) = %e\n", i, j, iv, jv, delta);
                double p = d_l - d_r - delta;
                if (d_l < 0.0 && d_r <= 0.0) {
                    ep_l(i, j, iv, jv) = Kokkos::min(1.0, delta / d_l);
                } else if (d_l * d_r < 0.0 && p < 0.0) {
                    ep_l(i, j, iv, jv) = delta / (d_l - d_r);
                } else {
                    ep_l(i, j, iv, jv) = 1.0;
                }
                if (d_l >= 0.0 && d_r > 0.0) {
                    ep_r(i, j, iv, jv) = Kokkos::min(1.0, -delta / d_r);
                } else if (d_l * d_r < 0.0 && p < 0.0) {
                    ep_r(i, j, iv, jv) = delta / (d_l - d_r);
                } else {
                    ep_r(i, j, iv, jv) = 1.0;
                }
            });

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // modified flux
                double ep_left = 0.0, ep_right = 0.0;
                if (axis == 0) {
                    ep_left  = Kokkos::min(ep_r(i - 1, j, iv, jv), ep_l(i, j, iv, jv));
                    ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i + 1, j, iv, jv));
                } else if (axis == 1) {
                    ep_left  = Kokkos::min(ep_r(i, j - 1, iv, jv), ep_l(i, j, iv, jv));
                    ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j + 1, iv, jv));
                } else if (axis == 2) {
                    ep_left  = Kokkos::min(ep_r(i, j, iv - 1, jv), ep_l(i, j, iv, jv));
                    ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv + 1, jv));
                } else {
                    ep_left  = Kokkos::min(ep_r(i, j, iv, jv - 1), ep_l(i, j, iv, jv));
                    ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv, jv + 1));
                }
                double flux_hat_l =
                    ep_left * (flux_l(i, j, iv, jv) - flux_1st_l(i, j, iv, jv)) + flux_1st_l(i, j, iv, jv);
                double flux_hat_r =
                    ep_right * (flux_r(i, j, iv, jv) - flux_1st_r(i, j, iv, jv)) + flux_1st_r(i, j, iv, jv);

                // udpate distribution function
                if (j == ngc || i == ngc || j == ny - ngc - 1 || i == nx - ngc - 1) {
                    // TODO: 3rd order flux creates oscillation near edges, do 1st flux as work around
                    // can we do 3rd order flux without oscillation?
                    f(i, j, iv, jv, sp) += flux_1st_l(i, j, iv, jv) - flux_1st_r(i, j, iv, jv);
                } else {
                    f(i, j, iv, jv, sp) += flux_hat_l - flux_hat_r;
                }
                // fix any negative value due to numerical error
                if (f(i, j, iv, jv, sp) < 0.0) {
                    f(i, j, iv, jv, sp) = 0.0;
                }
            });
    }
    void advance(double dt) {
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        for (int sp = 0; sp < 2; ++sp) {
            pfc_update(dt / 2.0, 0, sp);
            pfc_update(dt / 2.0, 1, sp);
        }
        Kokkos::printf("(VlasovSolver) Solving electric field\n");
        world.particle_boundary_conditions();
        extrapolate_distribution_function();
        compute_charge_density();
        // world.poisson_jump_conditions();
        poisson_solver.solve();
        poisson_solver.compute_electric_field();
        Kokkos::printf("(VlasovSolver) PFC update along velocity by dt\n");
        for (int sp = 0; sp < 2; ++sp) {
            pfc_update(dt, 2, sp);
            pfc_update(dt, 3, sp);
        }
        world.particle_boundary_conditions();
        extrapolate_distribution_function();
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        for (int sp = 0; sp < 2; ++sp) {
            pfc_update(dt / 2.0, 0, sp);
            pfc_update(dt / 2.0, 1, sp);
        }
        world.particle_boundary_conditions();
        extrapolate_distribution_function();
    }

    void solve() {
        Kokkos::printf("Step %zu:\n", 0);
        world.initialize_distribution();
        world.particle_boundary_conditions();
        extrapolate_distribution_function();
        compute_charge_density();
        // world.poisson_jump_conditions();
        poisson_solver.solve();
        poisson_solver.compute_electric_field();
        writer.write(0);

        for (world.current_step = 1; world.current_step <= world.total_steps; ++world.current_step) {
            Kokkos::printf("Step %zu:\n", world.current_step);
            advance(world.dt);
            if (world.current_step % world.diag_steps == 0)
                writer.write(world.current_step * world.dt);
        }
        world.current_step--;
        if (world.current_step % world.diag_steps != 0)
            writer.write(world.total_time);
    };
};
