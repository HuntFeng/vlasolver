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
 * E_x = int (fi - fe) dv
 * where mu = m_i / m_e is the mass ratio of the ion to the electron.
 */
#include "grid.hpp"
#include "poisson.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>

template <typename World>
class Vlasolver {
  private:
    World& world;
    PoissonSolver<World>& poisson_solver;
    Writer<World>& writer;

  public:
    bool debug = false;

    // Vlasolver(World& world, PoissonSolver<World>& poisson_solver, Writer<World>& writer);
    Vlasolver(World& world, PoissonSolver<World>& poisson_solver, Writer<World>& writer)
        : world(world),
          poisson_solver(poisson_solver),
          writer(writer) {}

    /**
     * Compute flux using third order extrapolation
     */
    KOKKOS_FUNCTION
    double compute_flux(Kokkos::Array<int, DIM> index,
                        Kokkos::Array<double, DIM> spacing,
                        int axis,
                        double advection_velocity,
                        const Kokkos::View<double****>& f) const {
        // third order upwind-biased interpolation
        using Kokkos::max;
        using Kokkos::min;
        auto [i, j, iv, jv]     = index;
        auto [dx, dy, dvx, dvy] = spacing;

        int floor_v             = (int)Kokkos::floor(advection_velocity);
        auto f_val              = KOKKOS_CLASS_LAMBDA(int offset)->double {
            if (axis == 0) {
                int is = i - floor_v;
                return f(is + offset, j, iv, jv);
            } else if (axis == 1) {
                int js = j - floor_v;
                return f(i, js + offset, iv, jv);
            } else if (axis == 2) {
                int ivs = iv - floor_v;
                int ind = max(min(ivs + offset, f.extent_int(2) - 1), 0);
                return f(i, j, ind, jv);
            } else {
                int jvs = jv - floor_v;
                int ind = max(min(jvs + offset, f.extent_int(3) - 1), 0);
                return f(i, j, iv, ind);
            }
        };

        double fm1   = f_val(-1);
        double f0    = f_val(0);
        double fp1   = f_val(1);
        double f_min = 0.0;
        double f_max = 0.0;
        if (axis == 0) {
            for (int n = 0; n < f.extent_int(0); ++n) {
                f_max = Kokkos::max(f_max, f(n, j, iv, jv));
            }
        } else if (axis == 1) {
            for (int n = 0; n < f.extent_int(1); ++n) {
                f_max = Kokkos::max(f_max, f(i, n, iv, jv));
            }
        } else if (axis == 2) {
            for (int n = 0; n < f.extent_int(2); ++n) {
                f_max = Kokkos::max(f_max, f(i, j, n, jv));
            }
        } else {
            for (int n = 0; n < f.extent_int(3); ++n) {
                f_max = Kokkos::max(f_max, f(i, j, iv, n));
            }
        }

        // third order extrapolation
        double plus_diff  = fp1 - f0;
        double minus_diff = f0 - fm1;
        double ep_plus =
            (plus_diff >= 0) ? min(1.0, 2.0 * (f0 - f_min) / plus_diff) : min(1.0, 2.0 * (f0 - f_max) / plus_diff);
        double ep_minus =
            (minus_diff >= 0) ? min(1.0, 2.0 * (f_max - f0) / minus_diff) : min(1.0, 2.0 * (f_min - f0) / minus_diff);

        double flux = 0.0;
        double nu   = 0.0;
        if (advection_velocity >= 0.0) {
            // downwind
            nu   = advection_velocity - floor_v;
            flux = f0;
            flux += ep_plus * (1 - nu) * (2 - nu) * plus_diff / 6.0;
            flux += ep_minus * (1 - nu) * (1 + nu) * minus_diff / 6.0;
            flux *= nu;
        } else {
            // upwind
            nu   = advection_velocity - (floor_v + 1);
            flux = f0;
            flux += -ep_plus * (1 - nu) * (1 + nu) * plus_diff / 6.0;
            flux += -ep_minus * (2 + nu) * (1 + nu) * minus_diff / 6.0;
            flux *= nu;
        }

        if (axis == 0) {
            int is = i - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = is + 1; n <= i; ++n)
                    flux += f(n, j, iv, jv);
            } else {
                for (int n = i + 1; n <= is - 1; ++n)
                    flux -= f(n, j, iv, jv);
            }
        } else if (axis == 1) {
            int js = j - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = js + 1; n <= j; ++n)
                    flux += f(i, n, iv, jv);
            } else {
                for (int n = j + 1; n <= js - 1; ++n)
                    flux -= f(i, n, iv, jv);
            }
        } else if (axis == 2) {
            int ivs = iv - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = max(ivs + 1, 0); n <= iv; ++n)
                    flux += f(i, j, n, jv);
            } else {
                for (int n = iv + 1; n <= min(ivs - 1, f.extent_int(2) - 1); ++n)
                    flux -= f(i, j, n, jv);
            }
        } else {
            int jvs = jv - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = max(jvs + 1, 0); n <= jv; ++n)
                    flux += f(i, j, iv, n);
            } else {
                for (int n = jv + 1; n <= min(jvs - 1, f.extent_int(3) - 1); ++n)
                    flux -= f(i, j, iv, n);
            }
        }
        return flux;
    }

    void initialize_distribution() const {
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

    void apply_particle_boundary_conditions() const {
        using Kokkos::exp;
        using Kokkos::pow;
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [vx_min, vy_min, vx_max, vy_max] = grid.velocity_ranges;
                auto [dx, dy, dvx, dvy]               = grid.spacing;
                auto [x, y, vx, vy]                   = grid.center({i, j, iv, jv});
                if (i < ngc) {
                    f(i, j, iv, jv) = (vx > 0.0) ? exp(-pow((vx - 5), 2)) / 30 : 0.0; // left boundary, injection
                } else if (i >= nx - ngc) {
                    if (vx < 0.0)
                        f(i, j, iv, jv) = 0.0; // right boundary, zero-inflow
                } else if (j < ngc) {
                    f(i, j, iv, jv) = f(i, 2 * ngc - j - 1, iv, nvy - jv - 1); // bottom boundary, reflective
                } else if (j >= ny - ngc) {
                    f(i, j, iv, jv) = f(i, 2 * (ny - ngc) - j - 1, iv, nvy - jv - 1); // top boundary, reflective
                }
            });
    }
    void extrapolate_distribution_function() const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing[0];
        double dy               = grid.spacing[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
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
                if (eta * eta_l < 0.0) {
                    double v_dot_n = vx * n1 + vy * n2;
                    double f_F1    = f(i - 1, j, iv, jv);
                    double f_F2    = f(i - 2, j, iv, jv);
                    double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                    extrapolated_value += 2 * f_I - f_F1;
                    Ng++;
                }
                if (eta * eta_r < 0.0) {
                    double v_dot_n = vx * n1 + vy * n2;
                    double f_F1    = f(i + 1, j, iv, jv);
                    double f_F2    = f(i + 2, j, iv, jv);
                    double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                    extrapolated_value += 2 * f_I - f_F1;
                    Ng++;
                }
                if (eta * eta_b < 0.0) {
                    double v_dot_n = vx * n1 + vy * n2;
                    double f_F1    = f(i, j - 1, iv, jv);
                    double f_F2    = f(i, j - 2, iv, jv);
                    double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                    extrapolated_value += 2 * f_I - f_F1;
                    Ng++;
                }
                if (eta * eta_t < 0.0) {
                    double v_dot_n = vx * n1 + vy * n2;
                    double f_F1    = f(i, j + 1, iv, jv);
                    double f_F2    = f(i, j + 2, iv, jv);
                    double f_I     = (v_dot_n < 0.0) ? 1.5 * f_F1 - 0.5 * f_F2 : 0.0;
                    extrapolated_value += 2 * f_I - f_F1;
                    Ng++;
                }

                if (Ng > 0)
                    f(i, j, iv, jv) = extrapolated_value / Ng;
                // set lower bound to 0 to preserve positivity
                // then divide by Ng to get the average value
                // f(i, j, iv, jv) = Kokkos::max(extrapolated_value, 0.0) / Ng;
            });
    }
    void compute_charge_density() const {
        auto& phi               = world.phi;
        auto& rho               = world.rho;
        auto& f                 = world.f;
        auto& n                 = world.n;
        auto& grid              = world.grid;
        auto [dx, dy, dvx, dvy] = grid.spacing;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::deep_copy(n, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
                if (world.surface(x, y) < 0.0)
                    return;

                double number_density = 0.0;
                for (int iv = ngc; iv < nvx - ngc; ++iv)
                    for (int jv = ngc; jv < nvy - ngc; ++jv)
                        number_density += f(i, j, iv, jv) * dvx * dvy;
                n(i, j) = number_density;
                // rho(i, j) = number_density; // only count ions, electrons follow Boltzmann distribution
                // rho(i, j) = number_density - Kokkos::exp((phi(i, j) - 0.3) / 1.5);
                rho(i, j) = number_density - Kokkos::exp(phi(i, j));
            });
    }
    void compute_poisson_jump_conditions() const {
        auto& grid = world.grid;
        auto& a    = world.a;
        auto& b    = world.b;
        auto& eps  = world.eps;
        int nx     = world.grid.ncells[0];
        int ny     = world.grid.ncells[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                a(i, j)             = 0.0;
                b(i, j)             = 0.0;
                auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
                if (world.surface(x, y) < 0.0) {
                    eps(i, j) = 1000.0;
                } else {
                    eps(i, j) = 1.0;
                }
            });
    }
    void compute_electric_field() const {
        auto& b    = world.b;
        auto& E    = world.E;
        auto& phi  = world.phi;
        auto& eps  = world.eps;
        auto& grid = world.grid;
        double dx  = grid.spacing[0];
        double dy  = grid.spacing[1];
        int nx     = grid.ncells[0];
        int ny     = grid.ncells[1];
        int ngc    = grid.ngc;

        Kokkos::deep_copy(E, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // for non-boundary cells, compute electric field using central difference
                E(i, j, 0) = -(phi(i + 1, j) - phi(i - 1, j)) / (2.0 * dx);
                E(i, j, 1) = -(phi(i, j + 1) - phi(i, j - 1)) / (2.0 * dy);

                // for boundary cells, compute electric field using jump conditions
                auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
                double eta          = world.surface(x, y);
                double eta_l        = world.surface(x - dx, y);
                double eta_r        = world.surface(x + dx, y);
                double eta_b        = world.surface(x, y - dy);
                double eta_t        = world.surface(x, y + dy);
                if (eta * eta_l <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_l      = eps(i - 1, j);
                    double theta      = abs(eta_l) / (abs(eta) + abs(eta_l));
                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_l, n2_l] = world.normal(x - dx, y, dx, dy);
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
                    double phi_I = eps_c * theta * phi(i, j) + eps_l * (1 - theta) * phi(i - 1, j);
                    phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dx;
                    phi_I /= eps_c * theta + eps_l * (1 - theta);
                    E(i, j, 0) = -(phi(i, j) - phi_I) / ((1 - theta) * dx);
                }
                if (eta * eta_r <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_r      = eps(i + 1, j);
                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_r, n2_r] = world.normal(x + dx, y, dx, dy);
                    double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
                    double b_gamma =
                        (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
                    double phi_I = eps_c * theta * phi(i, j) + eps_r * (1 - theta) * phi(i + 1, j);
                    phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dx;
                    phi_I /= eps_c * theta + eps_r * (1 - theta);
                    E(i, j, 0) = -(phi_I - phi(i, j)) / ((1 - theta) * dx);
                }
                if (eta * eta_b <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_b      = eps(i, j - 1);
                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_b, n2_b] = world.normal(x, y - dy, dx, dy);
                    double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
                    double phi_I = eps_c * theta * phi(i, j) + eps_b * (1 - theta) * phi(i, j - 1);
                    phi_I += ((eta <= 0.0) ? 1 : -1) * b_gamma * theta * (1 - theta) * dy;
                    phi_I /= eps_c * theta + eps_b * (1 - theta);
                    E(i, j, 1) = -(phi(i, j) - phi_I) / ((1 - theta) * dy);
                }
                if (eta * eta_t <= 0.0) {
                    double eps_c      = eps(i, j);
                    double eps_t      = eps(i, j + 1);
                    auto [n1, n2]     = world.normal(x, y, dx, dy);
                    auto [n1_t, n2_t] = world.normal(x, y + dy, dx, dy);
                    double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
                    double b_gamma =
                        (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
                    double phi_I = eps_c * theta * phi(i, j) + eps_t * (1 - theta) * phi(i, j + 1);
                    phi_I += ((eta <= 0.0) ? -1 : 1) * b_gamma * theta * (1 - theta) * dy;
                    phi_I /= eps_c * theta + eps_t * (1 - theta);
                    E(i, j, 1) = -(phi_I - phi(i, j)) / ((1 - theta) * dy);
                }
            });
    }
    void pfc_update(double dt, int axis) const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& flux_l            = world.flux_l;
        auto& flux_r            = world.flux_r;
        auto& flux_1st_l        = world.flux_1st_l;
        auto& flux_1st_r        = world.flux_1st_r;
        auto& ep_l              = world.ep_l;
        auto& ep_r              = world.ep_r;

        auto [dx, dy, dvx, dvy] = grid.spacing;
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
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                if (world.surface(x, y) < 0.0)
                    return; // skip interior of immersed object

                double f0 = 0.0, fp1 = 0.0, fm1 = 0.0;
                double advection_velocity = 0;
                int floor_v               = 0;
                int s                     = 0;
                if (axis == 0) {
                    advection_velocity = vx * dt / dx;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = i - floor_v;
                    f0                 = f(s, j, iv, jv);
                    fp1                = f(s + 1, j, iv, jv);
                    fm1                = f(s - 1, j, iv, jv);
                } else if (axis == 1) {
                    advection_velocity = vy * dt / dy;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = j - floor_v;
                    f0                 = f(i, s, iv, jv);
                    fp1                = f(i, s + 1, iv, jv);
                    fm1                = f(i, s - 1, iv, jv);
                } else if (axis == 2) {
                    advection_velocity = E(i, j, 0) * dt / dvx;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = iv - floor_v;
                    f0                 = f(i, j, s, jv);
                    fp1                = f(i, j, s + 1, jv);
                    fm1                = f(i, j, s - 1, jv);
                } else if (axis == 3) {
                    advection_velocity = E(i, j, 1) * dt / dvy;
                    floor_v            = (int)Kokkos::floor(advection_velocity);
                    s                  = jv - floor_v;
                    f0                 = f(i, j, iv, s);
                    fp1                = f(i, j, iv, s + 1);
                    fm1                = f(i, j, iv, s - 1);
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
                            flux += f(n, j, iv, jv);
                    } else {
                        for (int n = i + 1; n <= min(s - 1, f.extent_int(0) - 1); ++n)
                            // flux -= f(n, j, iv, jv);
                            flux += f(n, j, iv, jv);
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
                            flux += f(i, n, iv, jv);
                    } else {
                        for (int n = j + 1; n <= min(s - 1, f.extent_int(1) - 1); ++n)
                            flux += f(i, n, iv, jv);
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
                            flux += f(i, j, n, jv);
                    } else {
                        for (int n = iv + 1; n <= min(s - 1, f.extent_int(2) - 1); ++n)
                            flux += f(i, j, n, jv);
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
                            flux += f(i, j, iv, n);
                    } else {
                        for (int n = jv + 1; n <= min(s - 1, f.extent_int(3) - 1); ++n)
                            flux += f(i, j, iv, n);
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
        auto flux_1st_l_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flux_1st_l);
        auto flux_1st_r_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flux_1st_r);
        // Kokkos::printf("j=18, iv=31, jv=7\n");
        // Kokkos::printf("\nleft\n");
        // for (int i = 0; i <= 4; ++i) {
        //     Kokkos::printf("f_1st_l(%d) = %e", i, flux_1st_l_host(i, 18, 31, 7));
        // }
        // Kokkos::printf("\nright\n");
        // for (int i = 0; i <= 4; ++i) {
        //     Kokkos::printf("f_1st_r(%d) = %e", i, flux_1st_r_host(i, 18, 31, 7));
        // }

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                if (world.surface(x, y) < 0.0)
                    return; // skip interior of immersed object
                double d_r = flux_r(i, j, iv, jv) - flux_1st_r(i, j, iv, jv);
                double d_l = flux_l(i, j, iv, jv) - flux_1st_l(i, j, iv, jv);
                // double delta = -f(i, j, iv, jv) + flux_1st_l(i, j, iv, jv) - flux_1st_r(i, j, iv, jv);
                // should be left - right, I think the paper has a typo
                double delta = -f(i, j, iv, jv) - flux_1st_l(i, j, iv, jv) + flux_1st_r(i, j, iv, jv);
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

                // if (i == 3 && j == 3 && iv == 11 && jv == 5 && axis == 2) {
                //     Kokkos::printf("center(i=%d, j=%d, iv=%d, jv=%d) = [%f,%f,%f,%f]\n d_l=%e, d_r=%e, delta=%e,
                //     p=%e, "
                //                    "ep_l=%e, ep_r=%e\n",
                //                    i, j, iv, jv, x, y, vx, vy, d_l, d_r, delta, p, ep_l(i, j, iv, jv),
                //                    ep_r(i, j, iv, jv));
                // }
                // ep_r = -0.00000 seems a bit strange
                // if (i == 3 && j == 3 && iv == 10 && jv == 5 && axis == 2) {
                //     Kokkos::printf("center(i=%d, j=%d, iv=%d, jv=%d) = [%f,%f,%f,%f]\n d_l=%e, d_r=%e, delta=%e,
                //     p=%e, "
                //                    "ep_l=%e, ep_r=%e\n",
                //                    i, j, iv, jv, x, y, vx, vy, d_l, d_r, delta, p, ep_l(i, j, iv, jv),
                //                    ep_r(i, j, iv, jv));
                // }
            });

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                if (world.surface(x, y) < 0.0)
                    return; // skip interior of immersed object

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

                // if (i == 3 && j == 3 && iv == 11 && jv == 5 && axis == 2) {
                //     Kokkos::printf("flux_hat_l=%e, flux_hat_r=%e\n", flux_hat_l, flux_hat_r);
                //     Kokkos::printf("flux_l=%e, flux_r=%e, flux_1st_l=%e, flux_1st_r=%e\n", flux_l(i, j, iv, jv),
                //                    flux_r(i, j, iv, jv), flux_1st_l(i, j, iv, jv), flux_1st_r(i, j, iv, jv));
                //     Kokkos::printf("ep_left=%e, ep_right=%e\n", ep_left, ep_right);
                //     Kokkos::printf("f(%d, %d, %d, %d) = %e\n", i, j, iv, jv, f(i, j, iv, jv));
                // }
                // udpate distribution function
                f(i, j, iv, jv) += flux_hat_l - flux_hat_r;

                // if (f(i, j, iv, jv) < 0.0 && f(i, j, iv, jv) > -1e-16) {
                //     f(i, j, iv, jv) = 0.0;
                // }
                // if (f(i, j, iv, jv) < -1e-20) {
                if (f(i, j, iv, jv) < 0.0) {
                    f(i, j, iv, jv) = 0.0;
                    // Kokkos::printf("Negative f(%d, %d, %d, %d) = %e\n", i, j, iv, jv, f(i, j, iv, jv));
                }
            });
    }
    void advance(double dt) {
        Kokkos::printf("start pfc update along space by dt/2------------------------------------------\n");
        // pfc_update_along_space(dt / 2.0);
        pfc_update(dt / 2.0, 0);
        pfc_update(dt / 2.0, 1);
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
        // pfc_update_along_velocity(dt);
        pfc_update(dt, 2);
        pfc_update(dt, 3);
        Kokkos::fence();
        apply_particle_boundary_conditions();
        Kokkos::fence();
        extrapolate_distribution_function();
        Kokkos::fence();
        Kokkos::printf("start pfc update along space by dt/2------------------------------------------\n");
        // pfc_update_along_space(dt / 2.0);
        pfc_update(dt / 2.0, 0);
        pfc_update(dt / 2.0, 1);
        Kokkos::fence();
        apply_particle_boundary_conditions();
        Kokkos::fence();
        extrapolate_distribution_function();
    }

    void solve() {
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
    };
};
