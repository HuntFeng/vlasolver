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
#include "poisson.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_MinMax.hpp>

class Vlasolver {
  private:
    World& world;
    PoissonSolver& poisson_solver;
    Writer& writer;

  public:
    bool debug = false;

    Vlasolver(World& world, PoissonSolver& poisson_solver, Writer& writer);

    void initialize_distribution() const;

    void apply_particle_boundary_conditions() const;

    void extrapolate_distribution_function() const;

    void compute_charge_density() const;

    void compute_poisson_jump_conditions() const;

    void compute_electric_field() const;

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

        double fm2 = f_val(-2);
        double fm1 = f_val(-1);
        double f0  = f_val(0);
        double fp1 = f_val(1);
        double fp2 = f_val(2);

        // double f_max1 = max(max(fm1, f0), min(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        // double f_max2 = max(max(fp1, f0), min(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        // double f_min1 = min(min(fm1, f0), max(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        // double f_min2 = min(min(fp1, f0), max(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        // double f_max  = max(f_max1, f_max2);
        // double f_min  = max(0.0, min(f_min1, f_min2));

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

        // limiter
        double plus_diff  = fp1 - f0;
        double minus_diff = f0 - fm1;
        double ep_plus =
            (plus_diff >= 0) ? min(1.0, 2.0 * (f0 - f_min) / plus_diff) : min(1.0, 2.0 * (f0 - f_max) / plus_diff);
        double ep_minus =
            (minus_diff >= 0) ? min(1.0, 2.0 * (f_max - f0) / minus_diff) : min(1.0, 2.0 * (f_min - f0) / minus_diff);

        double flux = 0.0;
        if (advection_velocity >= 0.0) {
            // downwind
            double nu = advection_velocity - floor_v;
            flux      = f0;
            flux += ep_plus * (1 - nu) * (2 - nu) * plus_diff / 6.0;
            flux += ep_minus * (1 - nu) * (1 + nu) * minus_diff / 6.0;
            flux *= nu;
            // flux      = nu * f0 + nu * (1.0 - nu) * (2.0 - nu) * Lp / 6.0 + nu * (1.0 - nu) * (1.0 + nu) * Lm / 6.0;
        } else {
            // upwind
            double nu = advection_velocity - (floor_v + 1);
            flux      = f0;
            flux += -ep_plus * (1 - nu) * (1 + nu) * plus_diff / 6.0;
            flux += -ep_minus * (2 + nu) * (1 + nu) * minus_diff / 6.0;
            flux *= nu;
            // flux      = nu * f0 - nu * (1.0 - nu) * (1.0 + nu) * Lp / 6.0 - nu * (1.0 + nu) * (2.0 + nu) * Lm / 6.0;
        }

        if (axis == 0) {
            int is = i - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = is + 1; n <= i; ++n)
                    flux += f(n, j, iv, jv) * dx;
            } else {
                for (int n = i + 1; n <= is - 1; ++n)
                    flux -= f(n, j, iv, jv) * dx;
            }
        } else if (axis == 1) {
            int js = j - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = js + 1; n <= j; ++n)
                    flux += f(i, n, iv, jv) * dy;
            } else {
                for (int n = j + 1; n <= js - 1; ++n)
                    flux -= f(i, n, iv, jv) * dy;
            }
        } else if (axis == 2) {
            int ivs = iv - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = max(ivs + 1, 0); n <= iv; ++n)
                    flux += f(i, j, n, jv) * dvx;
            } else {
                for (int n = iv + 1; n <= min(ivs - 1, f.extent_int(2) - 1); ++n)
                    flux -= f(i, j, n, jv) * dvx;
            }
        } else {
            int jvs = jv - floor_v;
            if (advection_velocity >= 0.0) {
                for (int n = max(jvs + 1, 0); n <= jv; ++n)
                    flux += f(i, j, iv, n) * dvy;
            } else {
                for (int n = jv + 1; n <= min(jvs - 1, f.extent_int(3) - 1); ++n)
                    flux -= f(i, j, iv, n) * dvy;
            }
        }

        if (axis == 0) {
            if (i == 4 && j == 40 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) is = %d\n", i - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
            if (i == 3 && j == 40 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) is = %d\n", i - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
        } else if (axis == 1) {
            if (i == 4 && j == 40 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) is = %d\n", j - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
            if (i == 4 && j == 39 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) is = %d\n", j - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
        } else if (axis == 2) {
            if (i == 3 && j == 40 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) ivs = %d\n", iv - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
            if (i == 3 && j == 40 && iv == 66 && jv == 130) {
                Kokkos::printf("(Compute Flux) ivs = %d\n", iv - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
        } else {
            if (i == 3 && j == 40 && iv == 67 && jv == 130) {
                Kokkos::printf("(Compute Flux) jvs = %d\n", jv - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e, f_max = %e\n", fm1, f0, fp1, f_max);
            }
            if (i == 3 && j == 40 && iv == 67 && jv == 129) {
                Kokkos::printf("(Compute Flux) jvs = %d\n", jv - floor_v);
                Kokkos::printf("(Compute Flux) flux_%d(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, flux);
                Kokkos::printf("(Compute Flux) fm1 = %e, f0 = %e, fp1 = %e,  = %e\n", fm1, f0, fp1, f_max);
            }
        }
        return flux;
    }

    void pfc_update_along_space(double dt) const;

    void pfc_update_along_velocity(double dt) const;

    void advance(double dt);

    void solve();
};
