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

class Vlasolver {
  private:
    World& world;
    PoissonSolver& poisson_solver;
    Writer& writer;

  public:
    Vlasolver(World& world, PoissonSolver& poisson_solver, Writer& writer);

    void initialize_distribution() const;

    void apply_particle_boundary_conditions() const;

    void extrapolate_distribution_function() const;

    void compute_charge_density() const;

    void compute_poisson_jump_conditions() const;

    void compute_electric_field() const;

    KOKKOS_FUNCTION
    double compute_flux(int i,
                        int j,
                        int iv,
                        int jv,
                        int s,
                        int axis,
                        double advection_velocity,
                        const Kokkos::View<double*****>& f) const {
        // third order upwind-biased interpolation
        using Kokkos::max;
        using Kokkos::min;

        int floor_v = (int)Kokkos::floor(advection_velocity);
        auto f_val  = KOKKOS_CLASS_LAMBDA(int offset)->double {
            if (axis == 0) {
                // return f(i + offset, j, iv, jv, s);
                int is = i - floor_v;
                if (is + offset < 0 || is + offset >= f.extent(0))
                    Kokkos::printf("offset = %d, is = %d\n", i, offset, is);
                return f(is + offset, j, iv, jv, s);
            } else if (axis == 1) {
                // return f(i, j + offset, iv, jv, s);
                int js = j - floor_v;
                if (js + offset < 0 || js + offset >= f.extent(1))
                    Kokkos::printf("offset = %d, js = %d\n", i, offset, js);
                return f(i, js + offset, iv, jv, s);
            } else if (axis == 2) {
                // return f(i, j, iv + offset, jv, s);
                int ivs = iv - floor_v;
                int ind = max(min(ivs + offset, (int)f.extent(2) - 1), 0);
                return f(i, j, ind, jv, s);
            } else {
                // return f(i, j, iv, jv + offset, s);
                int jvs = jv - floor_v;
                int ind = max(min(jvs + offset, (int)f.extent(3) - 1), 0);
                return f(i, j, iv, ind, s);
            }
        };

        double fm2    = f_val(-2);
        double fm1    = f_val(-1);
        double f0     = f_val(0);
        double fp1    = f_val(1);
        double fp2    = f_val(2);

        double f_max1 = max(max(fm1, f0), min(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        double f_max2 = max(max(fp1, f0), min(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        double f_min1 = min(min(fm1, f0), max(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        double f_min2 = min(min(fp1, f0), max(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        double f_max  = max(f_max1, f_max2);
        double f_min  = max(0.0, min(f_min1, f_min2));

        // limiter
        double Lp   = (fp1 >= f0) ? min(2.0 * (f0 - f_min), fp1 - f0) : max(2.0 * (f0 - f_max), fp1 - f0);
        double Lm   = (f0 >= fm1) ? min(2.0 * (f_max - f0), f0 - fm1) : max(2.0 * (f_min - f0), f0 - fm1);
        double flux = 0.0;
        if (advection_velocity >= 0.0) {
            // downwind
            double nu = advection_velocity - floor_v;
            flux      = nu * f0 + nu * (1.0 - nu) * (2.0 - nu) * Lp / 6.0 + nu * (1.0 - nu) * (1.0 + nu) * Lm / 6.0;
        } else {
            // upwind
            // flux = nu * f0 + nu * (1.0 + nu) * (2.0 + nu) * Lp / 6.0 + nu * (1.0 - nu) * (1.0 + nu) * Lm / 6.0;
            double nu = advection_velocity - (floor_v + 1);
            flux      = nu * f0 - nu * (1.0 - nu) * (1.0 + nu) * Lp / 6.0 - nu * (1.0 + nu) * (2.0 + nu) * Lm / 6.0;
        }

        if (axis == 0) {
            int is = i - floor_v;
            for (int ix = is + 1; ix <= i; ++ix)
                flux += (advection_velocity >= 0.0) ? f(ix, j, iv, jv, s) : -f(ix, j, iv, jv, s);
        } else if (axis == 1) {
            int js = j - floor_v;
            for (int jy = js + 1; jy <= j; ++jy)
                flux += (advection_velocity >= 0.0) ? f(i, jy, iv, jv, s) : -f(i, jy, iv, jv, s);
        } else if (axis == 2) {
            int ivs = iv - floor_v;
            for (int ivx = max(ivs + 1, 0); ivx <= iv; ++ivx)
                flux += (advection_velocity >= 0.0) ? f(i, j, ivx, jv, s) : -f(i, j, ivx, jv, s);
        } else {
            int jvs = jv - floor_v;
            for (int jvy = max(jvs + 1, 0); jvy <= jv; ++jvy)
                flux += (advection_velocity >= 0.0) ? f(i, j, iv, jvy, s) : -f(i, j, iv, jvy, s);
        }
        return flux;
    }

    void pfc_update_along_space(double dt) const;

    void pfc_update_along_velocity(double dt) const;

    void advance(double dt);

    void solve();
};
