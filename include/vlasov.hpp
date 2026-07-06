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
 *
 * This solver is templated on the World type, which carries a compile-time
 * species count (World::nspecies) and electron closure (World::electron_model).
 * - Kinetic model:   all species evolved kinetically; rho = sum q[sp] n[sp].
 * - Boltzmann model: only ions evolved kinetically; rho = sum q[sp] n[sp] - exp(phi).
 */
#include "matrix/solve.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_MinMax.hpp>

template <typename World, typename PoissonSolver>
class Vlasolver {
  private:
    World& world;
    PoissonSolver& poisson_solver;
    Writer<World>& writer;

    static constexpr int N              = World::nspecies;
    static constexpr ElectronModel EM   = World::electron_model;

  public:
    Vlasolver(World& world, PoissonSolver& poisson_solver, Writer<World>& writer)
        : world(world),
          poisson_solver(poisson_solver),
          writer(writer) {}

    void extrapolate_distribution_1st_order() const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& normal            = world.normal;
        auto& eta_field         = world.eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing(0, 0)[0]; // species doesn't matter here
        double dy               = grid.spacing(0, 0)[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                for (int sp = 0; sp < N; ++sp) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);

                    // always extrapolate dist function from the interior of the immersed object
                    double eta = eta_field(i, j);
                    if (eta >= 0.0)
                        return;

                    // now (x,y) is the interior of the immersed object
                    double eta_l = eta_field(i - 1, j);
                    double eta_r = eta_field(i + 1, j);
                    double eta_b = eta_field(i, j - 1);
                    double eta_t = eta_field(i, j + 1);
                    double n1    = normal(i, j, 0);
                    double n2    = normal(i, j, 1);
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
                        f(i, j, iv, jv, sp) = Kokkos::max(0.0, extrapolated_value / Ng);
                }
            });
    }

    void extrapolate_distribution_2nd_order() const {
        using Kokkos::abs;
        using Kokkos::sqrt;
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& normal            = world.normal;
        auto& eta_field         = world.eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing(0, 0)[0]; // species doesn't matter here
        double dy               = grid.spacing(0, 0)[1];
        // search region range from -offset_range to +offset_range
        const int offset_range = 5;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                for (int sp = 0; sp < N; ++sp) {
                    auto [x0, y0, vx, vy] = grid.center(i, j, iv, jv, sp);

                    // always extrapolate dist function from the interior of the immersed object
                    double eta = eta_field(i, j);
                    if (eta >= 0.0)
                        return;

                    // now (x0,y0) is the interior of the immersed object
                    double eta_l   = eta_field(i - 1, j);
                    double eta_r   = eta_field(i + 1, j);
                    double eta_b   = eta_field(i, j - 1);
                    double eta_t   = eta_field(i, j + 1);
                    double n1      = normal(i, j, 0);
                    double n2      = normal(i, j, 1);
                    double v_dot_n = vx * n1 + vy * n2;

                    if (eta * eta_l < 0.0 || eta * eta_r < 0.0 || eta * eta_b < 0.0 || eta * eta_t < 0.0) {
                        Kokkos::Array<Kokkos::Array<double, 3>, 3> A{}; // use list init to initialize elements to 0
                        Kokkos::Array<double, 3> b{};

                        // We assume f near the boundary can be approximated by a plane: f(x,y) = c0 * x + c1 * y + c2
                        // solve least square problem A @ c = b (where c = [c0, c1, c2])
                        // the normal equation is A^T @ A @ c = A^T @ b
                        // sum x_i^2 * c0 + sum x_i*y_i * c1 + sum x_i * c2 = sum x_i*f_i
                        // sum x_i*y_i * c0 + sum y_i^2 * c1 + sum y_i * c2 = sum y_i*f_i
                        // sum x_i * c0 + sum y_i * c1 + sum 1 * c2 = sum f_i
                        for (int x_offset = -offset_range; x_offset <= offset_range; ++x_offset) {
                            for (int y_offset = -offset_range; y_offset <= offset_range; ++y_offset) {
                                int I               = i + x_offset;
                                int J               = j + y_offset;
                                auto [x, y, vx, vy] = grid.center(I, J, iv, jv, sp);
                                // skip ghost cells
                                if (I < ngc || I > nx - ngc - 1 || J < ngc || J > ny - ngc - 1 || eta_field(I, J) < 0)
                                    continue;

                                // only extrapolate using fluid cells
                                double z = f(I, J, iv, jv, sp);
                                A[0][0] += x * x;
                                A[1][1] += y * y;
                                A[2][2] += 1.0;
                                A[1][0] += x * y;
                                A[0][1] += x * y;
                                A[2][0] += x;
                                A[0][2] += x;
                                A[1][2] += y;
                                A[2][1] += y;

                                b[0] += x * z;
                                b[1] += y * z;
                                b[2] += z;
                            }
                        }
                        // For inflow, we force zero-inflow condition. i.e. f(x_b, y_b) = 0,
                        // where (x_b, y_b) is the intersection point between the normal line and the boundary
                        // now the last row of normal equation becomes
                        // sum x_i^2 * c0 + sum x_i*y_i * c1 + 0 = sum x_i*f_i
                        // sum x_i*y_i * c0 + sum y_i^2 * c1 + 0 = sum y_i*f_i
                        // b[0] * x_b + b[1] * y_b + b[2] = 0
                        if (v_dot_n >= 0.0) {
                            double x_b = x0 - eta * n1;
                            double y_b = y0 - eta * n2;
                            A[0][2]    = 0.0;
                            A[1][2]    = 0.0;
                            A[2][0]    = x_b;
                            A[2][1]    = y_b;
                            A[2][2]    = 1.0;
                            b[2]       = 0.0;
                        }
                        solve_linear_system(A, b);

                        // extrapolate using best-fit plane
                        f(i, j, iv, jv, sp) = b[0] * x0 + b[1] * y0 + b[2];
                    }
                }
            });
    }

    void compute_charge_density() const {
        auto& phi               = world.phi;
        auto& rho               = world.rho;
        auto& f                 = world.f;
        auto& n                 = world.n;
        auto& grid              = world.grid;
        auto& eta_field         = world.eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        auto& q                 = world.q;

        Kokkos::deep_copy(n, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        // range must include ghost cells to resolve periodic boundary correctly
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (eta_field(i, j) < 0.0)
                    return;

                for (int sp = 0; sp < N; ++sp) {
                    auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                    double number_density   = 0.0;
                    for (int iv = ngc; iv < nvx - ngc; ++iv)
                        for (int jv = ngc; jv < nvy - ngc; ++jv)
                            number_density += f(i, j, iv, jv, sp) * dvx * dvy;

                    n(i, j, sp) = number_density;
                    rho(i, j) += q[sp] * number_density;
                }

                // Boltzmann electrons: n_e = exp(phi), contributing -exp(phi) to the charge density
                // NOTE: plain `if` (not `if constexpr`): nvcc forbids first-capturing a
                // variable in a constexpr-if context inside an extended device lambda.
                // EM is a compile-time constant, so the dead branch is pruned regardless.
                if (EM == ElectronModel::Boltzmann)
                    rho(i, j) -= Kokkos::exp(phi(i, j));
            });
    }

    void pfc_update(double dt, int axis, int sp) const {
        using Kokkos::abs;
        using Kokkos::floor;
        using Kokkos::max;
        using Kokkos::min;
        using Kokkos::trunc;

        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& eta_field         = world.eta;
        auto& flux              = world.flux;
        auto& flux_1st          = world.flux_1st;
        auto& ep_l              = world.ep_l;
        auto& ep_r              = world.ep_r;
        auto& m                 = world.m;
        auto& q                 = world.q;

        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::deep_copy(flux, 0.0);
        Kokkos::deep_copy(flux_1st, 0.0);
        Kokkos::deep_copy(ep_l, 1.0);
        Kokkos::deep_copy(ep_r, 1.0);

        // Step 1: Compute fluxes at cell interfaces
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc - 1, ngc - 1, ngc - 1, ngc - 1}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy]       = grid.center(i, j, iv, jv, sp);
                auto [dx, dy, dvx, dvy]   = grid.spacing(sp);
                double advection_velocity = (axis == 0)   ? vx * dt / dx
                                            : (axis == 1) ? vy * dt / dy
                                            : (axis == 2) ? q[sp] / m[sp] * E(i, j, 0) * dt / dvx
                                                          : q[sp] / m[sp] * E(i, j, 1) * dt / dvy;
                double floor_velocity     = floor(advection_velocity);
                int axis_idx              = (axis == 0) ? i : (axis == 1) ? j : (axis == 2) ? iv : jv;
                int s                     = axis_idx - (int)floor_velocity;
                // The departure cell s can fall outside the grid when the CFL number
                // |advection_velocity| exceeds the available cells (large E in velocity space).
                // Outside the array f = 0 (vacuum at the velocity boundary), so guard the reads.
                int ext           = f.extent_int(axis);
                double f0         = (s >= 0 && s < ext) ? ((axis == 0)   ? f(s, j, iv, jv, sp)
                                                           : (axis == 1) ? f(i, s, iv, jv, sp)
                                                           : (axis == 2) ? f(i, j, s, jv, sp)
                                                                         : f(i, j, iv, s, sp))
                                                        : 0.0;
                double fp1        = (s + 1 >= 0 && s + 1 < ext) ? ((axis == 0)   ? f(s + 1, j, iv, jv, sp)
                                                                   : (axis == 1) ? f(i, s + 1, iv, jv, sp)
                                                                   : (axis == 2) ? f(i, j, s + 1, jv, sp)
                                                                                 : f(i, j, iv, s + 1, sp))
                                                                : 0.0;
                double fm1        = (s - 1 >= 0 && s - 1 < ext) ? ((axis == 0)   ? f(s - 1, j, iv, jv, sp)
                                                                   : (axis == 1) ? f(i, s - 1, iv, jv, sp)
                                                                   : (axis == 2) ? f(i, j, s - 1, jv, sp)
                                                                                 : f(i, j, iv, s - 1, sp))
                                                                : 0.0;

                double plus_diff  = fp1 - f0;
                double minus_diff = f0 - fm1;
                double nu         = advection_velocity - trunc(advection_velocity);
                // high order flux correction (Liu/Xiong PFC)
                double diff_0          = (advection_velocity >= 0.0) ? plus_diff : minus_diff;
                double diff_1          = (advection_velocity < 0.0) ? plus_diff : minus_diff;
                double flux_correction = 0.0;
                flux_correction += abs(nu) * (1 - abs(nu)) * (2 - abs(nu)) * diff_0 / 6.0;
                flux_correction += abs(nu) * (1 - abs(nu)) * (1 + abs(nu)) * diff_1 / 6.0;

                // integer grid-shift contribution to the flux
                double flux_shift = 0.0;
                if (advection_velocity >= 0.0) {
                    int n_start = max(s + 1, 0);
                    int n_end   = (axis == 0) ? i : (axis == 1) ? j : (axis == 2) ? iv : jv;
                    for (int n = n_start; n <= n_end; ++n)
                        // prevent using negative dist at ghost cells
                        flux_shift += (axis == 0)   ? max(f(n, j, iv, jv, sp), 0.0)
                                      : (axis == 1) ? max(f(i, n, iv, jv, sp), 0.0)
                                      : (axis == 2) ? max(f(i, j, n, jv, sp), 0.0)
                                                    : max(f(i, j, iv, n, sp), 0.0);
                } else {
                    int n_start = (axis == 0) ? i + 1 : (axis == 1) ? j + 1 : (axis == 2) ? iv + 1 : jv + 1;
                    int n_end   = min(s - 1, f.extent_int(axis) - 1);
                    for (int n = n_start; n <= n_end; ++n)
                        flux_shift -= (axis == 0)   ? max(f(n, j, iv, jv, sp), 0.0)
                                      : (axis == 1) ? max(f(i, n, iv, jv, sp), 0.0)
                                      : (axis == 2) ? max(f(i, j, n, jv, sp), 0.0)
                                                    : max(f(i, j, iv, n, sp), 0.0);
                }

                // boundary-aware 1st order flux, it should be monotone
                flux_1st(i, j, iv, jv) = nu * max(f0, 0.0) + flux_shift;
                flux(i, j, iv, jv)     = flux_1st(i, j, iv, jv) + flux_correction;
            });

        // Step 2: Compute flux limiters
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // flux at left interface is flux(i-1,...) along the active axis, right interface is flux(i,...)
                double flux_left      = (axis == 0)   ? flux(i - 1, j, iv, jv)
                                        : (axis == 1) ? flux(i, j - 1, iv, jv)
                                        : (axis == 2) ? flux(i, j, iv - 1, jv)
                                                      : flux(i, j, iv, jv - 1);
                double flux_right     = flux(i, j, iv, jv);
                double flux_1st_left  = (axis == 0)   ? flux_1st(i - 1, j, iv, jv)
                                        : (axis == 1) ? flux_1st(i, j - 1, iv, jv)
                                        : (axis == 2) ? flux_1st(i, j, iv - 1, jv)
                                                      : flux_1st(i, j, iv, jv - 1);
                double flux_1st_right = flux_1st(i, j, iv, jv);

                double d_l            = flux_left - flux_1st_left;   // high order correction, left
                double d_r            = flux_right - flux_1st_right; // high order correction, right
                double delta =
                    -f(i, j, iv, jv, sp) - flux_1st_left + flux_1st_right; // 0.0 - 1st_order_update, always <= 0
                double p = d_l - d_r - delta;                              // p is the 3rd order update without limiter

                // Xiong2014 (http://dx.doi.org/10.1016/j.jcp.2014.05.033)
                if (d_l >= 0 && d_r <= 0.0) {
                    ep_l(i, j, iv, jv) = 1.0;
                    ep_r(i, j, iv, jv) = 1.0;
                } else if (d_l >= 0.0 && d_r > 0.0) {
                    ep_l(i, j, iv, jv) = 1.0;
                    ep_r(i, j, iv, jv) = min(1.0, delta / (-d_r));
                } else if (d_l < 0.0 && d_r <= 0.0) {
                    ep_l(i, j, iv, jv) = min(1.0, delta / d_l);
                    ep_r(i, j, iv, jv) = 1.0;
                } else if (d_l < 0.0 && d_r > 0.0) {
                    if (p >= 0) {
                        ep_l(i, j, iv, jv) = 1.0;
                        ep_r(i, j, iv, jv) = 1.0;
                    } else {
                        ep_l(i, j, iv, jv) = delta / (d_l - d_r);
                        ep_r(i, j, iv, jv) = delta / (d_l - d_r);
                    }
                }
            });

        // Step 3: Update distribution function
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                double eta   = eta_field(i, j);
                double eta_l = eta_field(i - 1, j);
                double eta_r = eta_field(i + 1, j);
                double eta_b = eta_field(i, j - 1);
                double eta_t = eta_field(i, j + 1);

                if (eta < 0.0)
                    return;

                // Get fluxes and limiters at left and right interfaces
                double flux_left      = (axis == 0)   ? flux(i - 1, j, iv, jv)
                                        : (axis == 1) ? flux(i, j - 1, iv, jv)
                                        : (axis == 2) ? flux(i, j, iv - 1, jv)
                                                      : flux(i, j, iv, jv - 1);
                double flux_right     = flux(i, j, iv, jv);
                double flux_1st_left  = (axis == 0)   ? flux_1st(i - 1, j, iv, jv)
                                        : (axis == 1) ? flux_1st(i, j - 1, iv, jv)
                                        : (axis == 2) ? flux_1st(i, j, iv - 1, jv)
                                                      : flux_1st(i, j, iv, jv - 1);
                double flux_1st_right = flux_1st(i, j, iv, jv);

                double ep_left = 0.0, ep_right = 0.0;
                if (axis == 0) {
                    ep_left  = (eta_l < 0.0) ? ep_l(i, j, iv, jv) : min(ep_r(i - 1, j, iv, jv), ep_l(i, j, iv, jv));
                    ep_right = (eta_r < 0.0) ? ep_r(i, j, iv, jv) : min(ep_r(i, j, iv, jv), ep_l(i + 1, j, iv, jv));
                } else if (axis == 1) {
                    ep_left  = (eta_b < 0.0) ? ep_l(i, j, iv, jv) : min(ep_r(i, j - 1, iv, jv), ep_l(i, j, iv, jv));
                    ep_right = (eta_t < 0.0) ? ep_r(i, j, iv, jv) : min(ep_r(i, j, iv, jv), ep_l(i, j + 1, iv, jv));
                } else if (axis == 2) {
                    ep_left  = min(ep_r(i, j, iv - 1, jv), ep_l(i, j, iv, jv));
                    ep_right = min(ep_r(i, j, iv, jv), ep_l(i, j, iv + 1, jv));
                } else {
                    ep_left  = min(ep_r(i, j, iv, jv - 1), ep_l(i, j, iv, jv));
                    ep_right = min(ep_r(i, j, iv, jv), ep_l(i, j, iv, jv + 1));
                }

                double flux_hat_l = ep_left * (flux_left - flux_1st_left) + flux_1st_left;
                double flux_hat_r = ep_right * (flux_right - flux_1st_right) + flux_1st_right;
                f(i, j, iv, jv, sp) += flux_hat_l - flux_hat_r;

                // integrity check: a correct MPP update should never produce a
                // meaningfully negative distribution in a fluid cell
                if (eta > 0.0 && f(i, j, iv, jv, sp) < -1e-16) {
                    auto [x, y, vx, vy]       = grid.center(i, j, iv, jv, sp);
                    auto [dx, dy, dvx, dvy]   = grid.spacing(sp);
                    double f_old              = f(i, j, iv, jv, sp) - (flux_hat_l - flux_hat_r);
                    double d_l                = flux_left - flux_1st_left;
                    double d_r                = flux_right - flux_1st_right;
                    double delta              = -f_old - flux_1st_left + flux_1st_right;
                    double p                  = d_l - d_r - delta;
                    double advection_velocity = 0.0;
                    double floor_velocity     = 0.0;
                    int s                     = 0;
                    if (axis == 0) {
                        advection_velocity = vx * dt / dx;
                        floor_velocity     = floor(advection_velocity);
                        s                  = i - (int)floor_velocity;
                    } else if (axis == 1) {
                        advection_velocity = vy * dt / dy;
                        floor_velocity     = floor(advection_velocity);
                        s                  = j - (int)floor_velocity;
                    } else if (axis == 2) {
                        advection_velocity = q[sp] / m[sp] * E(i, j, 0) * dt / dvx;
                        floor_velocity     = floor(advection_velocity);
                        s                  = iv - (int)floor_velocity;
                    } else if (axis == 3) {
                        advection_velocity = q[sp] / m[sp] * E(i, j, 1) * dt / dvy;
                        floor_velocity     = floor(advection_velocity);
                        s                  = jv - (int)floor_velocity;
                    }
                    Kokkos::printf("\nWarning: Negative distribution\n\
                        sp = %d, axis = %d, s = %d\n\
                        f(%d, %d, %d, %d) = %e\n\
                        f_old = %e\n\
                        (left) ep_r = %e, ep_l = %e   (right) ep_r = %e, ep_l = %e\n\
                        ep_left = %e, ep_right = %e\n\
                        flux_left = %e, flux_right = %e\n\
                        flux_1st_left = %e, flux_1st_right = %e\n\
                        flux_hat_l = %e, flux_hat_r = %e\n\
                        v_adv = %e\n\
                        d_l = %e, d_r = %e, delta = %e, p = %e\n",
                                   sp, axis, s, i, j, iv, jv, f(i, j, iv, jv, sp), f_old, ep_r(i - 1, j, iv, jv),
                                   ep_l(i, j, iv, jv), ep_r(i, j, iv, jv), ep_l(i + 1, j, iv, jv), ep_left, ep_right,
                                   flux_left, flux_right, flux_1st_left, flux_1st_right, flux_hat_l, flux_hat_r,
                                   advection_velocity, d_l, d_r, delta, p);
                    if (f(i, j, iv, jv, sp) < -1e-15)
                        Kokkos::abort("Abort: Negative distribution");
                }

                // fix any negative value due to numerical error
                f(i, j, iv, jv, sp) = max(f(i, j, iv, jv, sp), 0.0);
            });
    }

    void advance(double dt) {
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        for (int sp = 0; sp < N; ++sp) {
            pfc_update(dt / 2.0, 0, sp);
            pfc_update(dt / 2.0, 1, sp);
        }
        Kokkos::printf("(VlasovSolver) Applying Particle BC\n");
        world.particle_boundary_conditions();
        Kokkos::printf("(VlasovSolver) IB Extrapolation\n");
        extrapolate_distribution_2nd_order();
        Kokkos::printf("(VlasovSolver) Computing charge density\n");
        compute_charge_density();
        Kokkos::printf("(VlasovSolver) Solving electric field\n");
        poisson_solver.solve();
        poisson_solver.compute_electric_field();
        Kokkos::printf("(VlasovSolver) PFC update along velocity by dt\n");
        for (int sp = 0; sp < N; ++sp) {
            pfc_update(dt, 2, sp);
            pfc_update(dt, 3, sp);
        }
        Kokkos::printf("(VlasovSolver) Applying Particle BC\n");
        world.particle_boundary_conditions();
        Kokkos::printf("(VlasovSolver) IB Extrapolation\n");
        extrapolate_distribution_2nd_order();
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        for (int sp = 0; sp < N; ++sp) {
            pfc_update(dt / 2.0, 0, sp);
            pfc_update(dt / 2.0, 1, sp);
        }
        Kokkos::printf("(VlasovSolver) Applying Particle BC\n");
        world.particle_boundary_conditions();
        Kokkos::printf("(VlasovSolver) IB Extrapolation\n");
        extrapolate_distribution_2nd_order();
        Kokkos::printf("(VlasovSolver) Computing charge density\n");
        compute_charge_density();
    }

    void solve() {
        Kokkos::printf("Step %zu:\n", 0);
        world.initialize_distribution();
        world.particle_boundary_conditions();
        extrapolate_distribution_2nd_order();
        compute_charge_density();
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
