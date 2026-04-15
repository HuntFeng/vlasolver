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
#include "matrix/solve.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>

template <typename World, typename PoissonSolver>
class Vlasolver {
  private:
    World& world;
    PoissonSolver& poisson_solver;
    Writer<World>& writer;

  public:
    Vlasolver(World& world, PoissonSolver& poisson_solver, Writer<World>& writer)
        : world(world),
          poisson_solver(poisson_solver),
          writer(writer) {}

    void extrapolate_distribution_1st_order() const {
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
                    // f(i, j, iv, jv) = extrapolated_value / Ng; // this doesn't work
                    f(i, j, iv, jv) = Kokkos::max(0.0, extrapolated_value / Ng);
            });
    }

    void extrapolate_distribution_2nd_order_normal() const {
        using Kokkos::abs;
        using Kokkos::sqrt;
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing[0];
        double dy               = grid.spacing[1];

        // first point must be further enough so interpolation won't involve ghost cell
        double s = sqrt(dx * dx + dy * dy); // cell diagonal length

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});

                // always extrapolate dist function from the interior of the immersed object
                double eta = world.surface(x, y);
                if (eta >= 0.0)
                    return;

                // now (x,y) is the interior of the immersed object
                double eta_l   = world.surface(x - dx, y);
                double eta_r   = world.surface(x + dx, y);
                double eta_b   = world.surface(x, y - dy);
                double eta_t   = world.surface(x, y + dy);
                auto [n1, n2]  = world.normal(x, y, dx, dy);
                double v_dot_n = vx * n1 + vy * n2;
                // extrapolate outflow (v.n < 0), zero-inflow(v.n >= 0)
                if (eta * eta_l < 0.0 || eta * eta_r < 0.0 || eta * eta_b < 0.0 || eta * eta_t < 0.0) {

                    // best fit line
                    const size_t N = 4;
                    Kokkos::Array<double, N> s_vals;
                    Kokkos::Array<double, N> f_vals;
                    for (int k = 0; k < N; ++k) {
                        s_vals[k]                 = (k + 1) * s;
                        double x_k                = x + s_vals[k] * n1;
                        double y_k                = y + s_vals[k] * n2;
                        auto [ic, jc, ivxc, ivyc] = grid.coord({x_k, y_k, vx, vy});
                        auto [xc, yc, vxc, vyc]   = grid.center({ic, jc, ivxc, ivyc});
                        f_vals[k] = f(ic, jc, iv, jv) * (1.0 - (x_k - xc) / dx) * (1.0 - (y_k - yc) / dy) +
                                    f(ic + 1, jc, iv, jv) * ((x_k - xc) / dx) * (1.0 - (y_k - yc) / dy) +
                                    f(ic, jc + 1, iv, jv) * (1.0 - (x_k - xc) / dx) * ((y_k - yc) / dy) +
                                    f(ic + 1, jc + 1, iv, jv) * ((x_k - xc) / dx) * ((y_k - yc) / dy);
                    }
                    // for inflow case, we take the zero-inflow condition into account
                    double c0 = 0.0, c1 = 0.0;
                    if (v_dot_n >= 0.0) {
                        s_vals[N - 1] = abs(eta);
                        f_vals[N - 1] = 0.0;
                        double bot = 0.0, top = 0.0;
                        for (int k = 0; k < N; ++k) {
                            top += s_vals[k] * f_vals[k];
                            bot += s_vals[k] * s_vals[k];
                        }
                        c1 = top / bot;
                        c0 = 0.0;
                    } else {
                        double f_mean = 0.0, s_mean = 0.0, bot = 0.0, top = 0.0;
                        for (int k = 0; k < N; ++k) {
                            if (s_mean == 0.0) {
                                for (int m = 0; m < N; ++m) {
                                    s_mean += s_vals[k] / N;
                                    f_mean += f_vals[k] / N;
                                }
                            }

                            top += (s_vals[k] - s_mean) * (f_vals[k] - f_mean);
                            bot += (s_vals[k] - s_mean) * (s_vals[k] - s_mean);
                        }

                        c1 = top / bot;
                        c0 = f_mean - c1 * s_mean;
                    }

                    f(i, j, iv, jv) = c0 + c1 * eta; // eta is negative

                    // double f_F1 = 0.0, f_F2 = 0.0;
                    // {
                    //     double x1                 = x + s1 * n1;
                    //     double y1                 = y + s1 * n2;
                    //     auto [ic, jc, ivxc, ivyc] = grid.coord({x1, y1, vx, vy});
                    //     auto [xc, yc, vxc, vyc]   = grid.center({ic, jc, ivxc, ivyc});
                    //     f_F1 = f(ic, jc, iv, jv) * (1.0 - (x1 - xc) / dx) * (1.0 - (y1 - yc) / dy) +
                    //            f(ic + 1, jc, iv, jv) * ((x1 - xc) / dx) * (1.0 - (y1 - yc) / dy) +
                    //            f(ic, jc + 1, iv, jv) * (1.0 - (x1 - xc) / dx) * ((y1 - yc) / dy) +
                    //            f(ic + 1, jc + 1, iv, jv) * ((x1 - xc) / dx) * ((y1 - yc) / dy);
                    // };
                    // {
                    //     double x2                 = x + s2 * n1;
                    //     double y2                 = y + s2 * n2;
                    //     auto [ic, jc, ivxc, ivyc] = grid.coord({x2, y2, vx, vy});
                    //     auto [xc, yc, vxc, vyc]   = grid.center({ic, jc, ivxc, ivyc});
                    //     f_F2 = f(ic, jc, iv, jv) * (1.0 - (x2 - xc) / dx) * (1.0 - (y2 - yc) / dy) +
                    //            f(ic + 1, jc, iv, jv) * ((x2 - xc) / dx) * (1.0 - (y2 - yc) / dy) +
                    //            f(ic, jc + 1, iv, jv) * (1.0 - (x2 - xc) / dx) * ((y2 - yc) / dy) +
                    //            f(ic + 1, jc + 1, iv, jv) * ((x2 - xc) / dx) * ((y2 - yc) / dy);
                    // };
                    //
                    // double f_I =
                    //     (v_dot_n < 0.0) ? (s2 - abs(eta)) / (s2 - s1) * f_F1 + (abs(eta) - s1) / (s2 - s1) * f_F2 :
                    //     0.0;
                    // f(i, j, iv, jv) = s1 / (s1 - abs(eta)) * f_I - abs(eta) / (s1 - abs(eta)) * f_F1;
                    f(i, j, iv, jv) = Kokkos::max(f(i, j, iv, jv), 0.0);
                }
            });
    }

    void extrapolate_distribution_2nd_order() const {
        using Kokkos::abs;
        using Kokkos::sqrt;
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing[0];
        double dy               = grid.spacing[1];
        // search region range from -offset_range to +offset_range
        const int offset_range = 5;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x0, y0, vx, vy] = grid.center({i, j, iv, jv});

                // always extrapolate dist function from the interior of the immersed object
                double eta = world.surface(x0, y0);
                if (eta >= 0.0)
                    return;

                // now (x0,y0) is the interior of the immersed object
                double eta_l   = world.surface(x0 - dx, y0);
                double eta_r   = world.surface(x0 + dx, y0);
                double eta_b   = world.surface(x0, y0 - dy);
                double eta_t   = world.surface(x0, y0 + dy);
                auto [n1, n2]  = world.normal(x0, y0, dx, dy);
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
                            auto [x, y, vx, vy] = grid.center({I, J, iv, jv});
                            // skip ghost cells
                            if (I < ngc || I > nx - ngc - 1 || J < ngc || J > ny - ngc - 1 || world.surface(x, y) < 0)
                                continue;

                            // only extrapolate using fluid cells
                            double z = f(I, J, iv, jv);
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
                    f(i, j, iv, jv) = b[0] * x0 + b[1] * y0 + b[2];
                }
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

    void pfc_update(double dt, int axis) const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& flux              = world.flux;
        auto& flux_1st          = world.flux_1st;
        auto& ep_l              = world.ep_l;
        auto& ep_r              = world.ep_r;

        auto [dx, dy, dvx, dvy] = grid.spacing;
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
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
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
                double flux_val   = 0.0;
                double nu         = 0.0;
                if (advection_velocity >= 0.0) {
                    nu       = advection_velocity - floor_v;
                    flux_val = f0;
                    flux_val += (1 - nu) * (2 - nu) * plus_diff / 6.0;
                    flux_val += (1 - nu) * (1 + nu) * minus_diff / 6.0;
                    flux_val *= nu;
                } else {
                    nu       = advection_velocity - (floor_v + 1);
                    flux_val = f0;
                    flux_val += -(1 - nu) * (1 + nu) * plus_diff / 6.0;
                    flux_val += -(2 + nu) * (1 + nu) * minus_diff / 6.0;
                    flux_val *= nu;
                }

                double flux_shift = 0.0;
                if (axis == 0) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= i; ++n)
                            flux_shift += f(n, j, iv, jv);
                    } else {
                        for (int n = i + 1; n <= min(s - 1, f.extent_int(0) - 1); ++n)
                            flux_shift += f(n, j, iv, jv);
                    }
                } else if (axis == 1) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= j; ++n)
                            flux_shift += f(i, n, iv, jv);
                    } else {
                        for (int n = j + 1; n <= min(s - 1, f.extent_int(1) - 1); ++n)
                            flux_shift += f(i, n, iv, jv);
                    }
                } else if (axis == 2) {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= iv; ++n)
                            flux_shift += f(i, j, n, jv);
                    } else {
                        for (int n = iv + 1; n <= min(s - 1, f.extent_int(2) - 1); ++n)
                            flux_shift += f(i, j, n, jv);
                    }
                } else {
                    if (advection_velocity >= 0.0) {
                        for (int n = max(s + 1, 0); n <= jv; ++n)
                            flux_shift += f(i, j, iv, n);
                    } else {
                        for (int n = jv + 1; n <= min(s - 1, f.extent_int(3) - 1); ++n)
                            flux_shift += f(i, j, iv, n);
                    }
                }
                // Store flux at right interface of cell i
                flux(i, j, iv, jv)     = flux_val + flux_shift;
                flux_1st(i, j, iv, jv) = nu * f0 + flux_shift;
            });
        Kokkos::fence();

        // Step 2: Compute flux limiters
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // flux at left interface is flux(i-1,...), flux at right interface is flux(i,...)
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

                double d_l            = flux_left - flux_1st_left;
                double d_r            = flux_right - flux_1st_right;
                double delta          = -f(i, j, iv, jv) - flux_1st_left + flux_1st_right;
                if (delta < 1e-16)
                    delta = Kokkos::min(delta, 0.0);
                double p       = d_l - d_r - delta;

                double ep_left = 1.0, ep_right = 1.0;
                if (d_l < 0.0 && d_r <= 0.0) {
                    ep_l(i, j, iv, jv) = Kokkos::min(1.0, delta / d_l);
                } else if (d_l * d_r < 0.0 && p < 0.0) {
                    ep_l(i, j, iv, jv) = delta / (d_l - d_r);
                }
                if (d_l >= 0.0 && d_r > 0.0) {
                    ep_r(i, j, iv, jv) = Kokkos::min(1.0, -delta / d_r);
                } else if (d_l * d_r < 0.0 && p < 0.0) {
                    ep_r(i, j, iv, jv) = delta / (d_l - d_r);
                }
            });
        Kokkos::fence();

        // Step 3: Update distribution function
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
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

                auto [x, y, vx, vy]   = grid.center({i, j, iv, jv});
                double eta            = world.surface(x, y);
                double eta_l          = world.surface(x - dx, y);
                double eta_r          = world.surface(x + dx, y);
                double eta_b          = world.surface(x, y - dy);
                double eta_t          = world.surface(x, y + dy);
                double ep_left = 0.0, ep_right = 0.0;
                if (eta > 0.0) {
                    if (axis == 0) {
                        ep_left  = (eta_l < 0.0) ? ep_l(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i - 1, j, iv, jv), ep_l(i, j, iv, jv));
                        ep_right = (eta_r < 0.0) ? ep_r(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv, jv), ep_l(i + 1, j, iv, jv));
                    } else if (axis == 1) {
                        ep_left  = (eta_b < 0.0) ? ep_l(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j - 1, iv, jv), ep_l(i, j, iv, jv));
                        ep_right = (eta_t < 0.0) ? ep_r(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j + 1, iv, jv));
                    } else if (axis == 2) {
                        ep_left  = (eta_l < 0.0) ? ep_l(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv - 1, jv), ep_l(i, j, iv, jv));
                        ep_right = (eta_r < 0.0) ? ep_r(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv + 1, jv));
                        // ep_left  = Kokkos::min(ep_r(i, j, iv - 1, jv), ep_l(i, j, iv, jv));
                        // ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv + 1, jv));
                    } else {
                        ep_left  = (eta_b < 0.0) ? ep_l(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv, jv - 1), ep_l(i, j, iv, jv));
                        ep_right = (eta_t < 0.0) ? ep_r(i, j, iv, jv)
                                                 : Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv, jv + 1));
                        // ep_left  = Kokkos::min(ep_r(i, j, iv, jv - 1), ep_l(i, j, iv, jv));
                        // ep_right = Kokkos::min(ep_r(i, j, iv, jv), ep_l(i, j, iv, jv + 1));
                    }

                    if (ep_left < -1e-16 || ep_right < -1e-16 || ep_left > 1.0 + 1e-16 || ep_right > 1.0 + 1e-16) {
                        Kokkos::printf("(axis %d) ep_left(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, ep_left);
                        Kokkos::printf("(axis %d) ep_right(%d, %d, %d, %d) = %e\n", axis, i, j, iv, jv, ep_right);
                        Kokkos::abort("Abort: Limiter out of bound");
                    }
                    // fix any negative values due to numerical error
                    ep_l(i, j, iv, jv) = Kokkos::max(ep_l(i, j, iv, jv), 0.0);
                    ep_r(i, j, iv, jv) = Kokkos::max(ep_r(i, j, iv, jv), 0.0);
                }

                double flux_hat_l = ep_left * (flux_left - flux_1st_left) + flux_1st_left;
                double flux_hat_r = ep_right * (flux_right - flux_1st_right) + flux_1st_right;

                f(i, j, iv, jv) += flux_hat_l - flux_hat_r;
                // fix any negative value due to numerical error
                f(i, j, iv, jv) = Kokkos::max(0.0, f(i, j, iv, jv));
            });
        Kokkos::fence();
    }

    void advance(double dt) {
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        pfc_update(dt / 2.0, 0);
        pfc_update(dt / 2.0, 1);
        Kokkos::printf("(VlasovSolver) Solving electric field\n");
        world.particle_boundary_conditions();
        // extrapolate_distribution_1st_order();
        extrapolate_distribution_2nd_order();
        compute_charge_density();
        poisson_solver.solve();
        poisson_solver.compute_electric_field();
        Kokkos::printf("(VlasovSolver) PFC update along velocity by dt\n");
        pfc_update(dt, 2);
        // Kokkos::printf("(VlasovSolver) PFC update along velocity by dt, vy\n");
        pfc_update(dt, 3);
        world.particle_boundary_conditions();
        // extrapolate_distribution_1st_order();
        extrapolate_distribution_2nd_order();
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        pfc_update(dt / 2.0, 0);
        pfc_update(dt / 2.0, 1);
        world.particle_boundary_conditions();
        // extrapolate_distribution_1st_order();
        extrapolate_distribution_2nd_order();
        compute_charge_density();
    }

    void solve() {
        Kokkos::printf("Step %zu:\n", 0);
        world.initialize_distribution();
        world.particle_boundary_conditions();
        // extrapolate_distribution_1st_order();
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
