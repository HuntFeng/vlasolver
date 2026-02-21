/*
 * Solve laplacian phi = -rho (Au=rhs)
 * This Poisson sovler uses the algorithm described in:
 * A Second-Order Boundary Condition Capturing Method for Solving the Elliptic Interface Problems on Irregular Domains
 * by Hyuntae Cho 2019, Journal of Scientific Computing, doi: https://doi.org/10.1007/s10915-019-01016-y
 *
 **/
#pragma once
#include "reduced/world.hpp"
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_LUPrec.hpp>
#include <KokkosSparse_gmres.hpp>
#include <KokkosSparse_par_ilut.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <Kokkos_Core.hpp>
#include <bit>
#include <vector>

enum Direction : size_t {
    R = 1 << 0, // 0001
    T = 1 << 1, // 0010
    L = 1 << 2, // 0100
    B = 1 << 3, // 1000
};

struct InterfaceValue {
    double u_l, u_r, u_b, u_t;
    double theta_l, theta_r, theta_b, theta_t;
};

template <typename World>
class PoissonSolver {
  private:
    // some types
    using EXSP         = Kokkos::DefaultExecutionSpace;
    using MESP         = EXSP::memory_space;
    using CRS          = KokkosSparse::CrsMatrix<double, int, EXSP>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;

    KernelHandle kh;

    // input params
    World& world;
    double tol;
    int gmres_m;
    int max_restart;
    bool verbose;

    // some useful params
    const int nx    = world.grid.ncells[0];
    const int ny    = world.grid.ncells[1];
    const double dx = world.grid.spacing[0];
    const double dy = world.grid.spacing[1];

    // using coordinate format for constructing sparse matrix -nabla^2
    std::vector<int> rows_coo;
    std::vector<int> cols_coo;
    std::vector<double> vals_coo;

    // use to crs format for GMRES performance
    CRS A;
    // KokkosSparse::Experimental::LUPrec<CRS, KernelHandle> prec;
    std::unique_ptr<KokkosSparse::Experimental::LUPrec<CRS, KernelHandle>> prec;
    Kokkos::View<double*> u;
    Kokkos::View<double*> rhs;
    // rhs_h encodes jumps and boundary conditions
    Kokkos::View<double*, Kokkos::HostSpace> rhs_h;

    // jump conditions
    Kokkos::View<double**, Kokkos::HostSpace> a;
    Kokkos::View<double**, Kokkos::HostSpace> b;
    Kokkos::View<double**, Kokkos::HostSpace> a_tau;

    // normal
    Kokkos::View<double**, Kokkos::HostSpace> n1;
    Kokkos::View<double**, Kokkos::HostSpace> n2;

  public:
    PoissonSolver(World& world, double tol = 1e-12, int gmres_m = 100, int max_restart = 30, bool verbose = false)
        : world(world),
          tol(tol),
          gmres_m(gmres_m),
          max_restart(max_restart),
          verbose(verbose) {

        construct_fields();
        construct_matrix();
        construct_preconditioner();
        // construct_preconditioner_spiluk();

        // prepare gmres
        kh.create_gmres_handle(gmres_m, tol, max_restart);
        auto gmres_handle = kh.get_gmres_handle();
        using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
        gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
        gmres_handle->set_verbose(verbose);
    }

    KOKKOS_INLINE_FUNCTION
    int index(int i, int j) const { return i * ny + j; }

    inline bool isclose(double val1, double val2) { return Kokkos::abs(val1 - val2) < 1e-6 ? true : false; }

    double compute_theta(size_t direction, int i, int j) {
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;

        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double eta          = world.surface(x, y);
        double eta_r        = world.surface(x + dx, y);
        double eta_l        = world.surface(x - dx, y);
        double eta_t        = world.surface(x, y + dy);
        double eta_b        = world.surface(x, y - dy);

        double dx_eta       = (eta_r - eta_l) / 2;
        double dy_eta       = (eta_t - eta_b) / 2;
        double dxx_eta      = (eta_r - 2 * eta + eta_l) / 2;
        double dyy_eta      = (eta_t - 2 * eta + eta_b) / 2;

        double d1 = 0.0, d2 = 0.0;
        double dir_sign = 0.0;
        double eta_sign = eta > 0.0 ? 1.0 : -1.0;
        switch (direction) {
        case Direction::R:
            d1       = dx_eta;
            d2       = dxx_eta;
            dir_sign = -1.0;
            break;
        case Direction::T:
            d1       = dy_eta;
            d2       = dyy_eta;
            dir_sign = -1.0;
            break;
        case Direction::L:
            d1       = dx_eta;
            d2       = dxx_eta;
            dir_sign = +1.0;
            break;
        case Direction::B:
            d1       = dy_eta;
            d2       = dyy_eta;
            dir_sign = +1.0;
            break;
        default:
            return 1.0;
        }

        if (isclose(d2, 0.0))
            return abs(eta / d1);

        double disc = d1 * d1 - 4.0 * d2 * eta;
        return (dir_sign * d1 - eta_sign * sqrt(disc)) / (2.0 * d2);
    }

    /**
     * cubic interpolation
     */
    double interp(size_t direction, double theta, int i, int j, Kokkos::View<double**, Kokkos::HostSpace>& field) {
        using Kokkos::pow;
        // double t_matrix[4]    = {1, theta, pow(theta, 2), pow(theta, 3)};
        // double c_matrix[4][4] = {
        //     {0.0, 2.0, 0.0, 0.0},
        //     {-1.0, 0.0, 1.0, 0.0},
        //     {2.0, -5.0, 4.0, -1.0},
        //     {-1.0, 3.0, -3.0, 1.0},
        // };
        // double points[4];
        // if (direction == Direction::R)
        //     double points[4] = {field(i - 1, j), field(i, j), field(i + 1, j), field(i + 2, j)};
        // else if (direction == Direction::T)
        //     double points[4] = {field(i, j - 1), field(i, j), field(i, j + 1), field(i, j + 2)};
        // else if (direction == Direction::L)
        //     double points[4] = {field(i + 1, j), field(i, j), field(i - 1, j), field(i - 2, j)};
        // else if (direction == Direction::B)
        //     double points[4] = {field(i, j + 1), field(i, j), field(i, j - 1), field(i, j - 2)};
        // else
        //     Kokkos::abort("Interp invalid direction for interpolation");
        Kokkos::Array<double, 4> t_matrix{1, theta, pow(theta, 2), pow(theta, 3)};
        Kokkos::Array<Kokkos::Array<double, 4>, 4> c_matrix{{
            {0.0, 2.0, 0.0, 0.0},
            {-1.0, 0.0, 1.0, 0.0},
            {2.0, -5.0, 4.0, -1.0},
            {-1.0, 3.0, -3.0, 1.0},
        }};
        Kokkos::Array<double, 4> points;
        if (direction == Direction::R)
            points = {field(i - 1, j), field(i, j), field(i + 1, j), field(i + 2, j)};
        else if (direction == Direction::T)
            points = {field(i, j - 1), field(i, j), field(i, j + 1), field(i, j + 2)};
        else if (direction == Direction::L)
            points = {field(i + 1, j), field(i, j), field(i - 1, j), field(i - 2, j)};
        else if (direction == Direction::B)
            points = {field(i, j + 1), field(i, j), field(i, j - 1), field(i, j - 2)};
        else
            Kokkos::abort("Interp invalid direction for interpolation");
        double val_I = 0.0;
        for (int p = 0; p < 4; ++p) {
            for (int q = 0; q < 4; ++q) {
                val_I += 0.5 * t_matrix[p] * c_matrix[p][q] * points[q];
            }
        }

        return val_I;
    }

    /**
     * Matrix entry for cells having no cuts by interface
     */
    void coeff_case0(int i, int j) {
        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double bot_x        = dx * dx;
        double bot_y        = dy * dy;

        double eps_l        = world.permittivity(x - dx / 2, y);
        double eps_r        = world.permittivity(x + dx / 2, y);
        double eps_b        = world.permittivity(x, y - dy / 2);
        double eps_t        = world.permittivity(x, y + dy / 2);

        int row_idx         = index(i, j);
        rows_coo.insert(rows_coo.end(), {
                                            row_idx,
                                            row_idx,
                                            row_idx,
                                            row_idx,
                                            row_idx,
                                        });
        cols_coo.insert(cols_coo.end(), {
                                            index(i - 1, j),
                                            index(i + 1, j),
                                            index(i, j - 1),
                                            index(i, j + 1),
                                            index(i, j),
                                        });

        vals_coo.insert(vals_coo.end(), {
                                            // u_[i - 1, j]
                                            eps_l / bot_x,
                                            // u_[i + 1, j]
                                            eps_r / bot_x,
                                            // u_[i, j - 1]
                                            eps_b / bot_y,
                                            // u_[i, j + 1]
                                            eps_t / bot_y,
                                            // u_[i, j]
                                            (-(eps_l + eps_r) / bot_x - (eps_b + eps_t) / bot_y),
                                        });
    }

    /**
     * Matrix entry for cells having 1 cut by interface
     */
    void coeff_case1(size_t direction, int i, int j) {
        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        int row_idx         = index(i, j);         // laplacian matrix row index
        double eta          = world.surface(x, y); // assume this is negative for now
        double theta        = compute_theta(direction, i, j);
        double a_tau_I      = interp(direction, theta, i, j, a_tau);
        double a_I          = interp(direction, theta, i, j, a);
        double b_I          = interp(direction, theta, i, j, b);
        double n1_I         = interp(direction, theta, i, j, n1);
        double n2_I         = interp(direction, theta, i, j, n2);

        if (direction == Direction::R) {
            double theta_l = 1.0;
            double theta_r = theta;
            double theta_t = 1.0;
            double theta_b = 1.0;

            // common denominator in discretization
            double bot_x = (theta_r + theta_l) / 2 * dx * dx;
            double bot_y = (theta_t + theta_b) / 2 * dy * dy;

            // world.permittivity
            double eps_r = world.permittivity(x + theta_r * dx / 2, y);
            double eps_l = world.permittivity(x - dx / 2, y);
            double eps_t = world.permittivity(x, y + dy / 2);
            double eps_b = world.permittivity(x, y - dy / 2);

            double eps_p, eps_m, eps_jump, _eps_p, _eps_m;
            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y);
                eps_jump = _eps_p - _eps_m;
                // swap these two variable in the d expression
                eps_p = _eps_m;
                eps_m = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = (-a_tau_I * eps_p * n2_I * dx + b_I * n1_I * dx +
                        a_I * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r)));

            if (eta > 0) {
                // in the following formulas, world.permittivity signs are also swapped
                // the world.permittivity jump stays the same
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M    = (-eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r)) -
                        eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1)) -
                        eps_jump * n2_I * n2_I * (2 * theta_r + 1) / (theta_r * (theta_r + 1)));

            double N[7] = {
                // u[i,j]
                -eps_jump * n1_I * n2_I * theta_r * dx / dy -
                    (eps_jump * n2_I * n2_I + eps_m) * (1 + theta_r) / theta_r,
                // u[i+1,j]
                -eps_p * (theta_r - 2) / (theta_r - 1),
                // u[i+2,j]
                eps_p * (theta_r - 1) / (theta_r - 2),
                // u[i-1,j]
                eps_jump * n1_I * n2_I * theta_r * dx / dy + (eps_jump * n2_I * n2_I + eps_m) * theta_r / (1 + theta_r),
                // u[i,j-1]
                eps_jump * n1_I * n2_I * (2 * theta_r + 1) * dx / (2 * dy),
                // u[i,j+1]
                -eps_jump * n1_I * n2_I * dx / (2 * dy),
                // u[i-1,j-1]
                -eps_jump * n1_I * n2_I * theta_r * dx / dy,
            };

            rhs_h(row_idx) -= (d / M) * eps_r / theta_r / bot_x;

            rows_coo.insert(rows_coo.end(), 7, row_idx);
            cols_coo.insert(cols_coo.end(), {
                                                index(i, j),
                                                index(i + 1, j),
                                                index(i + 2, j),
                                                index(i - 1, j),
                                                index(i, j - 1),
                                                index(i, j + 1),
                                                index(i - 1, j - 1),
                                            });
            vals_coo.insert(vals_coo.end(),
                            {
                                // u[i,j]
                                (N[0] / M) * eps_r / theta_r / bot_x - (eps_r / theta_r + eps_l / theta_l) / bot_x -
                                    (eps_t / theta_t + eps_b / theta_b) / bot_y,
                                // u[i+1,j]
                                (N[1] / M) * eps_r / theta_r / bot_x,
                                // u[i+2,j]
                                (N[2] / M) * eps_r / theta_r / bot_x,
                                // u[i-1,j]
                                (N[3] / M) * eps_r / theta_r / bot_x + eps_l / theta_l / bot_x,
                                // u[i,j-1]
                                (N[4] / M) * eps_r / theta_r / bot_x + eps_b / theta_b / bot_y,
                                // u[i,j+1]
                                (N[5] / M) * eps_r / theta_r / bot_x + eps_t / theta_t / bot_y,
                                // u_ext at [i-1,j-1]
                                (N[6] / M) * eps_r / theta_r / bot_x,
                            });
        } else if (direction == Direction::T) {
            double theta_l = 1.0;
            double theta_r = 1.0;
            double theta_t = theta;
            double theta_b = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + dx / 2.0, y);
            double eps_l   = world.permittivity(x - dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + theta_t * dy / 2.0);
            double eps_b   = world.permittivity(x, y - dy / 2.0);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = a_tau_I * eps_p * n1_I * dy + b_I * n2_I * dy +
                       a_I * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                       eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                       eps_jump * (n1_I * n1_I) * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            double N[7] = {// u[i,j]
                           -eps_jump * n1_I * n2_I * theta_t * dy / dx -
                               (eps_jump * (n1_I * n1_I) + eps_m) * (1.0 + theta_t) / theta_t,
                           // u[i,j+1]
                           -eps_p * (theta_t - 2.0) / (theta_t - 1.0),
                           // u[i,j+2]
                           eps_p * (theta_t - 1.0) / (theta_t - 2.0),
                           // u[i,j-1]
                           eps_jump * n1_I * n2_I * theta_t * dy / dx +
                               (eps_jump * (n1_I * n1_I) + eps_m) * theta_t / (1.0 + theta_t),
                           // u[i-1,j]
                           eps_jump * n1_I * n2_I * (2.0 * theta_t + 1.0) * dy / (2.0 * dx),
                           // u[i+1,j]
                           -eps_jump * n1_I * n2_I * dy / (2.0 * dx),
                           // u[i-1,j-1]
                           -eps_jump * n1_I * n2_I * theta_t * dy / dx};

            rhs_h(row_idx) -= (d / M) * eps_t / theta_t / bot_y;

            rows_coo.insert(rows_coo.end(), 7, row_idx);

            cols_coo.insert(cols_coo.end(), {
                                                index(i, j),
                                                index(i, j + 1),
                                                index(i, j + 2),
                                                index(i, j - 1),
                                                index(i - 1, j),
                                                index(i + 1, j),
                                                index(i - 1, j - 1),
                                            });

            vals_coo.insert(vals_coo.end(),
                            {
                                (N[0] / M) * eps_t / theta_t / bot_y - (eps_r / theta_r + eps_l / theta_l) / bot_x -
                                    (eps_t / theta_t + eps_b / theta_b) / bot_y,
                                (N[1] / M * eps_t / theta_t / bot_y),
                                (N[2] / M * eps_t / theta_t / bot_y),
                                (N[3] / M * eps_t / theta_t / bot_y + eps_b / theta_b / bot_y),
                                (N[4] / M * eps_t / theta_t / bot_y + eps_l / theta_l / bot_x),
                                (N[5] / M * eps_t / theta_t / bot_y + eps_r / theta_r / bot_x),
                                (N[6] / M * eps_t / theta_t / bot_y),
                            });
        } else if (direction == Direction::L) {
            double theta_l = theta;
            double theta_r = 1.0;
            double theta_t = 1.0;
            double theta_b = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + dx / 2.0, y);
            double eps_l   = world.permittivity(x - theta_l * dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + dy / 2.0);
            double eps_b   = world.permittivity(x, y - dy / 2.0);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = -a_tau_I * eps_p * n2_I * dx + b_I * n1_I * dx -
                       a_I * eps_p * (3.0 - 2.0 * theta_l) / ((2.0 - theta_l) * (1.0 - theta_l));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = eps_p * (3.0 - 2.0 * theta_l) / ((1.0 - theta_l) * (2.0 - theta_l)) +
                       eps_m * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0)) +
                       eps_jump * (n2_I * n2_I) * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0));

            double N[7] = {// u[i,j]
                           -eps_jump * n1_I * n2_I * theta_l * dx / dy +
                               (eps_jump * (n2_I * n2_I) + eps_m) * (1.0 + theta_l) / theta_l,
                           // u[i-1,j]
                           eps_p * (theta_l - 2.0) / (theta_l - 1.0),
                           // u[i-2,j]
                           -eps_p * (theta_l - 1.0) / (theta_l - 2.0),
                           // u[i+1,j]
                           eps_jump * n1_I * n2_I * theta_l * dx / dy -
                               (eps_jump * (n2_I * n2_I) + eps_m) * theta_l / (1.0 + theta_l),
                           // u[i,j-1]
                           eps_jump * n1_I * n2_I * (2.0 * theta_l + 1.0) * dx / (2.0 * dy),
                           // u[i,j+1]
                           -eps_jump * n1_I * n2_I * dx / (2.0 * dy),
                           // u[i+1,j-1]
                           -eps_jump * n1_I * n2_I * theta_l * dx / dy};

            rhs_h(row_idx) -= (d / M) * eps_l / theta_l / bot_x;

            rows_coo.insert(rows_coo.end(), 7, row_idx);

            cols_coo.insert(cols_coo.end(), {
                                                index(i, j),
                                                index(i - 1, j),
                                                index(i - 2, j),
                                                index(i + 1, j),
                                                index(i, j - 1),
                                                index(i, j + 1),
                                                index(i + 1, j - 1),
                                            });

            vals_coo.insert(vals_coo.end(),
                            {
                                // u[i,j]
                                (N[0] / M) * eps_l / theta_l / bot_x - (eps_r / theta_r + eps_l / theta_l) / bot_x -
                                    (eps_t / theta_t + eps_b / theta_b) / bot_y,
                                // u[i-1,j]
                                (N[1] / M) * eps_l / theta_l / bot_x,
                                // u[i-2,j]
                                (N[2] / M) * eps_l / theta_l / bot_x,
                                // u[i+1,j]
                                (N[3] / M) * eps_l / theta_l / bot_x + eps_r / theta_r / bot_x,
                                // u[i,j+1]
                                (N[4] / M) * eps_l / theta_l / bot_x + eps_t / theta_t / bot_y,
                                // u[i,j-1]
                                (N[5] / M) * eps_l / theta_l / bot_x + eps_b / theta_b / bot_y,
                                // u_ext at [i+1,j-1]
                                (N[6] / M) * eps_l / theta_l / bot_x,
                            });
        } else if (direction == Direction::B) {
            double theta_l = 1.0;
            double theta_r = 1.0;
            double theta_t = 1.0;
            double theta_b = theta;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + dx / 2.0, y);
            double eps_l   = world.permittivity(x - dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + dy / 2.0);
            double eps_b   = world.permittivity(x, y - theta_b * dy / 2.0);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = a_tau_I * eps_p * n1_I * dy + b_I * n2_I * dy -
                       a_I * eps_p * (3.0 - 2.0 * theta_b) / ((2.0 - theta_b) * (1.0 - theta_b));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = eps_p * (3.0 - 2.0 * theta_b) / ((1.0 - theta_b) * (2.0 - theta_b)) +
                       eps_m * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0)) +
                       eps_jump * (n1_I * n1_I) * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0));

            double N[7] = {// u[i,j]
                           -eps_jump * n1_I * n2_I * theta_b * dy / dx +
                               (eps_jump * (n1_I * n1_I) + eps_m) * (1.0 + theta_b) / theta_b,
                           // u[i,j-1]
                           eps_p * (theta_b - 2.0) / (theta_b - 1.0),
                           // u[i,j-2]
                           -eps_p * (theta_b - 1.0) / (theta_b - 2.0),
                           // u[i,j+1]
                           eps_jump * n1_I * n2_I * theta_b * dy / dx -
                               (eps_jump * (n1_I * n1_I) + eps_m) * theta_b / (1.0 + theta_b),
                           // u[i-1,j]
                           eps_jump * n1_I * n2_I * (2.0 * theta_b + 1.0) * dy / (2.0 * dx),
                           // u[i+1,j]
                           -eps_jump * n1_I * n2_I * dy / (2.0 * dx),
                           // u[i-1,j+1]
                           -eps_jump * n1_I * n2_I * theta_b * dy / dx};

            rhs_h(row_idx) -= (d / M) * eps_b / theta_b / bot_y;

            rows_coo.insert(rows_coo.end(), 7, row_idx);

            cols_coo.insert(cols_coo.end(), {
                                                index(i, j),
                                                index(i, j - 1),
                                                index(i, j - 2),
                                                index(i, j + 1),
                                                index(i - 1, j),
                                                index(i + 1, j),
                                                index(i - 1, j + 1),
                                            });

            vals_coo.insert(vals_coo.end(),
                            {
                                // u[i,j]
                                (N[0] / M) * eps_b / theta_b / bot_y - (eps_r / theta_r + eps_l / theta_l) / bot_x -
                                    (eps_t / theta_t + eps_b / theta_b) / bot_y,
                                // u[i,j-1]
                                (N[1] / M) * eps_b / theta_b / bot_y,
                                // u[i,j-2]
                                (N[2] / M) * eps_b / theta_b / bot_y,
                                // u[i,j+1]
                                (N[3] / M) * eps_b / theta_b / bot_y + eps_t / theta_t / bot_y,
                                // u[i-1,j]
                                (N[4] / M) * eps_b / theta_b / bot_y + eps_l / theta_l / bot_x,
                                // u[i+1,j]
                                (N[5] / M) * eps_b / theta_b / bot_y + eps_r / theta_r / bot_x,
                                // u_ext at [i-1,j+1]
                                (N[6] / M) * eps_b / theta_b / bot_y,
                            });
        } else {
            Kokkos::printf("coeff_case1(): Invalid direction %d\n", direction);
            Kokkos::abort("Terminated");
        }
    }

    KOKKOS_INLINE_FUNCTION
    int offset(int offset_x, int offset_y) { return (offset_x + 2) * 5 + (offset_y + 2); };

    KOKKOS_INLINE_FUNCTION
    void solve2x2(const double M[2][2], const double rhs[2], double sol[2]) {
        double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
        sol[0]     = (rhs[0] * M[1][1] - rhs[1] * M[0][1]) / det;
        sol[1]     = (-rhs[0] * M[1][0] + rhs[1] * M[0][0]) / det;
    }

    /**
     * Matrix entry for cells having 2 cuts by interface
     */
    void coeff_case2(size_t direction, int i, int j) {
        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double eta          = world.surface(x, y);
        int row_idx         = index(i, j); // laplacian matrix row index
        // use = {} so they're zero-initialized
        double d[2]     = {};
        double M[2][2]  = {};
        double N[2][25] = {};

        if (direction == (Direction::R | Direction::T)) {
            double theta_r = compute_theta(Direction::R, i, j);
            double theta_t = compute_theta(Direction::T, i, j);
            double theta_l = 1.0;
            double theta_b = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + theta_r * dx / 2.0, y);
            double eps_l   = world.permittivity(x - dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + theta_t * dy / 2.0);
            double eps_b   = world.permittivity(x, y - dy / 2.0);

            // normal evaluated at x_R and x_T
            double n1_x = interp(Direction::R, theta_r, i, j, n1);
            double n2_x = interp(Direction::R, theta_r, i, j, n2);
            double n1_y = interp(Direction::T, theta_t, i, j, n1);
            double n2_y = interp(Direction::T, theta_t, i, j, n2);

            // a_tau at x_R and x_T
            double a_tau_x = interp(Direction::R, theta_r, i, j, a_tau);
            double a_tau_y = interp(Direction::T, theta_t, i, j, a_tau);

            // jump conditions at x_R and x_T
            double a_x = interp(Direction::R, theta_r, i, j, a);
            double a_y = interp(Direction::T, theta_t, i, j, a);
            double b_x = interp(Direction::R, theta_r, i, j, b);
            double b_y = interp(Direction::T, theta_t, i, j, b);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx +
                   a_x * eps_p * (3.0 - 2.0 * theta_r) / ((2.0 - theta_r) * (1.0 - theta_r));
            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy +
                   a_y * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = -eps_p * (3.0 - 2.0 * theta_r) / ((1.0 - theta_r) * (2.0 - theta_r)) -
                      eps_m * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0)) -
                      eps_jump * (n2_x * n2_x) * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0));
            M[0][1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1.0));
            M[1][0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1.0));
            M[1][1] = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                      eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                      eps_jump * (n1_y * n1_y) * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            // u[i,j]
            N[0][offset(0, 0)] = -(eps_m + eps_jump * (n2_x * n2_x)) * (theta_r + 1.0) / theta_r -
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_r + theta_t - 1.0) / theta_t);
            // u[i+1,j]
            N[0][offset(1, 0)] = -eps_p * (theta_r - 2.0) / (theta_r - 1.0);
            // u[i+2,j]
            N[0][offset(2, 0)] = eps_p * (theta_r - 1.0) / (theta_r - 2.0);
            // u[i-1,j]
            N[0][offset(-1, 0)] = (eps_m + eps_jump * (n2_x * n2_x)) * theta_r / (theta_r + 1.0) +
                                  eps_jump * n1_x * n2_x * theta_r * (dx / dy);
            // u[i,j-1]
            N[0][offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1.0) + theta_r);
            // u[i-1,j-1]
            N[0][offset(-1, -1)] = -eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            // u[i,j]
            N[1][offset(0, 0)] = -(eps_m + eps_jump * (n1_y * n1_y)) * (theta_t + 1.0) / theta_t -
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_t + theta_r - 1.0) / theta_r);
            // u[i,j+1]
            N[1][offset(0, 1)] = -eps_p * (theta_t - 2.0) / (theta_t - 1.0);
            // u[i,j+2]
            N[1][offset(0, 2)] = eps_p * (theta_t - 1.0) / (theta_t - 2.0);
            // u[i,j-1]
            N[1][offset(0, -1)] = (eps_m + eps_jump * (n1_y * n1_y)) * theta_t / (theta_t + 1.0) +
                                  eps_jump * n1_y * n2_y * theta_t * (dy / dx);
            // u[i-1,j]
            N[1][offset(-1, 0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1.0) + theta_t);
            // u[i-1,j-1]
            N[1][offset(-1, -1)] = -eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            // Solve M * x = d and M * X = N via explicit 2x2 inverse
            double det        = M[0][0] * M[1][1] - M[0][1] * M[1][0];
            double invM[2][2] = {{M[1][1] / det, -M[0][1] / det}, {-M[1][0] / det, M[0][0] / det}};

            double M_inv_d[2] = {invM[0][0] * d[0] + invM[0][1] * d[1], invM[1][0] * d[0] + invM[1][1] * d[1]};

            double M_inv_N[2][25];
            for (int k = 0; k < 25; ++k) {
                M_inv_N[0][k] = invM[0][0] * N[0][k] + invM[0][1] * N[1][k];
                M_inv_N[1][k] = invM[1][0] * N[0][k] + invM[1][1] * N[1][k];
            }

            rhs_h(row_idx) -= M_inv_d[0] * eps_r / theta_r / bot_x + M_inv_d[1] * eps_t / theta_t / bot_y;

            for (int offset_x = -2; offset_x <= 2; ++offset_x) {
                for (int offset_y = -2; offset_y <= 2; ++offset_y) {
                    double value = M_inv_N[0][offset(offset_x, offset_y)] * eps_r / theta_r / bot_x +
                                   M_inv_N[1][offset(offset_x, offset_y)] * eps_t / theta_t / bot_y;

                    if (offset_x == 0 && offset_y == 0) {
                        value +=
                            -(eps_r / theta_r + eps_l / theta_l) / bot_x - (eps_t / theta_t + eps_b / theta_b) / bot_y;
                    } else if (offset_x == -1 && offset_y == 0) {
                        value += eps_l / theta_l / bot_x;
                    } else if (offset_x == 0 && offset_y == -1) {
                        value += eps_b / theta_b / bot_y;
                    }

                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(index(i + offset_x, j + offset_y));
                    vals_coo.push_back(value);
                }
            }
        } else if (direction == (Direction::L | Direction::T)) {
            double theta_l = compute_theta(Direction::L, i, j);
            double theta_t = compute_theta(Direction::T, i, j);
            double theta_r = 1.0;
            double theta_b = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + dx / 2.0, y);
            double eps_l   = world.permittivity(x - theta_l * dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + theta_t * dy / 2.0);
            double eps_b   = world.permittivity(x, y - dy / 2.0);

            // normal evaluated at x_L and x_T
            double n1_x = interp(Direction::L, theta_l, i, j, n1);
            double n2_x = interp(Direction::L, theta_l, i, j, n2);
            double n1_y = interp(Direction::T, theta_t, i, j, n1);
            double n2_y = interp(Direction::T, theta_t, i, j, n2);

            // jump conditions at x_L and x_T
            double a_x = interp(Direction::L, theta_l, i, j, a);
            double a_y = interp(Direction::T, theta_t, i, j, a);
            double b_x = interp(Direction::L, theta_l, i, j, b);
            double b_y = interp(Direction::T, theta_t, i, j, b);

            // a_tau at x_L and x_T
            double a_tau_x = interp(Direction::L, theta_l, i, j, a_tau);
            double a_tau_y = interp(Direction::T, theta_t, i, j, a_tau);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx -
                   a_x * eps_p * (3.0 - 2.0 * theta_l) / ((2.0 - theta_l) * (1.0 - theta_l));
            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy +
                   a_y * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = eps_p * (3.0 - 2.0 * theta_l) / ((1.0 - theta_l) * (2.0 - theta_l)) +
                      eps_m * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0)) +
                      eps_jump * (n2_x * n2_x) * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0));
            M[0][1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1.0));
            M[1][0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1.0));
            M[1][1] = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                      eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                      eps_jump * (n1_y * n1_y) * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            N[0][offset(0, 0)] = (eps_m + eps_jump * (n2_x * n2_x)) * (theta_l + 1.0) / theta_l -
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_l + theta_t - 1.0) / theta_t);
            N[0][offset(-1, 0)] = eps_p * (theta_l - 2.0) / (theta_l - 1.0);
            N[0][offset(-2, 0)] = -eps_p * (theta_l - 1.0) / (theta_l - 2.0);
            N[0][offset(1, 0)]  = -(eps_m + eps_jump * (n2_x * n2_x)) * theta_l / (theta_l + 1.0) +
                                 eps_jump * n1_x * n2_x * theta_l * (dx / dy);
            N[0][offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1.0) + theta_l);
            N[0][offset(1, -1)] = -eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            N[1][offset(0, 0)]  = -(eps_m + eps_jump * (n1_y * n1_y)) * (theta_t + 1.0) / theta_t +
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_t + theta_l - 1.0) / theta_l);
            N[1][offset(0, 1)]  = -eps_p * (theta_t - 2.0) / (theta_t - 1.0);
            N[1][offset(0, 2)]  = eps_p * (theta_t - 1.0) / (theta_t - 2.0);
            N[1][offset(0, -1)] = (eps_m + eps_jump * (n1_y * n1_y)) * theta_t / (theta_t + 1.0) -
                                  eps_jump * n1_y * n2_y * theta_t * (dy / dx);
            N[1][offset(1, 0)]  = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1.0) + theta_t);
            N[1][offset(1, -1)] = eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            // Solve M * x = d and M * X = N via explicit 2x2 inverse
            double det        = M[0][0] * M[1][1] - M[0][1] * M[1][0];
            double invM[2][2] = {{M[1][1] / det, -M[0][1] / det}, {-M[1][0] / det, M[0][0] / det}};

            double M_inv_d[2] = {invM[0][0] * d[0] + invM[0][1] * d[1], invM[1][0] * d[0] + invM[1][1] * d[1]};

            double M_inv_N[2][25];
            for (int k = 0; k < 25; ++k) {
                M_inv_N[0][k] = invM[0][0] * N[0][k] + invM[0][1] * N[1][k];
                M_inv_N[1][k] = invM[1][0] * N[0][k] + invM[1][1] * N[1][k];
            }

            rhs_h(row_idx) -= M_inv_d[0] * eps_l / theta_l / bot_x + M_inv_d[1] * eps_t / theta_t / bot_y;

            for (int offset_x = -2; offset_x <= 2; ++offset_x) {
                for (int offset_y = -2; offset_y <= 2; ++offset_y) {
                    double value = M_inv_N[0][offset(offset_x, offset_y)] * eps_l / theta_l / bot_x +
                                   M_inv_N[1][offset(offset_x, offset_y)] * eps_t / theta_t / bot_y;

                    if (offset_x == 0 && offset_y == 0) {
                        value +=
                            -(eps_r / theta_r + eps_l / theta_l) / bot_x - (eps_t / theta_t + eps_b / theta_b) / bot_y;
                    } else if (offset_x == 1 && offset_y == 0) {
                        value += eps_r / theta_r / bot_x;
                    } else if (offset_x == 0 && offset_y == -1) {
                        value += eps_b / theta_b / bot_y;
                    }

                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(index(i + offset_x, j + offset_y));
                    vals_coo.push_back(value);
                }
            }
        } else if (direction == (Direction::R | Direction::B)) {
            double theta_r = compute_theta(Direction::R, i, j);
            double theta_b = compute_theta(Direction::B, i, j);
            double theta_l = 1.0;
            double theta_t = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + theta_r * dx / 2.0, y);
            double eps_l   = world.permittivity(x - dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + dy / 2.0);
            double eps_b   = world.permittivity(x, y - theta_b * dy / 2.0);

            // normal evaluated at x_R and x_B
            double n1_x = interp(Direction::R, theta_r, i, j, n1);
            double n2_x = interp(Direction::R, theta_r, i, j, n2);
            double n1_y = interp(Direction::B, theta_b, i, j, n1);
            double n2_y = interp(Direction::B, theta_b, i, j, n2);

            // jump conditions at x_R and x_B
            double a_x = interp(Direction::R, theta_r, i, j, a);
            double a_y = interp(Direction::B, theta_b, i, j, a);
            double b_x = interp(Direction::R, theta_r, i, j, b);
            double b_y = interp(Direction::B, theta_b, i, j, b);

            // a_tau at x_R and x_B
            double a_tau_x = interp(Direction::R, theta_r, i, j, a_tau);
            double a_tau_y = interp(Direction::B, theta_b, i, j, a_tau);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx +
                   a_x * eps_p * (3.0 - 2.0 * theta_r) / ((2.0 - theta_r) * (1.0 - theta_r));
            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy -
                   a_y * eps_p * (3.0 - 2.0 * theta_b) / ((2.0 - theta_b) * (1.0 - theta_b));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = -eps_p * (3.0 - 2.0 * theta_r) / ((1.0 - theta_r) * (2.0 - theta_r)) -
                      eps_m * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0)) -
                      eps_jump * (n2_x * n2_x) * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0));
            M[0][1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1.0));
            M[1][0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1.0));
            M[1][1] = eps_p * (3.0 - 2.0 * theta_b) / ((1.0 - theta_b) * (2.0 - theta_b)) +
                      eps_m * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0)) +
                      eps_jump * (n1_y * n1_y) * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0));

            N[0][offset(0, 0)] = -(eps_m + eps_jump * (n2_x * n2_x)) * (theta_r + 1.0) / theta_r +
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_r + theta_b - 1.0) / theta_b);
            N[0][offset(1, 0)]  = -eps_p * (theta_r - 2.0) / (theta_r - 1.0);
            N[0][offset(2, 0)]  = eps_p * (theta_r - 1.0) / (theta_r - 2.0);
            N[0][offset(-1, 0)] = (eps_m + eps_jump * (n2_x * n2_x)) * theta_r / (theta_r + 1.0) -
                                  eps_jump * n1_x * n2_x * theta_r * (dx / dy);
            N[0][offset(0, 1)]  = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1.0) + theta_r);
            N[0][offset(-1, 1)] = eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            N[1][offset(0, 0)]  = (eps_m + eps_jump * (n1_y * n1_y)) * (theta_b + 1.0) / theta_b -
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_b + theta_r - 1.0) / theta_r);
            N[1][offset(0, -1)] = eps_p * (theta_b - 2.0) / (theta_b - 1.0);
            N[1][offset(0, -2)] = -eps_p * (theta_b - 1.0) / (theta_b - 2.0);
            N[1][offset(0, 1)]  = -(eps_m + eps_jump * (n1_y * n1_y)) * theta_b / (theta_b + 1.0) +
                                 eps_jump * n1_y * n2_y * theta_b * (dy / dx);
            N[1][offset(-1, 0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1.0) + theta_b);
            N[1][offset(-1, 1)] = -eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            // Solve M * x = d and M * X = N via explicit 2x2 inverse
            double det        = M[0][0] * M[1][1] - M[0][1] * M[1][0];
            double invM[2][2] = {{M[1][1] / det, -M[0][1] / det}, {-M[1][0] / det, M[0][0] / det}};

            double M_inv_d[2] = {invM[0][0] * d[0] + invM[0][1] * d[1], invM[1][0] * d[0] + invM[1][1] * d[1]};

            double M_inv_N[2][25];
            for (int k = 0; k < 25; ++k) {
                M_inv_N[0][k] = invM[0][0] * N[0][k] + invM[0][1] * N[1][k];
                M_inv_N[1][k] = invM[1][0] * N[0][k] + invM[1][1] * N[1][k];
            }

            rhs_h(row_idx) -= M_inv_d[0] * eps_r / theta_r / bot_x + M_inv_d[1] * eps_b / theta_b / bot_y;

            for (int offset_x = -2; offset_x <= 2; ++offset_x) {
                for (int offset_y = -2; offset_y <= 2; ++offset_y) {
                    double value = M_inv_N[0][offset(offset_x, offset_y)] * eps_r / theta_r / bot_x +
                                   M_inv_N[1][offset(offset_x, offset_y)] * eps_b / theta_b / bot_y;

                    if (offset_x == 0 && offset_y == 0) {
                        value +=
                            -(eps_r / theta_r + eps_l / theta_l) / bot_x - (eps_t / theta_t + eps_b / theta_b) / bot_y;
                    } else if (offset_x == -1 && offset_y == 0) {
                        value += eps_l / theta_l / bot_x;
                    } else if (offset_x == 0 && offset_y == 1) {
                        value += eps_t / theta_t / bot_y;
                    }

                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(index(i + offset_x, j + offset_y));
                    vals_coo.push_back(value);
                }
            }
        } else if (direction == (Direction::L | Direction::B)) {
            double theta_l = compute_theta(Direction::L, i, j);
            double theta_b = compute_theta(Direction::B, i, j);
            double theta_r = 1.0;
            double theta_t = 1.0;

            double bot_x   = (theta_r + theta_l) / 2.0 * (dx * dx);
            double bot_y   = (theta_t + theta_b) / 2.0 * (dy * dy);

            double eps_r   = world.permittivity(x + dx / 2.0, y);
            double eps_l   = world.permittivity(x - theta_l * dx / 2.0, y);
            double eps_t   = world.permittivity(x, y + dy / 2.0);
            double eps_b   = world.permittivity(x, y - theta_b * dy / 2.0);

            // normal evaluated at x_L and x_B
            double n1_x = interp(Direction::L, theta_l, i, j, n1);
            double n2_x = interp(Direction::L, theta_l, i, j, n2);
            double n1_y = interp(Direction::B, theta_b, i, j, n1);
            double n2_y = interp(Direction::B, theta_b, i, j, n2);

            // jump conditions at x_L and x_B
            double a_x = interp(Direction::L, theta_l, i, j, a);
            double a_y = interp(Direction::B, theta_b, i, j, a);
            double b_x = interp(Direction::L, theta_l, i, j, b);
            double b_y = interp(Direction::B, theta_b, i, j, b);

            // a_tau at x_L and x_B
            double a_tau_x = interp(Direction::L, theta_l, i, j, a_tau);
            double a_tau_y = interp(Direction::B, theta_b, i, j, a_tau);

            double _eps_p, _eps_m;
            double eps_jump;
            double eps_p, eps_m;

            if (eta > 0.0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx -
                   a_x * eps_p * (3.0 - 2.0 * theta_l) / ((2.0 - theta_l) * (1.0 - theta_l));
            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy -
                   a_y * eps_p * (3.0 - 2.0 * theta_b) / ((2.0 - theta_b) * (1.0 - theta_b));

            if (eta > 0.0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = eps_p * (3.0 - 2.0 * theta_l) / ((1.0 - theta_l) * (2.0 - theta_l)) +
                      eps_m * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0)) +
                      eps_jump * (n2_x * n2_x) * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0));
            M[0][1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1.0));
            M[1][0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1.0));
            M[1][1] = eps_p * (3.0 - 2.0 * theta_b) / ((1.0 - theta_b) * (2.0 - theta_b)) +
                      eps_m * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0)) +
                      eps_jump * (n1_y * n1_y) * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0));

            N[0][offset(0, 0)] = (eps_m + eps_jump * (n2_x * n2_x)) * (theta_l + 1.0) / theta_l +
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_l + theta_b - 1.0) / theta_b);
            N[0][offset(-1, 0)] = eps_p * (theta_l - 2.0) / (theta_l - 1.0);
            N[0][offset(-2, 0)] = -eps_p * (theta_l - 1.0) / (theta_l - 2.0);
            N[0][offset(1, 0)]  = -(eps_m + eps_jump * (n2_x * n2_x)) * theta_l / (theta_l + 1.0) -
                                 eps_jump * n1_x * n2_x * theta_l * (dx / dy);
            N[0][offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1.0) + theta_l);
            N[0][offset(1, 1)] = eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            N[1][offset(0, 0)] = (eps_m + eps_jump * (n1_y * n1_y)) * (theta_b + 1.0) / theta_b +
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_b + theta_l - 1.0) / theta_l);
            N[1][offset(0, -1)] = eps_p * (theta_b - 2.0) / (theta_b - 1.0);
            N[1][offset(0, -2)] = -eps_p * (theta_b - 1.0) / (theta_b - 2.0);
            N[1][offset(0, 1)]  = -(eps_m + eps_jump * (n1_y * n1_y)) * theta_b / (theta_b + 1.0) -
                                 eps_jump * n1_y * n2_y * theta_b * (dy / dx);
            N[1][offset(1, 0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1.0) + theta_b);
            N[1][offset(1, 1)] = eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            // Solve M * x = d and M * X = N via explicit 2x2 inverse
            double det        = M[0][0] * M[1][1] - M[0][1] * M[1][0];
            double invM[2][2] = {{M[1][1] / det, -M[0][1] / det}, {-M[1][0] / det, M[0][0] / det}};

            double M_inv_d[2] = {invM[0][0] * d[0] + invM[0][1] * d[1], invM[1][0] * d[0] + invM[1][1] * d[1]};

            double M_inv_N[2][25];
            for (int k = 0; k < 25; ++k) {
                M_inv_N[0][k] = invM[0][0] * N[0][k] + invM[0][1] * N[1][k];
                M_inv_N[1][k] = invM[1][0] * N[0][k] + invM[1][1] * N[1][k];
            }

            rhs_h(row_idx) -= M_inv_d[0] * eps_l / theta_l / bot_x + M_inv_d[1] * eps_b / theta_b / bot_y;

            for (int offset_x = -2; offset_x <= 2; ++offset_x) {
                for (int offset_y = -2; offset_y <= 2; ++offset_y) {
                    double value = M_inv_N[0][offset(offset_x, offset_y)] * eps_l / theta_l / bot_x +
                                   M_inv_N[1][offset(offset_x, offset_y)] * eps_b / theta_b / bot_y;

                    if (offset_x == 0 && offset_y == 0) {
                        value +=
                            -(eps_r / theta_r + eps_l / theta_l) / bot_x - (eps_t / theta_t + eps_b / theta_b) / bot_y;
                    } else if (offset_x == 1 && offset_y == 0) {
                        value += eps_r / theta_r / bot_x;
                    } else if (offset_x == 0 && offset_y == 1) {
                        value += eps_t / theta_t / bot_y;
                    }

                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(index(i + offset_x, j + offset_y));
                    vals_coo.push_back(value);
                }
            }
        }
    }

    InterfaceValue interface_value_case0(int i, int j, const auto& u) {
        return {
            u(i - 1, j), u(i + 1, j), u(i, j - 1), u(i, j + 1), 1.0, 1.0, 1.0, 1.0,
        };
    }

    InterfaceValue interface_value_case1(size_t direction, int i, int j, const auto& u) {
        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double eta          = world.surface(x, y);

        double theta        = compute_theta(direction, i, j);

        double a_tau_I      = interp(direction, theta, i, j, a_tau);
        double a_I          = interp(direction, theta, i, j, a);
        double b_I          = interp(direction, theta, i, j, b);
        double n1_I         = interp(direction, theta, i, j, n1);
        double n2_I         = interp(direction, theta, i, j, n2);

        double theta_l, theta_r, theta_b, theta_t;

        if (direction == Direction::R) {

            theta_l = 1.0;
            theta_r = theta;
            theta_b = 1.0;
            theta_t = 1.0;

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = -a_tau_I * eps_p * n2_I * dx + b_I * n1_I * dx +
                       a_I * eps_p * (3.0 - 2.0 * theta_r) / ((2.0 - theta_r) * (1.0 - theta_r));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = -eps_p * (3.0 - 2.0 * theta_r) / ((1.0 - theta_r) * (2.0 - theta_r)) -
                       eps_m * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0)) -
                       eps_jump * n2_I * n2_I * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0));

            double N[7]     = {-eps_jump * n1_I * n2_I * theta_r * dx / dy -
                                   (eps_jump * n2_I * n2_I + eps_m) * (1.0 + theta_r) / theta_r,

                               -eps_p * (theta_r - 2.0) / (theta_r - 1.0),

                               eps_p * (theta_r - 1.0) / (theta_r - 2.0),

                               eps_jump * n1_I * n2_I * theta_r * dx / dy +
                                   (eps_jump * n2_I * n2_I + eps_m) * theta_r / (1.0 + theta_r),

                               eps_jump * n1_I * n2_I * (2.0 * theta_r + 1.0) * dx / (2.0 * dy),

                               -eps_jump * n1_I * n2_I * dx / (2.0 * dy),

                               -eps_jump * n1_I * n2_I * theta_r * dx / dy};

            double u_arr[7] = {u(i, j),     u(i + 1, j), u(i + 2, j),    u(i - 1, j),
                               u(i, j - 1), u(i, j + 1), u(i - 1, j - 1)};

            double dot      = 0.0;
            for (int k = 0; k < 7; k++) {
                dot += N[k] * u_arr[k];
            }

            double u_I = (dot + d) / M;

            return {u(i - 1, j), u_I, u(i, j - 1), u(i, j + 1), theta_l, theta_r, theta_b, theta_t};
        } else if (direction == Direction::T) {

            theta_l = 1.0;
            theta_r = 1.0;
            theta_b = 1.0;
            theta_t = theta;

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = a_tau_I * eps_p * n1_I * dy + b_I * n2_I * dy +
                       a_I * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                       eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                       eps_jump * n1_I * n1_I * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            double N[7]     = {-eps_jump * n1_I * n2_I * theta_t * dy / dx -
                                   (eps_jump * n1_I * n1_I + eps_m) * (1.0 + theta_t) / theta_t,

                               -eps_p * (theta_t - 2.0) / (theta_t - 1.0),

                               eps_p * (theta_t - 1.0) / (theta_t - 2.0),

                               eps_jump * n1_I * n2_I * theta_t * dy / dx +
                                   (eps_jump * n1_I * n1_I + eps_m) * theta_t / (1.0 + theta_t),

                               eps_jump * n1_I * n2_I * (2.0 * theta_t + 1.0) * dy / (2.0 * dx),

                               -eps_jump * n1_I * n2_I * dy / (2.0 * dx),

                               -eps_jump * n1_I * n2_I * theta_t * dy / dx};

            double u_arr[7] = {u(i, j),     u(i, j + 1), u(i, j + 2),    u(i, j - 1),
                               u(i - 1, j), u(i + 1, j), u(i - 1, j - 1)};

            double dot      = 0.0;
            for (int k = 0; k < 7; k++) {
                dot += N[k] * u_arr[k];
            }

            double u_I = (dot + d) / M;

            return {u(i - 1, j), u(i + 1, j), u(i, j - 1), u_I, theta_l, theta_r, theta_b, theta_t};
        } else if (direction == Direction::L) {

            theta_l = theta;
            theta_r = 1.0;
            theta_b = 1.0;
            theta_t = 1.0;

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = -a_tau_I * eps_p * n2_I * dx + b_I * n1_I * dx -
                       a_I * eps_p * (3.0 - 2.0 * theta_l) / ((2.0 - theta_l) * (1.0 - theta_l));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = eps_p * (3.0 - 2.0 * theta_l) / ((1.0 - theta_l) * (2.0 - theta_l)) +
                       eps_m * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0)) +
                       eps_jump * n2_I * n2_I * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0));

            double N[7]     = {-eps_jump * n1_I * n2_I * theta_l * dx / dy +
                                   (eps_jump * n2_I * n2_I + eps_m) * (1.0 + theta_l) / theta_l,

                               eps_p * (theta_l - 2.0) / (theta_l - 1.0),

                               -eps_p * (theta_l - 1.0) / (theta_l - 2.0),

                               eps_jump * n1_I * n2_I * theta_l * dx / dy -
                                   (eps_jump * n2_I * n2_I + eps_m) * theta_l / (1.0 + theta_l),

                               eps_jump * n1_I * n2_I * (2.0 * theta_l + 1.0) * dx / (2.0 * dy),

                               -eps_jump * n1_I * n2_I * dx / (2.0 * dy),

                               -eps_jump * n1_I * n2_I * theta_l * dx / dy};

            double u_arr[7] = {u(i, j),     u(i - 1, j), u(i - 2, j),    u(i + 1, j),
                               u(i, j - 1), u(i, j + 1), u(i + 1, j - 1)};

            double dot      = 0.0;
            for (int k = 0; k < 7; k++) {
                dot += N[k] * u_arr[k];
            }

            double u_I = (dot + d) / M;

            return {u_I, u(i + 1, j), u(i, j - 1), u(i, j + 1), theta_l, theta_r, theta_b, theta_t};
        } else if (direction == Direction::B) {

            theta_l = 1.0;
            theta_r = 1.0;
            theta_b = theta;
            theta_t = 1.0;

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            double d = a_tau_I * eps_p * n1_I * dy + b_I * n2_I * dy -
                       a_I * eps_p * (3.0 - 2.0 * theta_b) / ((2.0 - theta_b) * (1.0 - theta_b));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            double M = eps_p * (3.0 - 2.0 * theta_b) / ((1.0 - theta_b) * (2.0 - theta_b)) +
                       eps_m * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0)) +
                       eps_jump * n1_I * n1_I * (2.0 * theta_b + 1.0) / (theta_b * (theta_b + 1.0));

            double N[7]     = {-eps_jump * n1_I * n2_I * theta_b * dy / dx +
                                   (eps_jump * n1_I * n1_I + eps_m) * (1.0 + theta_b) / theta_b,

                               eps_p * (theta_b - 2.0) / (theta_b - 1.0),

                               -eps_p * (theta_b - 1.0) / (theta_b - 2.0),

                               eps_jump * n1_I * n2_I * theta_b * dy / dx -
                                   (eps_jump * n1_I * n1_I + eps_m) * theta_b / (1.0 + theta_b),

                               eps_jump * n1_I * n2_I * (2.0 * theta_b + 1.0) * dy / (2.0 * dx),

                               -eps_jump * n1_I * n2_I * dy / (2.0 * dx),

                               -eps_jump * n1_I * n2_I * theta_b * dy / dx};

            double u_arr[7] = {u(i, j),     u(i, j - 1), u(i, j - 2),    u(i, j + 1),
                               u(i - 1, j), u(i + 1, j), u(i - 1, j + 1)};

            double dot      = 0.0;
            for (int k = 0; k < 7; k++) {
                dot += N[k] * u_arr[k];
            }

            double u_I = (dot + d) / M;

            return {u(i - 1, j), u(i + 1, j), u_I, u(i, j + 1), theta_l, theta_r, theta_b, theta_t};
        }

        else {
            Kokkos::printf("interface_value_case1():Invalid direction %d", direction);
            Kokkos::abort("Exit");
        }
    }

    InterfaceValue interface_value_case2(size_t direction, int i, int j, const auto& u) {
        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double eta          = world.surface(x, y);

        // use = {} so they're zero-initialized
        double d[2]      = {};
        double M[2][2]   = {};
        double N[2][25]  = {};
        double u_arr[25] = {};
        for (int ox = -2; ox <= 2; ++ox) {
            for (int oy = -2; oy <= 2; ++oy) {
                u_arr[offset(ox, oy)] = u(i + ox, j + oy);
            }
        }

        if (direction == (Direction::R | Direction::T)) {

            double theta_r = compute_theta(Direction::R, i, j);
            double theta_t = compute_theta(Direction::T, i, j);
            double theta_l = 1.0;
            double theta_b = 1.0;

            // normal evaluated at x_R and x_T
            double n1_x = interp(Direction::R, theta_r, i, j, n1);
            double n2_x = interp(Direction::R, theta_r, i, j, n2);
            double n1_y = interp(Direction::T, theta_t, i, j, n1);
            double n2_y = interp(Direction::T, theta_t, i, j, n2);

            // a_tau at x_R and x_T
            double a_tau_x = interp(Direction::R, theta_r, i, j, a_tau);
            double a_tau_y = interp(Direction::T, theta_t, i, j, a_tau);

            // jump conditions at x_R and x_T
            double a_x = interp(Direction::R, theta_r, i, j, a);
            double a_y = interp(Direction::T, theta_t, i, j, a);
            double b_x = interp(Direction::R, theta_r, i, j, b);
            double b_y = interp(Direction::T, theta_t, i, j, b);

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx +
                   a_x * eps_p * (3.0 - 2.0 * theta_r) / ((2.0 - theta_r) * (1.0 - theta_r));

            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy +
                   a_y * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = -eps_p * (3.0 - 2.0 * theta_r) / ((1.0 - theta_r) * (2.0 - theta_r)) -
                      eps_m * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0)) -
                      eps_jump * n2_x * n2_x * (2.0 * theta_r + 1.0) / (theta_r * (theta_r + 1.0));

            M[0][1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1.0));

            M[1][0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1.0));

            M[1][1] = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                      eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                      eps_jump * n1_y * n1_y * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            // Row 0
            N[0][offset(0, 0)] = -(eps_m + eps_jump * n2_x * n2_x) * (theta_r + 1.0) / theta_r -
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_r + theta_t - 1.0) / theta_t);

            N[0][offset(1, 0)]  = -eps_p * (theta_r - 2.0) / (theta_r - 1.0);

            N[0][offset(2, 0)]  = eps_p * (theta_r - 1.0) / (theta_r - 2.0);

            N[0][offset(-1, 0)] = (eps_m + eps_jump * n2_x * n2_x) * theta_r / (theta_r + 1.0) +
                                  eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            N[0][offset(0, -1)]  = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1.0) + theta_r);

            N[0][offset(-1, -1)] = -eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            // Row 1
            N[1][offset(0, 0)] = -(eps_m + eps_jump * n1_y * n1_y) * (theta_t + 1.0) / theta_t -
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_t + theta_r - 1.0) / theta_r);

            N[1][offset(0, 1)]  = -eps_p * (theta_t - 2.0) / (theta_t - 1.0);

            N[1][offset(0, 2)]  = eps_p * (theta_t - 1.0) / (theta_t - 2.0);

            N[1][offset(0, -1)] = (eps_m + eps_jump * n1_y * n1_y) * theta_t / (theta_t + 1.0) +
                                  eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            N[1][offset(-1, 0)]  = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1.0) + theta_t);

            N[1][offset(-1, -1)] = -eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            // --- Compute rhs = N*u_arr + d ---
            double _rhs[2] = {d[0], d[1]};
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 25; ++c) {
                    _rhs[r] += N[r][c] * u_arr[c];
                }
            }

            double det  = M[0][0] * M[1][1] - M[0][1] * M[1][0];

            double u_I0 = (_rhs[0] * M[1][1] - _rhs[1] * M[0][1]) / det;

            double u_I1 = (M[0][0] * _rhs[1] - M[1][0] * _rhs[0]) / det;
            return {
                u(i - 1, j), u_I0, u(i, j - 1), u_I1, theta_l, theta_r, theta_b, theta_t,
            };
        } else if (direction == (Direction::L | Direction::T)) {
            double theta_l = compute_theta(Direction::L, i, j);
            double theta_t = compute_theta(Direction::T, i, j);
            double theta_r = 1.0;
            double theta_b = 1.0;

            // normal evaluated at x_L and x_T
            double n1_x = interp(Direction::L, theta_l, i, j, n1);
            double n2_x = interp(Direction::L, theta_l, i, j, n2);
            double n1_y = interp(Direction::T, theta_t, i, j, n1);
            double n2_y = interp(Direction::T, theta_t, i, j, n2);

            // jump conditions at x_L and x_T
            double a_x = interp(Direction::L, theta_l, i, j, a);
            double a_y = interp(Direction::T, theta_t, i, j, a);
            double b_x = interp(Direction::L, theta_l, i, j, b);
            double b_y = interp(Direction::T, theta_t, i, j, b);

            // a_tau at x_L and x_T
            double a_tau_x = interp(Direction::L, theta_l, i, j, a_tau);
            double a_tau_y = interp(Direction::T, theta_t, i, j, a_tau);

            double _eps_p, _eps_m, eps_jump, eps_p, eps_m;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y + dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y + dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = -a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx -
                   a_x * eps_p * (3.0 - 2.0 * theta_l) / ((2.0 - theta_l) * (1.0 - theta_l));

            d[1] = a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy +
                   a_y * eps_p * (3.0 - 2.0 * theta_t) / ((2.0 - theta_t) * (1.0 - theta_t));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = eps_p * (3.0 - 2.0 * theta_l) / ((1.0 - theta_l) * (2.0 - theta_l)) +
                      eps_m * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0)) +
                      eps_jump * n2_x * n2_x * (2.0 * theta_l + 1.0) / (theta_l * (theta_l + 1.0));

            M[0][1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1.0));

            M[1][0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1.0));

            M[1][1] = -eps_p * (3.0 - 2.0 * theta_t) / ((1.0 - theta_t) * (2.0 - theta_t)) -
                      eps_m * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0)) -
                      eps_jump * n1_y * n1_y * (2.0 * theta_t + 1.0) / (theta_t * (theta_t + 1.0));

            // ---------- Row 0 ----------
            N[0][offset(0, 0)] = (eps_m + eps_jump * n2_x * n2_x) * (theta_l + 1.0) / theta_l -
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_l + theta_t - 1.0) / theta_t);

            N[0][offset(-1, 0)] = eps_p * (theta_l - 2.0) / (theta_l - 1.0);

            N[0][offset(-2, 0)] = -eps_p * (theta_l - 1.0) / (theta_l - 2.0);

            N[0][offset(1, 0)]  = -(eps_m + eps_jump * n2_x * n2_x) * theta_l / (theta_l + 1.0) +
                                 eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            N[0][offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1.0) + theta_l);

            N[0][offset(1, -1)] = -eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            // ---------- Row 1 ----------
            N[1][offset(0, 0)] = -(eps_m + eps_jump * n1_y * n1_y) * (theta_t + 1.0) / theta_t +
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_t + theta_l - 1.0) / theta_l);

            N[1][offset(0, 1)]  = -eps_p * (theta_t - 2.0) / (theta_t - 1.0);

            N[1][offset(0, 2)]  = eps_p * (theta_t - 1.0) / (theta_t - 2.0);

            N[1][offset(0, -1)] = (eps_m + eps_jump * n1_y * n1_y) * theta_t / (theta_t + 1.0) -
                                  eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            N[1][offset(1, 0)]  = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1.0) + theta_t);

            N[1][offset(1, -1)] = eps_jump * n1_y * n2_y * theta_t * (dy / dx);

            // rhs = N*u_arr + d
            double _rhs[2] = {d[0], d[1]};
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 25; ++c) {
                    _rhs[r] += N[r][c] * u_arr[c];
                }
            }

            // solve 2x2 system
            double det  = M[0][0] * M[1][1] - M[0][1] * M[1][0];

            double u_I0 = (_rhs[0] * M[1][1] - _rhs[1] * M[0][1]) / det;

            double u_I1 = (M[0][0] * _rhs[1] - M[1][0] * _rhs[0]) / det;

            return {
                u_I0, u(i + 1, j), u(i, j - 1), u_I1, theta_l, theta_r, theta_b, theta_t,
            };
        } else if (direction == (Direction::R | Direction::B)) {
            double theta_r = compute_theta(Direction::R, i, j);
            double theta_b = compute_theta(Direction::B, i, j);
            double theta_l = 1.0;
            double theta_t = 1.0;

            // normal evaluated at x_R and x_B
            double n1_x = interp(Direction::R, theta_r, i, j, n1);
            double n2_x = interp(Direction::R, theta_r, i, j, n2);
            double n1_y = interp(Direction::B, theta_b, i, j, n1);
            double n2_y = interp(Direction::B, theta_b, i, j, n2);

            // jump conditions at x_R and x_B
            double a_x = interp(Direction::R, theta_r, i, j, a);
            double a_y = interp(Direction::B, theta_b, i, j, a);
            double b_x = interp(Direction::R, theta_r, i, j, b);
            double b_y = interp(Direction::B, theta_b, i, j, b);

            // a_tau at x_R and x_B
            double a_tau_x = interp(Direction::R, theta_r, i, j, a_tau);
            double a_tau_y = interp(Direction::B, theta_b, i, j, a_tau);

            double _eps_p, _eps_m, eps_p, eps_m, eps_jump;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x + dx, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x + dx, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = (-a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx +
                    a_x * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r)));

            d[1] = (a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy -
                    a_y * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b)));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r)) -
                      eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1)) -
                      eps_jump * n2_x * n2_x * (2 * theta_r + 1) / (theta_r * (theta_r + 1));

            M[0][1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1));

            M[1][0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1));

            M[1][1] = eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b)) +
                      eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1)) +
                      eps_jump * n1_y * n1_y * (2 * theta_b + 1) / (theta_b * (theta_b + 1));

            // ---- Fill N matrix ----
            N[0][offset(0, 0)] = -(eps_m + eps_jump * n2_x * n2_x) * (theta_r + 1) / theta_r +
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_r + theta_b - 1) / theta_b);

            N[0][offset(1, 0)]  = -eps_p * (theta_r - 2) / (theta_r - 1);

            N[0][offset(2, 0)]  = eps_p * (theta_r - 1) / (theta_r - 2);

            N[0][offset(-1, 0)] = (eps_m + eps_jump * n2_x * n2_x) * theta_r / (theta_r + 1) -
                                  eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            N[0][offset(0, 1)]  = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_r);

            N[0][offset(-1, 1)] = eps_jump * n1_x * n2_x * theta_r * (dx / dy);

            N[1][offset(0, 0)]  = (eps_m + eps_jump * n1_y * n1_y) * (theta_b + 1) / theta_b -
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_b + theta_r - 1) / theta_r);

            N[1][offset(0, -1)] = eps_p * (theta_b - 2) / (theta_b - 1);

            N[1][offset(0, -2)] = -eps_p * (theta_b - 1) / (theta_b - 2);

            N[1][offset(0, 1)]  = -(eps_m + eps_jump * n1_y * n1_y) * theta_b / (theta_b + 1) +
                                 eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            N[1][offset(-1, 0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1) + theta_b);

            N[1][offset(-1, 1)] = -eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            // ---- Form RHS:  _rhs = N * u_arr + d ----
            double _rhs[2] = {d[0], d[1]};

            for (int k = 0; k < 25; ++k) {
                _rhs[0] += N[0][k] * u_arr[k];
                _rhs[1] += N[1][k] * u_arr[k];
            }

            // ---- Solve 2x2 system M * u_I = _rhs ----
            double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];

            double uI0 = (_rhs[0] * M[1][1] - _rhs[1] * M[0][1]) / det;

            double uI1 = (M[0][0] * _rhs[1] - M[1][0] * _rhs[0]) / det;

            return {u(i - 1, j), uI0, uI1, u(i, j + 1), theta_l, theta_r, theta_b, theta_t};
        } else if (direction == (Direction::L | Direction::B)) {

            double theta_l = compute_theta(Direction::L, i, j);
            double theta_b = compute_theta(Direction::B, i, j);
            double theta_r = 1.0;
            double theta_t = 1.0;

            // normal evaluated at x_L and x_B
            double n1_x = interp(Direction::L, theta_l, i, j, n1);
            double n2_x = interp(Direction::L, theta_l, i, j, n2);
            double n1_y = interp(Direction::B, theta_b, i, j, n1);
            double n2_y = interp(Direction::B, theta_b, i, j, n2);

            // jump conditions at x_L and x_B
            double a_x = interp(Direction::L, theta_l, i, j, a);
            double a_y = interp(Direction::B, theta_b, i, j, a);
            double b_x = interp(Direction::L, theta_l, i, j, b);
            double b_y = interp(Direction::B, theta_b, i, j, b);

            // a_tau at x_L and x_B
            double a_tau_x = interp(Direction::L, theta_l, i, j, a_tau);
            double a_tau_y = interp(Direction::B, theta_b, i, j, a_tau);

            double _eps_p, _eps_m, eps_p, eps_m, eps_jump;

            if (eta > 0) {
                _eps_p   = world.permittivity(x, y);
                _eps_m   = world.permittivity(x - dx, y - dy);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_m;
                eps_m    = _eps_p;
            } else {
                _eps_p   = world.permittivity(x - dx, y - dy);
                _eps_m   = world.permittivity(x, y);
                eps_jump = _eps_p - _eps_m;
                eps_p    = _eps_p;
                eps_m    = _eps_m;
            }

            d[0] = (-a_tau_x * eps_p * n2_x * dx + b_x * n1_x * dx -
                    a_x * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l)));

            d[1] = (a_tau_y * eps_p * n1_y * dy + b_y * n2_y * dy -
                    a_y * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b)));

            if (eta > 0) {
                eps_p = -_eps_m;
                eps_m = -_eps_p;
            }

            M[0][0] = eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l)) +
                      eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1)) +
                      eps_jump * n2_x * n2_x * (2 * theta_l + 1) / (theta_l * (theta_l + 1));

            M[0][1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1));

            M[1][0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1));

            M[1][1] = eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b)) +
                      eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1)) +
                      eps_jump * n1_y * n1_y * (2 * theta_b + 1) / (theta_b * (theta_b + 1));

            // ---- N matrix entries ----
            N[0][offset(0, 0)] = (eps_m + eps_jump * n2_x * n2_x) * (theta_l + 1) / theta_l +
                                 (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_l + theta_b - 1) / theta_b);

            N[0][offset(-1, 0)] = eps_p * (theta_l - 2) / (theta_l - 1);

            N[0][offset(-2, 0)] = -eps_p * (theta_l - 1) / (theta_l - 2);

            N[0][offset(1, 0)]  = -(eps_m + eps_jump * n2_x * n2_x) * theta_l / (theta_l + 1) -
                                 eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            N[0][offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_l);

            N[0][offset(1, 1)] = eps_jump * n1_x * n2_x * theta_l * (dx / dy);

            N[1][offset(0, 0)] = (eps_m + eps_jump * n1_y * n1_y) * (theta_b + 1) / theta_b +
                                 (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_b + theta_l - 1) / theta_l);

            N[1][offset(0, -1)] = eps_p * (theta_b - 2) / (theta_b - 1);

            N[1][offset(0, -2)] = -eps_p * (theta_b - 1) / (theta_b - 2);

            N[1][offset(0, 1)]  = -(eps_m + eps_jump * n1_y * n1_y) * theta_b / (theta_b + 1) -
                                 eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            N[1][offset(1, 0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1) + theta_b);

            N[1][offset(1, 1)] = eps_jump * n1_y * n2_y * theta_b * (dy / dx);

            // ---- Build RHS:  _rhs = N*u_arr + d ----
            double _rhs[2] = {d[0], d[1]};

            for (int k = 0; k < 25; ++k) {
                _rhs[0] += N[0][k] * u_arr[k];
                _rhs[1] += N[1][k] * u_arr[k];
            }

            // ---- Solve 2x2 system M * u_I = _rhs ----
            double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];

            double uI0 = (_rhs[0] * M[1][1] - _rhs[1] * M[0][1]) / det;

            double uI1 = (M[0][0] * _rhs[1] - M[1][0] * _rhs[0]) / det;

            return {uI0, u(i + 1, j), uI1, u(i, j + 1), theta_l, theta_r, theta_b, theta_t};
        } else {
            Kokkos::printf("interface_value_case2(): Invalid direction", direction);
            Kokkos::abort("Exit");
        }
    }

    /**
     * Convert sparse matrix coo format to crs format
     */
    void coo2crs() {
        int nrows = nx * ny;
        int ncols = nx * ny;
        int nnz   = vals_coo.size();

        // make rowmap (counts -> prefix-sum)
        std::vector<int> rowmap(nrows + 1, 0);
        for (int k = 0; k < rows_coo.size(); ++k) {
            rowmap[rows_coo[k] + 1] += 1; // increment bucket for row
        }
        for (int i = 0; i < nrows; ++i) {
            rowmap[i + 1] += rowmap[i]; // prefix sum
        }

        // scatter coo into crs arrays (stable within row)
        std::vector<int> cur = rowmap; // current write pointer per row
        std::vector<int> cols_crs(nnz);
        std::vector<double> vals_crs(nnz);
        for (size_t k = 0; k < rows_coo.size(); ++k) {
            int r          = rows_coo[k];
            int dest       = cur[r]++;
            cols_crs[dest] = cols_coo[k];
            vals_crs[dest] = vals_coo[k];
        }

        // constructor will deep-copy to device
        A = CRS("A", nrows, ncols, nnz, vals_crs.data(), rowmap.data(), cols_crs.data());
        // make sure A's rows are sorted, so we can build preconditioner later
        KokkosSparse::sort_crs_matrix(A);
    }

    /**
     * Construct all necessary fiels
     * normal field, jump condition fields
     */
    void construct_fields() {
        u     = Kokkos::View<double*>("u", nx * ny);
        rhs   = Kokkos::View<double*>("rhs", nx * ny);
        rhs_h = Kokkos::View<double*, Kokkos::HostSpace>("rhs_h", nx * ny);

        n1    = Kokkos::View<double**, Kokkos::HostSpace>("n1", nx, ny);
        n2    = Kokkos::View<double**, Kokkos::HostSpace>("n2", nx, ny);
        a     = Kokkos::View<double**, Kokkos::HostSpace>("a", nx, ny);
        b     = Kokkos::View<double**, Kokkos::HostSpace>("b", nx, ny);
        a_tau = Kokkos::View<double**, Kokkos::HostSpace>("a_tau", nx, ny);
        // pre-compute fields
        using Kokkos::pow;
        using Kokkos::sqrt;
        int ngc = world.grid.ngc;

        for (int i = ngc; i < nx - ngc; ++i) {
            for (int j = ngc; j < ny - ngc; ++j) {
                auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});

                double dx_eta       = (-world.surface(x + 2 * dx, y) + 8 * world.surface(x + dx, y) -
                                 8 * world.surface(x - dx, y) + world.surface(x - 2 * dx, y)) /
                                (12 * dx);
                double dy_eta = (-world.surface(x, y + 2 * dy) + 8 * world.surface(x, y + dy) -
                                 8 * world.surface(x, y - dy) + world.surface(x, y - 2 * dy)) /
                                (12 * dy);
                double norm = sqrt(pow(dx_eta, 2) + pow(dy_eta, 2));

                // normal field
                if (isclose(norm, 0.0)) {
                    n1(i, j) = 0.0;
                    n2(i, j) = 0.0;
                } else {
                    n1(i, j) = dx_eta / norm;
                    n2(i, j) = dy_eta / norm;
                }

                // jump conditions
                a(i, j) = world.poisson_jump_condition_a(x, y);
                b(i, j) = world.poisson_jump_condition_b(x, y);
            }
        }

        // tangentual derivative of a
        for (int i = ngc; i < nx - ngc; ++i) {
            for (int j = ngc; j < ny - ngc; ++j) {
                double dx_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dx);
                double dy_a = (-a(i, j + 2) + 8 * a(i, j + 1) - 8 * a(i, j - 1) + a(i, j - 2)) / (12 * dy);
                a_tau(i, j) = -dx_a * n2(i, j) + dy_a * n1(i, j);
            }
        }
    }

    /**
     * Construct the Laplacian matrix nabla^2
     */
    void construct_matrix() {
        int ngc    = world.grid.ngc;
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int row_idx          = index(i, j);
                PoissonBCPair bc_map = world.poisson_bc_map(i, j);

                if (bc_map.type == PoissonBCType::Dirichlet) {
                    vals_coo.push_back(1.0);
                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(row_idx);
                    rhs_h(row_idx) = bc_map.val;
                } else if (bc_map.type == PoissonBCType::Neumann) {
                    if (i < ngc) {
                        vals_coo.insert(vals_coo.end(), {-1.0, 1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i + 1, j)});
                        rhs_h(row_idx) = bc_map.val;
                    } else if (i >= nx - ngc) {
                        vals_coo.insert(vals_coo.end(), {-1.0, 1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i - 1, j)});
                        rhs_h(row_idx) = -bc_map.val;
                    } else if (j < ngc) {
                        vals_coo.insert(vals_coo.end(), {-1.0, 1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i, j + 1)});
                        rhs_h(row_idx) = bc_map.val;
                    } else if (j >= ny - ngc) {
                        vals_coo.insert(vals_coo.end(), {-1.0, 1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i, j - 1)});
                        rhs_h(row_idx) = -bc_map.val;
                    } else {
                        Kokkos::printf("Neumann BC can only be applied at ghost cells");
                        Kokkos::abort("Terminated");
                    }
                } else if (bc_map.type == PoissonBCType::Periodic) {
                    if (i < ngc) {
                        vals_coo.insert(vals_coo.end(), {1.0, -1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(nx - 2 * ngc + i, j)});
                        rhs_h(row_idx) = 0.0;
                    } else if (i >= nx - ngc) {
                        vals_coo.insert(vals_coo.end(), {1.0, -1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i - nx + ngc, j)});
                        rhs_h(row_idx) = 0.0;
                    } else if (j < ngc) {
                        vals_coo.insert(vals_coo.end(), {1.0, -1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i, ny - 2 * ngc + j)});
                        rhs_h(row_idx) = 0.0;
                    } else if (j >= ny - ngc) {
                        vals_coo.insert(vals_coo.end(), {1.0, -1.0});
                        rows_coo.insert(rows_coo.end(), {row_idx, row_idx});
                        cols_coo.insert(cols_coo.end(), {row_idx, index(i, j - nx + ngc)});
                        rhs_h(row_idx) = 0.0;
                    } else {
                        Kokkos::printf("Periodic BC can only be applied at ghost cells");
                        Kokkos::abort("Terminated");
                    }
                } else {
                    // bc_map.type == BCType::None
                    auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
                    double eta          = world.surface(x, y);
                    double eta_l        = world.surface(x - dx, y);
                    double eta_r        = world.surface(x + dx, y);
                    double eta_b        = world.surface(x, y - dy);
                    double eta_t        = world.surface(x, y + dy);

                    size_t direction    = 0;
                    if (eta * eta_l < 0)
                        direction |= Direction::L; // L
                    if (eta * eta_r < 0)
                        direction |= Direction::R; // R
                    if (eta * eta_b < 0)
                        direction |= Direction::B; // B
                    if (eta * eta_t < 0)
                        direction |= Direction::T; // T

                    int ncuts = std::popcount(direction);
                    if (ncuts == 0) {
                        coeff_case0(i, j);
                    } else if (ncuts == 1) {
                        coeff_case1(direction, i, j);
                    } else if (ncuts == 2) {
                        coeff_case2(direction, i, j);
                    } else {
                        // Not implemented for >2 cuts
                        throw std::runtime_error("More than 2 cuts not implemented yet.");
                    }
                }
            }
        }
        coo2crs();
    }

    /**
     * Construct Parallel threshold incomplete LU factorization ILU(t) preconditioner
     * This must be called after the laplacian matrix A has been constructed
     */
    void construct_preconditioner() {
        // preconditioner
        kh.create_par_ilut_handle();
        auto par_ilut_handle = kh.get_par_ilut_handle();
        par_ilut_handle->set_max_iter(100);
        par_ilut_handle->set_residual_norm_delta_stop(1e-3);
        par_ilut_handle->set_fill_in_limit(5.0);
        par_ilut_handle->set_verbose(verbose);

        // Pull out views from CRS
        auto row_map = A.graph.row_map;
        auto entries = A.graph.entries;
        auto values  = A.values;

        // Allocate L and U CRS views as outputs
        Kokkos::View<int*> L_row_map("L_row_map", A.numRows() + 1);
        Kokkos::View<int*> U_row_map("U_row_map", A.numRows() + 1);

        // Initial L/U approximations for A
        KokkosSparse::Experimental::par_ilut_symbolic(&kh, row_map, entries, L_row_map, U_row_map);

        // estimates of nnz
        const int nnzL_est = par_ilut_handle->get_nnzL();
        const int nnzU_est = par_ilut_handle->get_nnzU();

        Kokkos::View<int*> L_entries("L_entries", nnzL_est);
        Kokkos::View<double*> L_values("L_values", nnzL_est);
        Kokkos::View<int*> U_entries("U_entries", nnzU_est);
        Kokkos::View<double*> U_values("U_values", nnzU_est);

        KokkosSparse::Experimental::par_ilut_numeric(&kh, row_map, entries, values, L_row_map, L_entries, L_values,
                                                     U_row_map, U_entries, U_values);

        // the get_nnzL/U are only estimates, use the actual numbers
        // otherwise it throws runtime annz != this->nnz()
        const int nnzL      = L_values.extent(0);
        const int nnzU      = U_values.extent(0);
        CRS L               = CRS("L", A.numRows(), A.numCols(), nnzL, L_values, L_row_map, L_entries);
        CRS U               = CRS("U", A.numRows(), A.numCols(), nnzU, U_values, U_row_map, U_entries);
        prec                = std::make_unique<KokkosSparse::Experimental::LUPrec<CRS, KernelHandle>>(L, U);

        const auto iters    = par_ilut_handle->get_num_iters();
        const auto residual = par_ilut_handle->get_end_rel_res();
        Kokkos::printf("par ILU status: iters=%d, residual=%e\n", iters, residual);
    }

    void construct_preconditioner_spiluk() {
        kh.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1, A.numRows() + 1,
                                A.numRows() + 1, A.numRows() + 1);

        auto spiluk_handle = kh.get_spiluk_handle();

        // estimates of nnz
        const int nnzL_est = spiluk_handle->get_nnzL();
        const int nnzU_est = spiluk_handle->get_nnzU();

        Kokkos::View<int*> L_row_map("L_row_map", A.numRows() + 1);
        Kokkos::View<int*> U_row_map("U_row_map", A.numRows() + 1);
        Kokkos::View<int*> L_entries("L_entries", nnzL_est);
        Kokkos::View<double*> L_values("L_values", nnzL_est);
        Kokkos::View<int*> U_entries("U_entries", nnzU_est);
        Kokkos::View<double*> U_values("U_values", nnzU_est);
        const int fill_level = 5;

        KokkosSparse::spiluk_numeric(&kh, fill_level, A.graph.row_map, A.graph.entries, A.values, L_row_map, L_entries,
                                     L_values, U_row_map, U_entries, U_values);

        // the get_nnzL/U are only estimates, use the actual numbers
        // otherwise it throws runtime annz != this->nnz()
        const int nnzL = L_values.extent(0);
        const int nnzU = U_values.extent(0);
        CRS L          = CRS("L", A.numRows(), A.numCols(), nnzL, L_values, L_row_map, L_entries);
        CRS U          = CRS("U", A.numRows(), A.numCols(), nnzU, U_values, U_row_map, U_entries);
        prec           = std::make_unique<KokkosSparse::Experimental::LUPrec<CRS, KernelHandle>>(L, U);
    }

    void construct_rhs() {
        auto rho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                PoissonBCPair bc_map = world.poisson_bc_map(i, j);
                if (bc_map.type == PoissonBCType::None) {
                    rhs_h(index(i, j)) = -rho_h(i, j);
                } else {
                    rhs_h(index(i, j)) = bc_map.val;
                }
            }
        }
        Kokkos::deep_copy(rhs, rhs_h);
    }
    /**
     * Solve the potential field by sparse GMRES
     */
    void solve() {
        construct_rhs();

        KokkosSparse::Experimental::gmres(&kh, A, rhs, u, prec.get());
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u_2d(u.data(), nx, ny);
        Kokkos::deep_copy(world.phi, u_2d);

        auto gmres_handle      = kh.get_gmres_handle();
        const auto max_restart = gmres_handle->get_max_restart();
        const auto gmres_m     = gmres_handle->get_m();
        const auto iters       = gmres_handle->get_num_iters();
        const auto conv        = gmres_handle->get_conv_flag_val();
        const auto residual    = gmres_handle->get_end_rel_res();
        using GMRESHandle      = typename std::remove_reference<decltype(*gmres_handle)>::type;
        Kokkos::printf("GMRES status: iters=%d, residual=%e, convergence=%s\n", iters, residual,
                       (conv == GMRESHandle::Conv     ? "Conv"
                        : conv == GMRESHandle::NoConv ? "NoConv"
                                                      : "LOA"));
    }

    /**
     * Compute electric field E = -grad phi
     */
    void compute_electric_field() {
        using Kokkos::pow;
        auto E_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        auto phi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        int ngc    = world.grid.ngc;

        for (int i = ngc; i < nx - ngc; ++i) {
            for (int j = ngc; j < ny - ngc; ++j) {
                auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
                double eta          = world.surface(x, y);
                double eta_l        = world.surface(x - dx, y);
                double eta_r        = world.surface(x + dx, y);
                double eta_b        = world.surface(x, y - dy);
                double eta_t        = world.surface(x, y + dy);

                size_t direction    = 0;
                if (eta * eta_l < 0)
                    direction |= Direction::L;
                if (eta * eta_r < 0)
                    direction |= Direction::R;
                if (eta * eta_b < 0)
                    direction |= Direction::B;
                if (eta * eta_t < 0)
                    direction |= Direction::T;

                int ncuts = std::popcount(direction);
                InterfaceValue ival;
                if (ncuts == 0) {
                    ival = interface_value_case0(i, j, phi_h);
                } else if (ncuts == 1) {
                    ival = interface_value_case1(direction, i, j, phi_h);
                } else if (ncuts == 2) {
                    ival = interface_value_case2(direction, i, j, phi_h);
                } else {
                    Kokkos::abort("compute_electric_field(): More than 2 cuts not implemented yet.");
                }

                double u_c     = phi_h(i, j);
                double u_l     = ival.u_l;
                double u_r     = ival.u_r;
                double u_b     = ival.u_b;
                double u_t     = ival.u_t;
                double theta_l = ival.theta_l;
                double theta_r = ival.theta_r;
                double theta_b = ival.theta_b;
                double theta_t = ival.theta_t;

                E_h(i, j, 0) =
                    -(-pow(theta_r, 2) * u_l + (pow(theta_r, 2) - pow(theta_l, 2)) * u_c + pow(theta_l, 2) * u_r) /
                    (theta_l * theta_r * (theta_l + theta_r) * dx);
                E_h(i, j, 1) =
                    -(-pow(theta_t, 2) * u_b + (pow(theta_t, 2) - pow(theta_b, 2)) * u_c + pow(theta_b, 2) * u_t) /
                    (theta_b * theta_t * (theta_b + theta_t) * dy);
            }
        }
        Kokkos::deep_copy(world.E, E_h);
    }
};
