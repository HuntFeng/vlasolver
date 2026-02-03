/*
 * Solve -laplacian phi = rho
 * This Poisson sovler uses the algorithm described in:
 * A Second-Order Boundary Condition Capturing Method for Solving the Elliptic Interface Problems on Irregular Domains
 * by Hyuntae Cho 2019, Journal of Scientific Computing, doi: https://doi.org/10.1007/s10915-019-01016-y
 *
 **/
#pragma once
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Preconditioner.hpp>
#include <KokkosSparse_gmres.hpp>
#include <Kokkos_Core.hpp>
#include <bit>
#include <vector>

enum Direction : size_t {
    R = 1 << 0, // 0001
    T = 1 << 1, // 0010
    L = 1 << 2, // 0100
    B = 1 << 3, // 1000
};

template <typename World>
class PoissonSolver {
  private:
    // some types
    using EXSP         = Kokkos::DefaultExecutionSpace;
    using MESP         = EXSP::memory_space;
    using CRS          = KokkosSparse::CrsMatrix<double, int, EXSP>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;

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

    // use to csr format for GMRES performance
    CRS A;
    Kokkos::View<double*> u;
    Kokkos::View<double*> rhs;
    Kokkos::View<double*, Kokkos::HostSpace> rhs_h;

    // jump conditions
    Kokkos::View<double**, Kokkos::HostSpace> a;
    Kokkos::View<double**, Kokkos::HostSpace> b;
    Kokkos::View<double**, Kokkos::HostSpace> a_tau;

    // normal
    Kokkos::View<double**, Kokkos::HostSpace> n1;
    Kokkos::View<double**, Kokkos::HostSpace> n2;

  public:
    __host__
    PoissonSolver(World& world, double tol = 1e-8, int gmres_m = 100, int max_restart = 10, bool verbose = false)
        : world(world),
          tol(tol),
          gmres_m(gmres_m),
          max_restart(max_restart),
          verbose(verbose) {
        u     = Kokkos::View<double*>("u", nx * ny);
        rhs   = Kokkos::View<double*>("rhs", nx * ny);
        rhs_h = Kokkos::View<double*, Kokkos::HostSpace>("rhs_h", nx * ny);

        n1    = Kokkos::View<double**, Kokkos::HostSpace>("n1", nx, ny);
        n2    = Kokkos::View<double**, Kokkos::HostSpace>("n2", nx, ny);
        a     = Kokkos::View<double**, Kokkos::HostSpace>("a", nx, ny);
        b     = Kokkos::View<double**, Kokkos::HostSpace>("b", nx, ny);
        a_tau = Kokkos::View<double**, Kokkos::HostSpace>("a_tau", nx, ny);

        // pre-compute fields
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
                double norm = sqrt(dx_eta * dx_eta + dy_eta * dy_eta);

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

        for (int i = ngc; i < nx - ngc; ++i) {
            for (int j = ngc; j < ny - ngc; ++j) {
                double dx_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dx);
                double dy_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dy);
                a_tau(i, j) = -dx_a * n2(i, j) + dy_a * n1(i, j);
            }
        }
    }

    inline int index(int i, int j) { return i * ny + j; }

    inline bool isclose(double val1, double val2) { return Kokkos::abs(val1 - val2) < 1e-6 ? true : false; }

    void compute_normal_field() {
        for (int i = 2; i < nx - 2; ++i) {
            for (int j = 2; j < ny - 2; ++j) {
                auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
                double dx_eta       = (-surface(x + 2 * dx, y) + 8 * surface(x + dx, y) - 8 * surface(x - dx, y) +
                                 surface(x - 2 * dx, y)) /
                                (12 * dx);
                double dy_eta = (-surface(x, y + 2 * dy) + 8 * surface(x, y + dy) - 8 * surface(x, y - dy) +
                                 surface(x, y - 2 * dy)) /
                                (12 * dy);
                double norm = sqrt(dx_eta * dx_eta + dy_eta * dy_eta);
                if (isclose(norm, 0.0)) {
                    n1(i, j) = 0.0;
                    n2(i, j) = 0.0;
                } else {
                    n1(i, j) = dx_eta / norm;
                    n2(i, j) = dy_eta / norm;
                }
            }
        }
    }

    /**
     * Compute Poisson jump conditions at (i, j)
     */
    void compute_jump_conditions_field() {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
                a(i, j)             = world.poisson_jump_condition_a(x, y);
                b(i, j)             = world.poisson_jump_condition_a(x, y);
            }
        }
    }

    /**
     * Compute tangential derivative of jump condition a at (i, j)
     */
    void compute_a_tau_field() {
        for (int i = 2; i < nx - 2; ++i) {
            for (int j = 2; j < ny - 2; ++j) {
                double dx_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dx);
                double dy_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dy);
                a_tau(i, j) = -dx_a * n2(i, j) + dy_a * n1(i, j);
            }
        }
    }

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
        double t_matrix[4]    = {1, theta, pow(theta, 2), pow(theta, 3)};
        double c_matrix[4][4] = {
            {0.0, 2.0, 0.0, 0.0},
            {-1.0, 0.0, 1.0, 0.0},
            {2.0, -5.0, 4.0, -1.0},
            {-1.0, 3.0, -3.0, 1.0},
        };
        double points[4];
        if (direction == Direction::R)
            double points[4] = {field(i - 1, j), field(i, j), field(i + 1, j), field(i + 2, j)};
        else if (direction == Direction::T)
            double points[4] = {field(i, j - 1), field(i, j), field(i, j + 1), field(i, j + 2)};
        else if (direction == Direction::L)
            double points[4] = {field(i + 1, j), field(i, j), field(i - 1, j), field(i - 2, j)};
        else if (direction == Direction::B)
            double points[4] = {field(i, j + 1), field(i, j), field(i, j - 1), field(i, j - 2)};
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

        // used to traverse N matrix
        static const auto offset = [](int offset_x, int offset_y) { return (offset_x + 2) * 5 + (offset_y + 2); };

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

    /**
     * Convert sparse matrix coo format to csr format
     */
    void coo2csr() {
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

        // scatter coo into csr arrays (stable within row)
        std::vector<int> cur = rowmap; // current write pointer per row
        std::vector<int> cols_csr(nnz);
        std::vector<double> vals_csr(nnz);
        for (size_t k = 0; k < rows_coo.size(); ++k) {
            int r          = rows_coo[k];
            int dest       = cur[r]++;
            cols_csr[dest] = cols_coo[k];
            vals_csr[dest] = vals_coo[k];
        }

        // constructor will deep-copy to device
        A = CRS("A", nrows, ncols, nnz, vals_csr.data(), rowmap.data(), cols_csr.data());
    }

    /**
     * Construct the Laplacian matrix -nabla^2
     */
    void construct_matrix() {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int row_idx = index(i, j);
                // TODO: assume dirichlet for now, let users modify this later
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    vals_coo.push_back(1.0);
                    rows_coo.push_back(row_idx);
                    cols_coo.push_back(row_idx);
                    rhs_h(row_idx) = 0.0;
                    continue;
                }

                // Detect interface cuts
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
        coo2csr();
    }

    /**
     * Solve the potential field by sparse GMRES
     */
    void solve() {
        KernelHandle kh;

        kh.create_gmres_handle(gmres_m, tol, max_restart);
        auto gmres_handle = kh.get_gmres_handle();
        using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
        gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
        gmres_handle->set_verbose(verbose);

        auto rho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                rhs_h(index(i, j)) = rho_h(i, j);

        construct_matrix();

        Kokkos::deep_copy(rhs, rhs_h);

        KokkosSparse::Experimental::gmres(&kh, A, rhs, u /*, precond */);

        // Note: capture these to access them in KOKKOS_LAMBDA
        // Note: don't use KOKKOS_CLASS_LAMBDA (although it captures nx, ny conveniently)
        // otherwise the class will be marked as __host__ __device__, it breaks the host only std::vector
        int _ny   = ny;
        auto& _u  = u;
        auto& phi = world.phi;
        Kokkos::parallel_for(
            "unflatten_phi", nx * ny, KOKKOS_LAMBDA(const int idx) {
                int i     = idx / _ny;
                int j     = idx % _ny;
                phi(i, j) = _u(idx);
            });

        const auto iters    = gmres_handle->get_num_iters();
        const auto conv     = gmres_handle->get_conv_flag_val();
        const auto residual = gmres_handle->get_end_rel_res();

        Kokkos::printf("GMRES status: iters=%d, residual=%e, convergence=%s\n", iters, residual,
                       (conv == GMRESHandle::Conv ? "Conv" : "NoConv/LOA/NotRun"));
    }

    /**
     * Compute electric field E = -grad phi
     */
    void compute_electric_field() {
        auto& phi = world.phi;
        auto& E   = world.E;
    }
};
