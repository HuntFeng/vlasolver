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

    // convert to csr format later for better GMRES perf
    CRS A;

    // jump conditions
    Kokkos::View<double**> a;
    Kokkos::View<double**> b;
    Kokkos::View<double**> a_tau;

    // normal
    Kokkos::View<double**> n1;
    Kokkos::View<double**> n2;

  public:
    __host__
    PoissonSolver(World& world, double tol = 1e-8, int gmres_m = 100, int max_restart = 10, bool verbose = false)
        : world(world),
          tol(tol),
          gmres_m(gmres_m),
          max_restart(max_restart),
          verbose(verbose) {}

    inline int index(int i, int j) { return i * ny + j; }

    inline bool isclose(double val1, double val2) { return Kokkos::abs(val1 - val2) < 1e-6 ? true : false; }

    void compute_normal_field() {
        n1 = Kokkos::View<double**>("n1", nx, ny);
        n2 = Kokkos::View<double**>("n2", nx, ny);
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
     * Compute tangential derivative of jump condition a at (i, j)
     */
    void compute_a_tau_field() {
        a_tau = Kokkos::View<double**>("a_tau", nx, ny);
        for (int i = 2; i < nx - 2; ++i) {
            for (int j = 2; j < ny - 2; ++j) {
                double dx_a = (-a(i, j) + 8 * a(i, j) - 8 * a(i, j) + a(i, j)) / (12 * dx);
                double dy_a = (-a(i, j) + 8 * a(i, j) - 8 * a(i, j) + a(i, j)) / (12 * dy);
                a_tau(i, j) = -dx_a * n2(i, j) + dy_a * n1(i, j);
            }
        }
    }

    double compute_theta(size_t direction, int i, int j) {
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;

        auto [x, y, vx, vy] = world.grid.center({i, j, 0, 0});
        double eta          = surface(x, y);
        double eta_r        = surface(x + dx, y);
        double eta_l        = surface(x - dx, y);
        double eta_t        = surface(x, y + dy);
        double eta_b        = surface(x, y - dy);

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
    double interp(size_t direction, double theta, int i, int j, Kokkos::View<double**>& field) {
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
        for (int p; p < 4; ++p) {
            for (int q; q < 4; ++q) {
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
    void coeff_case1(size_t direction, int i, int j) {}

    /**
     * Matrix entry for cells having 2 cuts by interface
     */
    void coeff_case2(size_t direction, int i, int j) {}

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
        auto& rho = world.rho;
        auto& phi = world.phi;
        KernelHandle kh;

        kh.create_gmres_handle(gmres_m, tol, max_restart);
        auto gmres_handle = kh.get_gmres_handle();
        using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
        gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
        gmres_handle->set_verbose(verbose);

        Kokkos::View<double*> rho1d("rho1d", nx * ny);
        Kokkos::View<double*> phi1d("phi1d", nx * ny);

        // Note: capture ny so we can access it in KOKKOS_LAMBDA
        const int _ny = ny;
        // Note: don't use KOKKOS_CLASS_LAMBDA (although it captures nx, ny conveniently)
        // otherwise the class will be marked as __host__ __device__, it breaks the host only std::vector
        Kokkos::parallel_for(
            "flatten_rho", nx * ny, KOKKOS_LAMBDA(const int idx) {
                int i      = idx / _ny;
                int j      = idx % _ny;
                rho1d(idx) = rho(i, j);
                phi1d(idx) = phi(i, j);
            });
        compute_normal_field();
        compute_a_tau_field();
        construct_matrix();

        KokkosSparse::Experimental::gmres(&kh, A, rho1d, phi1d /*, precond */);

        Kokkos::parallel_for(
            "unflatten_phi", nx * ny, KOKKOS_LAMBDA(const int idx) {
                int i     = idx / _ny;
                int j     = idx % _ny;
                phi(i, j) = phi1d(idx);
                rho(i, j) = rho1d(idx);
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
