/*
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
#include <Kokkos_Printf.hpp>
#include <View/Kokkos_ViewCtor.hpp>
#include <vector>

/**
 * PoissonSolver class implements the red-black Gauss-Seidel method to solve Poisson's equation
 *
 *  laplacian phi = -rho
 *
 **/
template <typename World>
class PoissonSolver {
  private:
    World& world;
    double tol;
    int gmres_m;
    int max_restart;
    bool verbose;

    // using coordinate format for constructing sparse matrix -nabla^2
    std::vector<int> rows_coo;
    std::vector<int> cols_coo;
    std::vector<double> vals_coo;

    // convert to csr format later for better GMRES perf
    using EXSP = Kokkos::DefaultExecutionSpace;
    using MESP = EXSP::memory_space;
    using CRS  = KokkosSparse::CrsMatrix<double, int, EXSP>;
    CRS A;

  public:
    PoissonSolver(World& world, double tol = 1e-6, int gmres_m = 100, int max_restart = 10, bool verbose = false)
        : world(world),
          tol(tol),
          gmres_m(gmres_m),
          max_restart(max_restart),
          verbose(verbose) {}

    inline int index(int i, int j) {
        int ny = world.grid.ncells[1];
        return i * ny + j;
    }

    /**
     * Matrix entry for cells having no cuts by interface
     */
    void coeff_case0(int i, int j) {
        double dx    = world.grid.spacing[0];
        double dy    = world.grid.spacing[1];
        double bot_x = dx * dx;
        double bot_y = dy * dy;

        double eps_l = 1.0;
        double eps_r = 1.0;
        double eps_b = 1.0;
        double eps_t = 1.0;

        int row_idx  = index(i, j);
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
    void coeff_case1() {}

    /**
     * Matrix entry for cells having 2 cuts by interface
     */
    void coeff_case2() {}

    /**
     * Convert sparse matrix coo format to csr format
     */
    void coo2csr() {
        int nx    = world.grid.ncells[0];
        int ny    = world.grid.ncells[1];
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
        const int nx = world.grid.ncells[0];
        const int ny = world.grid.ncells[1];
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

                coeff_case0(i, j);
            }
        }

        coo2csr();
    }

    /**
     * Solve the potential field by sparse GMRES
     */
    void solve() {
        int nx    = world.grid.ncells[0];
        int ny    = world.grid.ncells[1];
        auto& rho = world.rho;
        auto& phi = world.phi;

        world.potential_boundary_conditions(world.phi);

        using EXSP         = Kokkos::DefaultExecutionSpace;
        using MESP         = EXSP::memory_space;
        using CRS          = KokkosSparse::CrsMatrix<double, int, EXSP>;
        using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;
        KernelHandle kh;
        kh.create_gmres_handle(gmres_m, tol, max_restart);
        auto gmres_handle = kh.get_gmres_handle();
        using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
        gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
        gmres_handle->set_verbose(verbose);

        Kokkos::resize(Kokkos::WithoutInitializing, phi, nx * ny);
        Kokkos::resize(Kokkos::WithoutInitializing, rho, nx * ny);
        KokkosSparse::Experimental::gmres(&kh, A, rho, phi /*, precond */);
        Kokkos::resize(Kokkos::WithoutInitializing, phi, nx, ny);
        Kokkos::resize(Kokkos::WithoutInitializing, rho, nx, ny);

        const auto iters    = gmres_handle->get_num_iters();
        const auto conv     = gmres_handle->get_conv_flag_val();
        const auto residual = gmres_handle->get_end_rel_res();

        Kokkos::printf("GMRES status: iters=%d, residual=%e, convergence=%s\n", iters, residual, conv);
    }
};
