#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Preconditioner.hpp>
#include <KokkosSparse_gmres.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);
    using RowMapType   = Kokkos::View<int*>;
    using EntriesType  = Kokkos::View<int*>;
    using ValuesType   = Kokkos::View<double*>;

    using EXSP         = Kokkos::DefaultExecutionSpace;
    using MESP         = EXSP::memory_space;
    using CRS          = KokkosSparse::CrsMatrix<double, int, EXSP>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;

    const int nx       = 64;
    const int ny       = 64;
    double dx          = 1.0 / nx;
    double dy          = 1.0 / ny;
    std::vector<int> row_coo;
    std::vector<int> col_coo;
    std::vector<double> val_coo;
    auto index = [](int i, int j) { return i * ny + j; };
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int row_idx = index(i, j);
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                val_coo.push_back(1.0);
                row_coo.push_back(row_idx);
                col_coo.push_back(row_idx);
                continue;
            }

            double eps_l = 1.0;
            double eps_r = 1.0;
            double eps_b = 1.0;
            double eps_t = 1.0;
            row_coo.insert(row_coo.end(), {
                                              row_idx,
                                              row_idx,
                                              row_idx,
                                              row_idx,
                                              row_idx,
                                          });
            col_coo.insert(col_coo.end(), {
                                              index(i - 1, j),
                                              index(i + 1, j),
                                              index(i, j - 1),
                                              index(i, j + 1),
                                              index(i, j),
                                          });

            double bot_x = dx * dx;
            double bot_y = dy * dy;
            val_coo.insert(val_coo.end(), {
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
    }

    int nrows = nx * ny;
    int ncols = nx * ny;
    int nnz   = val_coo.size();

    // 1) make rowmap (counts -> prefix-sum)
    std::vector<int> rowmap(nrows + 1, 0);
    for (int k = 0; k < row_coo.size(); ++k) {
        rowmap[row_coo[k] + 1] += 1; // increment bucket for row
    }
    for (int i = 0; i < nrows; ++i) {
        rowmap[i + 1] += rowmap[i]; // prefix sum
    }

    // 2) scatter COO into CSR arrays (stable within row)
    std::vector<int> cur = rowmap; // current write pointer per row
    std::vector<int> cols_csr(nnz);
    std::vector<double> vals_csr(nnz);
    for (size_t k = 0; k < row_coo.size(); ++k) {
        int r          = row_coo[k];
        int dest       = cur[r]++;
        cols_csr[dest] = col_coo[k];
        vals_csr[dest] = val_coo[k];
    }

    // 3) Construct CrsMatrix from host raw arrays (the ctor will deep-copy to device)
    CRS A("A", nrows, ncols, nnz, vals_csr.data(), rowmap.data(), cols_csr.data());

    using Kokkos::sin;
    using Kokkos::numbers::pi;
    Kokkos::View<double*> B("B", nrows), X("X", nrows);
    Kokkos::deep_copy(X, 0.0);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            int idx  = i * ny + j;
            double x = (i + 0.5) * dx;
            double y = (j + 0.5) * dy;

            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                B(idx) = sin(pi * x) * sin(pi * y);
            else
                B(idx) = -2.0 * pi * pi * sin(pi * x) * sin(pi * y);
        });

    // --- Create kernel handle and GMRES handle
    KernelHandle kh;
    const double tol       = 1e-8; // convergence tolerance
    const int gmres_m      = 100;  // restart length
    const int max_restarts = 10;   // max restarts

    kh.create_gmres_handle(gmres_m, tol, max_restarts);
    auto gmres_handle = kh.get_gmres_handle();
    using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
    gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
    gmres_handle->set_verbose(true);

    // Optionally set a preconditioner here (not used in this simple example)
    // KokkosSparse::Experimental::Preconditioner<CRS>* precond = nullptr;

    // --- Run GMRES: solve A x = b
    KokkosSparse::Experimental::gmres(&kh, A, B, X /*, precond */);

    auto x_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);

    double error = 0.0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int idx  = index(i, j);
            double x = (i + 0.5) * dx;
            double y = (j + 0.5) * dy;
            error    = Kokkos::max(error, abs(x_host(idx) - sin(pi * x) * sin(pi * y)));
        }
    }
    Kokkos::printf("error %f\n", error);

    // print GMRES stats
    const auto numIters  = gmres_handle->get_num_iters();
    const auto convFlag  = gmres_handle->get_conv_flag_val();
    const auto endRelRes = gmres_handle->get_end_rel_res();

    std::cout << "GMRES stats: numIters=" << numIters << " endRelRes=" << endRelRes
              << " convFlag=" << (convFlag == GMRESHandle::Conv ? "Conv" : "NoConv/LOA/NotRun") << "\n";
    return 0;
}
