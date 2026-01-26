#include <KokkosKernels_Handle.hpp> // KokkosKernelsHandle
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>        // optional helpers
#include <KokkosSparse_Preconditioner.hpp> // optional
#include <KokkosSparse_gmres.hpp>          // gmres API
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <gtest/gtest.h>
#include <iostream>

TEST(poisson_test, gmres) {
    using RowMapType        = Kokkos::View<int*>;
    using EntriesType       = Kokkos::View<int*>;
    using ValuesType        = Kokkos::View<double*>;

    using EXSP              = Kokkos::DefaultExecutionSpace;
    using MESP              = EXSP::memory_space;
    using CRS               = KokkosSparse::CrsMatrix<double, int, EXSP>;
    using KernelHandle      = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;

    const int n             = 100;
    const int diagDominance = 1;
    // choose approx nnz (e.g., ~3 per row)
    int nnz = 3 * n;
    // generate a diagonally-dominant sparse matrix (helper in repo impl)
    CRS A = KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<CRS>(n, n, nnz, 0, int(0.1 * n),
                                                                                   diagDominance);

    // RHS and solution views
    Kokkos::View<double*> B("B", n), X("X", n);

    // set B = ones, X = zeros (initial guess)
    Kokkos::deep_copy(B, 1.0);
    Kokkos::deep_copy(X, 0.0);

    // --- Create kernel handle and GMRES handle
    KernelHandle kh;
    const int gmres_m      = 50;   // restart length
    const double tol       = 1e-8; // convergence tolerance
    const int max_restarts = 10;   // max restarts

    kh.create_gmres_handle(gmres_m, tol, max_restarts);
    auto gmres_handle = kh.get_gmres_handle();
    using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
    gmres_handle->set_ortho(GMRESHandle::Ortho::CGS2);
    gmres_handle->set_verbose(false);

    // Optionally set a preconditioner here (not used in this simple example)
    // KokkosSparse::Experimental::Preconditioner<CRS>* precond = nullptr;

    // --- Run GMRES: solve A x = b
    KokkosSparse::Experimental::gmres(&kh, A, B, X /*, precond */);

    // --- Read back results on host and print a few entries + solver stats
    auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);

    std::cout << "Solution (first 10 entries):\n";
    for (int i = 0; i < n; ++i)
        std::cout << "x[" << i << "] = " << x_host[i] << "\n";

    // print GMRES stats
    const auto numIters  = gmres_handle->get_num_iters();
    const auto convFlag  = gmres_handle->get_conv_flag_val();
    const auto endRelRes = gmres_handle->get_end_rel_res();

    std::cout << "GMRES stats: numIters=" << numIters << " endRelRes=" << endRelRes
              << " convFlag=" << (convFlag == GMRESHandle::Conv ? "Conv" : "NoConv/LOA/NotRun") << "\n";
}
