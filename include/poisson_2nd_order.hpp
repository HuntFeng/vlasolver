/*
 * Solve laplacian phi = -rho (Au=rhs)
 * This Poisson sovler uses the algorithm described in:
 * A Second-Order Boundary Condition Capturing Method for Solving the Elliptic Interface Problems on Irregular Domains
 * by Hyuntae Cho 2019, Journal of Scientific Computing, doi: https://doi.org/10.1007/s10915-019-01016-y
 *
 * FIXME: Suspect case 3 and 4 have minor bugs. In very coarse grid, a few (~<2) case 3 / 4 cells have large error.
 * This bug only affects a few cells in very coarse grid like 8x8, 16x16.
 * This bug does not affect the 2nd order convergence.
 **/
#pragma once
#include "grid.hpp"
#include "linalg.hpp"
#include "poisson.hpp"
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_LUPrec.hpp>
#include <KokkosSparse_SortCrs.hpp>
#include <KokkosSparse_coo2crs.hpp>
#include <KokkosSparse_gmres.hpp>
#include <KokkosSparse_par_ilut.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <Kokkos_Core.hpp>

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

struct InterCaseResult {
    double M_inv_d[3];
    double M_inv_N[3][49]; // [n_intf][stencil_size]; 49 = max (case3 7x7)
    double theta_r, theta_l, theta_t, theta_b;
    double bot_x, bot_y;
    double eps_r, eps_l, eps_t, eps_b;
    double eps[3];
    double theta[3];
    bool is_x[3];
    size_t dir[3];
    int n_intf;
    int stencil_size;
};

template <typename World>
class PoissonSolver2ndOrder : PoissonSolver<PoissonSolver2ndOrder<World>> {
  private:
    // some types
    using EXSP         = Kokkos::DefaultExecutionSpace;
    using MESP         = EXSP::memory_space;
    using CRS          = KokkosSparse::CrsMatrix<double, int, EXSP>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, EXSP, MESP, MESP>;

    // Held behind a shared_ptr so that capturing *this into KOKKOS_CLASS_LAMBDA
    // closures only bumps a refcount: KokkosKernelsHandle has shallow-copy +
    // freeing-destructor semantics that would double-free if copied by value.
    std::shared_ptr<KernelHandle> kh = std::make_shared<KernelHandle>();

    // input params
    World& world;
    // Value copy of the grid so cell geometry (center/spacing/ngc) is reachable
    // from device kernels without dereferencing the host-resident `world`.
    Grid<World::nspecies> grid = world.grid;
    double tol;
    int gmres_m;
    int max_restart;
    bool verbose;

    // preconditioner params
    double ilut_drop_tol;
    int ilut_max_iter;
    double ilut_fill_limit;

    // some useful params
    const int nx    = world.grid.ncells[0];
    const int ny    = world.grid.ncells[1];
    const double dx = world.grid.spacing(0, 0)[0];
    const double dy = world.grid.spacing(0, 0)[1];

    // diagonal scaling vector (D^{-1/2}) for preconditioner equilibration
    Kokkos::View<double*> D_inv_sqrt;

    // Maximum number of matrix entries (nonzeros) produced per cell. The richest
    // stencil is case3 (7x7 = 49). Each cell c=index(i,j) owns the fixed device
    // COO slot range [c*MAXNNZ, c*MAXNNZ + MAXNNZ); unused slots are padded with
    // a (row=c, col=c, val=0) entry which coo2crs sums harmlessly into the diagonal.
    static constexpr int MAXNNZ = 49;

    // Coordinate (COO) format device arrays for the sparse matrix -nabla^2
    Kokkos::View<int*> rows_coo;
    Kokkos::View<int*> cols_coo;
    Kokkos::View<double*> vals_coo;

    // use to crs format for GMRES performance
    CRS A;
    std::shared_ptr<KokkosSparse::Experimental::LUPrec<CRS, KernelHandle>> prec;
    Kokkos::View<double*> u;
    // rhs encodes the source term, jumps, and boundary conditions (device)
    Kokkos::View<double*> rhs;

    // jump conditions (device handles aliasing the world fields filled each solve)
    Kokkos::View<double**> a = world.jump_a;
    Kokkos::View<double**> b = world.jump_b;
    Kokkos::View<double**> a_tau;

    // normal (device handle aliasing world.normal, shape (nx, ny, 2):
    // component 0 is n1, component 1 is n2)
    Kokkos::View<double***> normal = world.normal;

    // device handles aliasing world level-set and permittivity fields
    Kokkos::View<double**> eta_field   = world.eta;
    Kokkos::View<double**> eps_p_field = world.eps_p;
    Kokkos::View<double**> eps_m_field = world.eps_m;

  public:
    PoissonSolver2ndOrder(World& world,
                          double tol             = 1e-12,
                          int gmres_m            = 100,
                          int max_restart        = 30,
                          bool verbose           = false,
                          double ilut_drop_tol   = 1e-10,
                          int ilut_max_iter      = 500,
                          double ilut_fill_limit = 20.0)
        : world(world),
          tol(tol),
          gmres_m(gmres_m),
          max_restart(max_restart),
          verbose(verbose),
          ilut_drop_tol(ilut_drop_tol),
          ilut_max_iter(ilut_max_iter),
          ilut_fill_limit(ilut_fill_limit) {

        // the 7x7 case3 stencil and the i+/-2 shifted-cell interpolations reach
        // three cells beyond the interior, so at least 3 ghost cells are required
        if (world.grid.ngc < 3)
            Kokkos::abort("PoissonSolver2ndOrder requires grid.ngc >= 3");

        // prepare fields (all device-resident)
        u          = Kokkos::View<double*>("u", nx * ny);
        rhs        = Kokkos::View<double*>("rhs", nx * ny);

        a_tau      = Kokkos::View<double**>("a_tau", nx, ny);

        rows_coo   = Kokkos::View<int*>("rows_coo", nx * ny * MAXNNZ);
        cols_coo   = Kokkos::View<int*>("cols_coo", nx * ny * MAXNNZ);
        vals_coo   = Kokkos::View<double*>("vals_coo", nx * ny * MAXNNZ);
        D_inv_sqrt = Kokkos::View<double*>("D_inv_sqrt", nx * ny);

        // prepare gmres
        kh->create_gmres_handle(gmres_m, tol, max_restart);
        auto gmres_handle = kh->get_gmres_handle();
        using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
        gmres_handle->set_ortho(GMRESHandle::Ortho::MGS);
        gmres_handle->set_verbose(verbose);
    }

    KOKKOS_INLINE_FUNCTION
    int index(int i, int j) const { return i * ny + j; }

    // Spatial center of cell (i,j) computed from the grid value-copy so it is
    // callable on device (mirrors Grid::center without dereferencing `world`).
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> center(int i, int j) const { return grid.center(i, j); }

    // Population count over the 4-bit Direction bitmask (std::popcount is host-only).
    KOKKOS_INLINE_FUNCTION
    int kk_popcount4(size_t d) const { return (int)((d & 1) + ((d >> 1) & 1) + ((d >> 2) & 1) + ((d >> 3) & 1)); }

    KOKKOS_INLINE_FUNCTION
    bool isclose(double val1, double val2, double rtol = 1e-12, double atol = 1e-12) const {
        return Kokkos::abs(val1 - val2) <= atol + rtol * Kokkos::abs(val2);
    }

    KOKKOS_INLINE_FUNCTION
    double compute_theta(size_t direction, int i, int j) const {
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;

        double eta     = eta_field(i, j);
        double eta_r   = eta_field(i + 1, j);
        double eta_l   = eta_field(i - 1, j);
        double eta_t   = eta_field(i, j + 1);
        double eta_b   = eta_field(i, j - 1);

        double dx_eta  = (eta_r - eta_l) / 2;
        double dy_eta  = (eta_t - eta_b) / 2;
        double dxx_eta = (eta_r - 2 * eta + eta_l) / 2;
        double dyy_eta = (eta_t - 2 * eta + eta_b) / 2;

        double d1 = 0.0, d2 = 0.0;
        double s        = dirsign(direction);
        double eta_sign = eta > 0.0 ? 1.0 : -1.0;
        bool is_x_dir   = (direction == Direction::R) || (direction == Direction::L);
        if (is_x_dir) {
            d1 = dx_eta;
            d2 = dxx_eta;
        } else {
            d1 = dy_eta;
            d2 = dyy_eta;
        }

        if (isclose(d2, 0.0))
            return abs(eta / d1);

        double disc  = d1 * d1 - 4.0 * d2 * eta;
        double theta = (-s * d1 - eta_sign * sqrt(disc)) / (2.0 * d2);
        if (theta < 1e-6 || theta > 1.0 - 1e-6)
            theta = 1.0;
        return theta;
    }

    KOKKOS_INLINE_FUNCTION
    double interp(size_t direction, double theta, int i, int j, const auto& field) const {
        using Kokkos::pow;
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
     * Map the region permittivities at an interface to the far/near convention
     * used by the stencil formulas.
     *
     * eps_p_I is the permittivity in the eta>0 region and eps_m_I the permittivity
     * in the eta<0 region (both interpolated to the interface). The formulas expect
     * eps_p = permittivity on the side opposite the cell center and eps_m =
     * permittivity on the cell-center side, so the assignment depends on the sign
     * of eta at the cell center.
     */
    KOKKOS_INLINE_FUNCTION
    void interface_eps(double eta, double eps_p_I, double eps_m_I, double& eps_p, double& eps_m) const {
        if (eta > 0.0) {
            eps_p = eps_m_I; // opposite (far) side is in the eta<0 region
            eps_m = eps_p_I; // cell-center (near) side is in the eta>0 region
        } else {
            eps_p = eps_p_I; // opposite (far) side is in the eta>0 region
            eps_m = eps_m_I; // cell-center (near) side is in the eta<0 region
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    double dirsign(size_t direction, bool is_extra = false) const {
        if (direction == Direction::R || direction == Direction::T)
            return is_extra ? -1.0 : 1.0;
        return is_extra ? 1.0 : -1.0;
    }

    KOKKOS_INLINE_FUNCTION
    void per_iface_algebraic(double s_eta,
                             double s,
                             double eps_p,
                             double eps_m,
                             double theta,
                             double a_I,
                             double& B_val,
                             Kokkos::Array<double, 4>& C_arr,
                             double& a_term) const {
        double _phi = (3.0 - 2.0 * theta) / ((1.0 - theta) * (2.0 - theta));
        double _psi = (2.0 * theta + 1.0) / (theta * (theta + 1.0));
        B_val       = s_eta * s * (eps_p * _phi + eps_m * _psi);
        C_arr[0]    = -s_eta * s * (-eps_m * theta / (1.0 + theta));
        C_arr[1]    = -s_eta * s * (eps_m * (1.0 + theta) / theta);
        C_arr[2]    = -s_eta * s * (eps_p * (2.0 - theta) / (1.0 - theta));
        C_arr[3]    = -s_eta * s * (-eps_p * (1.0 - theta) / (2.0 - theta));
        a_term      = -s * a_I * eps_p * _phi;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_P_inv(double x,
                       double y,
                       double x_r,
                       double x_l,
                       double x_ext,
                       double y_t,
                       double y_b,
                       double y_ext,
                       double P_inv[6][6]) const {
        // Build P matrix (6x6), rows: R, L, T, B, ij, ext
        // clang-format off
        double P_mat[6][6] = {
            {x_r * x_r, x_r * y, y * y, x_r, y, 1.0}, // R
            {x_l * x_l, x_l * y, y * y, x_l, y, 1.0}, // L
            {x * x, x * y_t, y_t * y_t, x, y_t, 1.0}, // T
            {x * x, x * y_b, y_b * y_b, x, y_b, 1.0}, // B
            {x * x, x * y, y * y, x, y, 1.0},         // ij
            {x_ext * x_ext, x_ext * y_ext, y_ext * y_ext, x_ext, y_ext, 1.0}, // ext
        };
        // clang-format on

        // Solve P * X = I column by column using solve_linear_system<6>
        Kokkos::Array<Kokkos::Array<double, 6>, 6> A;
        Kokkos::Array<double, 6> rhs;
        for (int col = 0; col < 6; ++col) {
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j)
                    A[i][j] = P_mat[i][j];
                rhs[i] = (i == col) ? 1.0 : 0.0;
            }
            solve_linear_system<6>(A, rhs);
            for (int i = 0; i < 6; ++i)
                P_inv[i][col] = rhs[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_grad_coeff(
        double x_I, double y_I, double n1_I, double n2_I, const double P_inv[6][6], double grad_coeff_out[6]) const {
        double grad_tau[6] = {
            -2.0 * x_I * n2_I, x_I * n1_I - y_I * n2_I, 2.0 * y_I * n1_I, -n2_I, n1_I, 0.0,
        };
        for (int j = 0; j < 6; ++j) {
            grad_coeff_out[j] = 0.0;
            for (int k = 0; k < 6; ++k)
                grad_coeff_out[j] += grad_tau[k] * P_inv[k][j];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void invert3x3(const double M[3][3], double invM[3][3]) const {
        double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                     M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        invM[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) / det;
        invM[0][1] = -(M[0][1] * M[2][2] - M[0][2] * M[2][1]) / det;
        invM[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) / det;
        invM[1][0] = -(M[1][0] * M[2][2] - M[1][2] * M[2][0]) / det;
        invM[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) / det;
        invM[1][2] = -(M[0][0] * M[1][2] - M[0][2] * M[1][0]) / det;
        invM[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) / det;
        invM[2][1] = -(M[0][0] * M[2][1] - M[0][1] * M[2][0]) / det;
        invM[2][2] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) / det;
    }

    static KOKKOS_INLINE_FUNCTION int offset25(int ox, int oy) { return (ox + 2) * 5 + (oy + 2); }
    static KOKKOS_INLINE_FUNCTION int offset49(int ox, int oy) { return (ox + 3) * 7 + (oy + 3); }

    // Stencil-size-templated offset (replaces a device-unsafe function pointer).
    template <int stencil_size>
    static KOKKOS_INLINE_FUNCTION int offset_(int ox, int oy) {
        if constexpr (stencil_size == 49)
            return (ox + 3) * 7 + (oy + 3);
        else
            return (ox + 2) * 5 + (oy + 2);
    }

    // Unified assembly of M, N, D for n_intf interfaces.
    // When B_diag is true,  B is a scalar per interface (size n_intf).
    // When B_diag is false, B is a full n_intf x n_intf matrix.
    template <int stencil_size>
    KOKKOS_INLINE_FUNCTION void assemble_MND(int n_intf,
                                             const double* B_diag,
                                             const double B_full[3][3],
                                             const Kokkos::Array<double, 4>* C,
                                             int c_size,
                                             const double* a_term,
                                             const double grad_coeff[3][6],
                                             const double* a_tau_term,
                                             const double* b_term,
                                             const size_t* dirs,
                                             const int offset_ext[2],
                                             double M[3][3],
                                             double N[3][stencil_size],
                                             double D[3]) const {
        int grad_idx[9]        = {};
        grad_idx[Direction::R] = 0;
        grad_idx[Direction::L] = 1;
        grad_idx[Direction::T] = 2;
        grad_idx[Direction::B] = 3;

        for (int d = 0; d < n_intf; ++d)
            D[d] = a_tau_term[d] + b_term[d] - a_term[d];

        // M starts from algebraic B and subtracts grad_coeff coupling
        for (int d = 0; d < n_intf; ++d) {
            for (int e = 0; e < n_intf; ++e) {
                double Bde = (d == e) ? (B_diag ? B_diag[d] : B_full[d][e]) : (B_diag ? 0.0 : B_full[d][e]);
                M[d][e]    = Bde - grad_coeff[d][grad_idx[dirs[e]]];
            }
        }

        for (int d = 0; d < n_intf; ++d)
            for (int k = 0; k < stencil_size; ++k)
                N[d][k] = 0.0;

        int dx_dir[9] = {}, dy_dir[9] = {};
        dx_dir[Direction::R] = 1;
        dy_dir[Direction::R] = 0;
        dx_dir[Direction::L] = -1;
        dy_dir[Direction::L] = 0;
        dx_dir[Direction::T] = 0;
        dy_dir[Direction::T] = 1;
        dx_dir[Direction::B] = 0;
        dy_dir[Direction::B] = -1;

        auto is_cut          = [&](size_t dflag) {
            for (int e = 0; e < n_intf; ++e)
                if (dirs[e] == dflag)
                    return true;
            return false;
        };
        for (int d = 0; d < n_intf; ++d) {
            N[d][offset_<stencil_size>(1, 0)]                         = is_cut(Direction::R) ? 0.0 : grad_coeff[d][0];
            N[d][offset_<stencil_size>(-1, 0)]                        = is_cut(Direction::L) ? 0.0 : grad_coeff[d][1];
            N[d][offset_<stencil_size>(0, 1)]                         = is_cut(Direction::T) ? 0.0 : grad_coeff[d][2];
            N[d][offset_<stencil_size>(0, -1)]                        = is_cut(Direction::B) ? 0.0 : grad_coeff[d][3];
            N[d][offset_<stencil_size>(0, 0)]                         = grad_coeff[d][4];
            N[d][offset_<stencil_size>(offset_ext[0], offset_ext[1])] = grad_coeff[d][5];

            int dx                                                    = dx_dir[dirs[d]];
            int dy                                                    = dy_dir[dirs[d]];
            for (int k = 0; k < c_size; ++k)
                N[d][offset_<stencil_size>((k - 1) * dx, (k - 1) * dy)] -= C[d][k];
        }
    }

    // -----------------------------------------------------------------------
    // Case 0 -- no cut
    // -----------------------------------------------------------------------
    // Emit one COO entry at slot p (returns p+1 so callers can chain).
    KOKKOS_INLINE_FUNCTION
    int emit(int p, int row, int col, double val) const {
        rows_coo(p) = row;
        cols_coo(p) = col;
        vals_coo(p) = val;
        return p + 1;
    }

    KOKKOS_INLINE_FUNCTION
    void coeff_case0(int i, int j) const {
        double bot_x = dx * dx;
        double bot_y = dy * dy;

        // no interface cuts this cell, so every face lies in the cell's region;
        // interpolate that region's permittivity to the half-cell faces
        double eta            = eta_field(i, j);
        const auto& eps_field = (eta > 0.0) ? eps_p_field : eps_m_field;
        double eps_l          = interp(Direction::L, 0.5, i, j, eps_field);
        double eps_r          = interp(Direction::R, 0.5, i, j, eps_field);
        double eps_b          = interp(Direction::B, 0.5, i, j, eps_field);
        double eps_t          = interp(Direction::T, 0.5, i, j, eps_field);

        int row_idx           = index(i, j);
        int p                 = row_idx * MAXNNZ;
        p                     = emit(p, row_idx, index(i - 1, j), eps_l / bot_x);
        p                     = emit(p, row_idx, index(i + 1, j), eps_r / bot_x);
        p                     = emit(p, row_idx, index(i, j - 1), eps_b / bot_y);
        p                     = emit(p, row_idx, index(i, j + 1), eps_t / bot_y);
        p                     = emit(p, row_idx, index(i, j), -(eps_l + eps_r) / bot_x - (eps_b + eps_t) / bot_y);
    }

    // -----------------------------------------------------------------------
    // Case 1 -- one interface cut
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    InterCaseResult case1(size_t direction, int i, int j) const {
        auto [x, y] = center(i, j);
        // 2D component views of the unit normal so interp() can index them as field(i, j)
        auto n1        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 0);
        auto n2        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 1);
        double eta     = eta_field(i, j);
        double s_eta   = (eta > 0.0) ? 1.0 : -1.0;
        double theta   = compute_theta(direction, i, j);
        double a_tau_I = interp(direction, theta, i, j, a_tau);
        double a_I     = interp(direction, theta, i, j, a);
        double b_I     = interp(direction, theta, i, j, b);
        double n1_I    = interp(direction, theta, i, j, n1);
        double n2_I    = interp(direction, theta, i, j, n2);
        double s       = dirsign(direction);
        bool is_x_dir  = (direction == Direction::R || direction == Direction::L);

        double theta_r = (direction == Direction::R) ? theta : 1.0;
        double theta_l = (direction == Direction::L) ? theta : 1.0;
        double theta_t = (direction == Direction::T) ? theta : 1.0;
        double theta_b = (direction == Direction::B) ? theta : 1.0;

        double eps_p_I = interp(direction, theta, i, j, eps_p_field);
        double eps_m_I = interp(direction, theta, i, j, eps_m_field);
        double eps_p, eps_m;
        interface_eps(eta, eps_p_I, eps_m_I, eps_p, eps_m);
        double eps_jump = eps_p_I - eps_m_I;

        // extension point
        double x_ext, y_ext;
        int offset_ext[2];
        if (direction == Direction::R || direction == Direction::T) {
            x_ext         = x - dx;
            y_ext         = y - dy;
            offset_ext[0] = -1;
            offset_ext[1] = -1;
        } else if (direction == Direction::L) {
            x_ext         = x + dx;
            y_ext         = y - dy;
            offset_ext[0] = 1;
            offset_ext[1] = -1;
        } else { // B
            x_ext         = x - dx;
            y_ext         = y + dy;
            offset_ext[0] = -1;
            offset_ext[1] = 1;
        }

        double x_r   = x + theta_r * dx;
        double x_l   = x - theta_l * dx;
        double y_t   = y + theta_t * dy;
        double y_b   = y - theta_b * dy;
        double bot_x = (theta_r + theta_l) / 2.0 * dx * dx;
        double bot_y = (theta_t + theta_b) / 2.0 * dy * dy;
        // half-cell faces lie in the cell's region; interpolate that region's
        // permittivity to the half-cell faces (theta/2 along each direction)
        const auto& eps_field = (eta > 0.0) ? eps_p_field : eps_m_field;
        double eps_r          = interp(Direction::R, theta_r / 2, i, j, eps_field);
        double eps_l          = interp(Direction::L, theta_l / 2, i, j, eps_field);
        double eps_t          = interp(Direction::T, theta_t / 2, i, j, eps_field);
        double eps_b          = interp(Direction::B, theta_b / 2, i, j, eps_field);

        // algebraic part
        double B_val;
        Kokkos::Array<double, 4> C_arr;
        double a_term;
        per_iface_algebraic(s_eta, s, eps_p, eps_m, theta, a_I, B_val, C_arr, a_term);

        // geometric part
        double P_inv[6][6];
        compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext, P_inv);

        double x_I = is_x_dir ? (x + s * theta * dx) : x;
        double y_I = is_x_dir ? y : (y + s * theta * dy);
        double grad_coeff[6];
        compute_grad_coeff(x_I, y_I, n1_I, n2_I, P_inv, grad_coeff);

        double dr     = is_x_dir ? dx : dy;
        double n_tang = is_x_dir ? -n2_I : n1_I;
        double n_norm = is_x_dir ? n1_I : n2_I;
        double gc[3][6];
        for (int k = 0; k < 6; ++k)
            gc[0][k] = dr * eps_jump * n_tang * grad_coeff[k];
        double a_tau_term = dr * a_tau_I * eps_p * n_tang;
        double b_term     = dr * b_I * n_norm;

        InterCaseResult r;
        r.n_intf       = 1;
        r.stencil_size = 25;
        double M[3][3], N[3][25], D[3];
        Kokkos::Array<double, 4> C_1[1] = {C_arr};
        size_t dirs[1]                  = {direction};
        assemble_MND<25>(1, &B_val, nullptr, C_1, 4, &a_term, gc, &a_tau_term, &b_term, dirs, offset_ext, M, N, D);

        r.M_inv_d[0] = D[0] / M[0][0];
        for (int k = 0; k < 25; ++k)
            r.M_inv_N[0][k] = N[0][k] / M[0][0];

        r.theta_r = theta_r;
        r.theta_l = theta_l;
        r.theta_t = theta_t;
        r.theta_b = theta_b;
        r.bot_x   = bot_x;
        r.bot_y   = bot_y;
        r.eps_r   = eps_r;
        r.eps_l   = eps_l;
        r.eps_t   = eps_t;
        r.eps_b   = eps_b;

        return r;
    }

    KOKKOS_INLINE_FUNCTION
    void coeff_case1(size_t direction, int i, int j) const {
        auto r      = case1(direction, i, j);
        int row_idx = index(i, j);

        double eps_dir, theta_dir, bot;
        if (direction == Direction::R) {
            eps_dir   = r.eps_r;
            theta_dir = r.theta_r;
            bot       = r.bot_x;
        } else if (direction == Direction::L) {
            eps_dir   = r.eps_l;
            theta_dir = r.theta_l;
            bot       = r.bot_x;
        } else if (direction == Direction::T) {
            eps_dir   = r.eps_t;
            theta_dir = r.theta_t;
            bot       = r.bot_y;
        } else {
            eps_dir   = r.eps_b;
            theta_dir = r.theta_b;
            bot       = r.bot_y;
        }
        double sub_coeff = eps_dir / theta_dir / bot;
        rhs(row_idx) -= r.M_inv_d[0] * sub_coeff;

        int p = row_idx * MAXNNZ;
        for (int ox = -2; ox <= 2; ++ox) {
            for (int oy = -2; oy <= 2; ++oy) {
                double value = r.M_inv_N[0][offset25(ox, oy)] * sub_coeff;
                if (ox == 0 && oy == 0) {
                    value += -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x -
                             (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y;
                } else if (ox == 1 && oy == 0 && !(direction & Direction::R)) {
                    value += r.eps_r / r.theta_r / r.bot_x;
                } else if (ox == -1 && oy == 0 && !(direction & Direction::L)) {
                    value += r.eps_l / r.theta_l / r.bot_x;
                } else if (ox == 0 && oy == 1 && !(direction & Direction::T)) {
                    value += r.eps_t / r.theta_t / r.bot_y;
                } else if (ox == 0 && oy == -1 && !(direction & Direction::B)) {
                    value += r.eps_b / r.theta_b / r.bot_y;
                }
                p = emit(p, row_idx, index(i + ox, j + oy), value);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Case 2 -- two interface cuts
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    InterCaseResult case2(size_t direction, int i, int j) const {
        auto [x, y] = center(i, j);
        // 2D component views of the unit normal so interp() can index them as field(i, j)
        auto n1        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 0);
        auto n2        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 1);
        double eta     = eta_field(i, j);
        double s_eta   = (eta > 0.0) ? 1.0 : -1.0;

        double theta_r = compute_theta(Direction::R, i, j);
        double theta_t = compute_theta(Direction::T, i, j);
        double theta_l = compute_theta(Direction::L, i, j);
        double theta_b = compute_theta(Direction::B, i, j);

        double x_r     = x + theta_r * dx;
        double x_l     = x - theta_l * dx;
        double y_t     = y + theta_t * dy;
        double y_b     = y - theta_b * dy;
        double bot_x   = (theta_r + theta_l) / 2.0 * dx * dx;
        double bot_y   = (theta_t + theta_b) / 2.0 * dy * dy;
        // half-cell faces lie in the cell's region; interpolate that region's
        // permittivity to the half-cell faces (theta/2 along each direction)
        const auto& eps_field = (eta > 0.0) ? eps_p_field : eps_m_field;
        double eps_r          = interp(Direction::R, theta_r / 2, i, j, eps_field);
        double eps_l          = interp(Direction::L, theta_l / 2, i, j, eps_field);
        double eps_t          = interp(Direction::T, theta_t / 2, i, j, eps_field);
        double eps_b          = interp(Direction::B, theta_b / 2, i, j, eps_field);

        size_t dir[2];
        double theta_arr[2], eps_arr[2];
        double x_I[2], y_I[2];
        double x_ext, y_ext;
        int offset_ext[2];

        if (direction == (Direction::R | Direction::T)) {
            dir[0]        = Direction::R;
            dir[1]        = Direction::T;
            theta_arr[0]  = theta_r;
            theta_arr[1]  = theta_t;
            eps_arr[0]    = eps_r;
            eps_arr[1]    = eps_t;
            x_I[0]        = x + theta_r * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y + theta_t * dy;
            x_ext         = x - dx;
            y_ext         = y - dy;
            offset_ext[0] = -1;
            offset_ext[1] = -1;
        } else if (direction == (Direction::L | Direction::T)) {
            dir[0]        = Direction::L;
            dir[1]        = Direction::T;
            theta_arr[0]  = theta_l;
            theta_arr[1]  = theta_t;
            eps_arr[0]    = eps_l;
            eps_arr[1]    = eps_t;
            x_I[0]        = x - theta_l * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y + theta_t * dy;
            x_ext         = x + dx;
            y_ext         = y - dy;
            offset_ext[0] = 1;
            offset_ext[1] = -1;
        } else if (direction == (Direction::L | Direction::B)) {
            dir[0]        = Direction::L;
            dir[1]        = Direction::B;
            theta_arr[0]  = theta_l;
            theta_arr[1]  = theta_b;
            eps_arr[0]    = eps_l;
            eps_arr[1]    = eps_b;
            x_I[0]        = x - theta_l * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y - theta_b * dy;
            x_ext         = x + dx;
            y_ext         = y + dy;
            offset_ext[0] = 1;
            offset_ext[1] = 1;
        } else { // R|B
            dir[0]        = Direction::R;
            dir[1]        = Direction::B;
            theta_arr[0]  = theta_r;
            theta_arr[1]  = theta_b;
            eps_arr[0]    = eps_r;
            eps_arr[1]    = eps_b;
            x_I[0]        = x + theta_r * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y - theta_b * dy;
            x_ext         = x - dx;
            y_ext         = y + dy;
            offset_ext[0] = -1;
            offset_ext[1] = 1;
        }

        double s[2]  = {dirsign(dir[0]), dirsign(dir[1])};
        bool is_x[2] = {dir[0] == Direction::R || dir[0] == Direction::L,
                        dir[1] == Direction::R || dir[1] == Direction::L};
        double dr[2] = {is_x[0] ? dx : dy, is_x[1] ? dx : dy};

        double eps_jump[2];
        double eps_p[2], eps_m[2];
        for (int d = 0; d < 2; ++d) {
            double eps_p_I = interp(dir[d], theta_arr[d], i, j, eps_p_field);
            double eps_m_I = interp(dir[d], theta_arr[d], i, j, eps_m_field);
            interface_eps(eta, eps_p_I, eps_m_I, eps_p[d], eps_m[d]);
            eps_jump[d] = eps_p_I - eps_m_I;
        }

        double n1_I[2], n2_I[2], a_tau_I[2], a_I[2], b_I[2];
        for (int d = 0; d < 2; ++d) {
            n1_I[d]    = interp(dir[d], theta_arr[d], i, j, n1);
            n2_I[d]    = interp(dir[d], theta_arr[d], i, j, n2);
            a_tau_I[d] = interp(dir[d], theta_arr[d], i, j, a_tau);
            a_I[d]     = interp(dir[d], theta_arr[d], i, j, a);
            b_I[d]     = interp(dir[d], theta_arr[d], i, j, b);
        }

        // algebraic
        double B[2];
        Kokkos::Array<double, 4> C[2];
        double a_term[2];
        for (int d = 0; d < 2; ++d)
            per_iface_algebraic(s_eta, s[d], eps_p[d], eps_m[d], theta_arr[d], a_I[d], B[d], C[d], a_term[d]);

        // geometric
        double P_inv[6][6];
        compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext, P_inv);
        double grad_coeff[3][6];
        for (int d = 0; d < 2; ++d) {
            double gc6[6];
            compute_grad_coeff(x_I[d], y_I[d], n1_I[d], n2_I[d], P_inv, gc6);
            double n_tang = is_x[d] ? -n2_I[d] : n1_I[d];
            for (int k = 0; k < 6; ++k)
                grad_coeff[d][k] = dr[d] * eps_jump[d] * n_tang * gc6[k];
        }

        double a_tau_term[2], b_term[2];
        for (int d = 0; d < 2; ++d) {
            double n_tang = is_x[d] ? -n2_I[d] : n1_I[d];
            double n_norm = is_x[d] ? n1_I[d] : n2_I[d];
            a_tau_term[d] = dr[d] * a_tau_I[d] * eps_p[d] * n_tang;
            b_term[d]     = dr[d] * b_I[d] * n_norm;
        }

        InterCaseResult r;
        r.n_intf       = 2;
        r.stencil_size = 25;
        double M[3][3], N[3][25], D[3];
        assemble_MND<25>(2, B, nullptr, C, 4, a_term, grad_coeff, a_tau_term, b_term, dir, offset_ext, M, N, D);

        double det    = M[0][0] * M[1][1] - M[0][1] * M[1][0];
        double invM00 = M[1][1] / det, invM01 = -M[0][1] / det;
        double invM10 = -M[1][0] / det, invM11 = M[0][0] / det;
        r.M_inv_d[0] = invM00 * D[0] + invM01 * D[1];
        r.M_inv_d[1] = invM10 * D[0] + invM11 * D[1];
        for (int k = 0; k < 25; ++k) {
            r.M_inv_N[0][k] = invM00 * N[0][k] + invM01 * N[1][k];
            r.M_inv_N[1][k] = invM10 * N[0][k] + invM11 * N[1][k];
        }

        r.theta_r  = theta_r;
        r.theta_l  = theta_l;
        r.theta_t  = theta_t;
        r.theta_b  = theta_b;
        r.bot_x    = bot_x;
        r.bot_y    = bot_y;
        r.eps_r    = eps_r;
        r.eps_l    = eps_l;
        r.eps_t    = eps_t;
        r.eps_b    = eps_b;
        r.eps[0]   = eps_arr[0];
        r.eps[1]   = eps_arr[1];
        r.theta[0] = theta_arr[0];
        r.theta[1] = theta_arr[1];

        return r;
    }

    KOKKOS_INLINE_FUNCTION
    void coeff_case2(size_t direction, int i, int j) const {
        auto r              = case2(direction, i, j);
        int row_idx         = index(i, j);

        double sub_coeff[2] = {
            r.eps[0] / r.theta[0] / r.bot_x,
            r.eps[1] / r.theta[1] / r.bot_y,
        };
        rhs(row_idx) -= r.M_inv_d[0] * sub_coeff[0] + r.M_inv_d[1] * sub_coeff[1];

        int p = row_idx * MAXNNZ;
        for (int ox = -2; ox <= 2; ++ox) {
            for (int oy = -2; oy <= 2; ++oy) {
                double value =
                    r.M_inv_N[0][offset25(ox, oy)] * sub_coeff[0] + r.M_inv_N[1][offset25(ox, oy)] * sub_coeff[1];
                if (ox == 0 && oy == 0) {
                    value += -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x -
                             (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y;
                } else if (ox == 1 && oy == 0 && !(direction & Direction::R)) {
                    value += r.eps_r / r.theta_r / r.bot_x;
                } else if (ox == -1 && oy == 0 && !(direction & Direction::L)) {
                    value += r.eps_l / r.theta_l / r.bot_x;
                } else if (ox == 0 && oy == 1 && !(direction & Direction::T)) {
                    value += r.eps_t / r.theta_t / r.bot_y;
                } else if (ox == 0 && oy == -1 && !(direction & Direction::B)) {
                    value += r.eps_b / r.theta_b / r.bot_y;
                }
                p = emit(p, row_idx, index(i + ox, j + oy), value);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Case 3 -- two cuts + one extra on outer ray
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    size_t case3_extra_dir(size_t direction, int i, int j) const {
        size_t extra = 0;
        if ((direction & Direction::R) && (eta_field(i + 1, j) * eta_field(i + 2, j) < 0))
            extra |= Direction::R;
        if ((direction & Direction::T) && (eta_field(i, j + 1) * eta_field(i, j + 2) < 0))
            extra |= Direction::T;
        if ((direction & Direction::L) && (eta_field(i - 1, j) * eta_field(i - 2, j) < 0))
            extra |= Direction::L;
        if ((direction & Direction::B) && (eta_field(i, j - 1) * eta_field(i, j - 2) < 0))
            extra |= Direction::B;
        return extra;
    }

    KOKKOS_INLINE_FUNCTION
    InterCaseResult case3(size_t direction, size_t extra, int i, int j) const {
        auto [x, y] = center(i, j);
        // 2D component views of the unit normal so interp() can index them as field(i, j)
        auto n1         = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 0);
        auto n2         = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 1);
        double eta      = eta_field(i, j);
        double s_eta    = (eta > 0.0) ? 1.0 : -1.0;

        double theta_r  = compute_theta(Direction::R, i, j);
        double theta_t  = compute_theta(Direction::T, i, j);
        double theta_l  = compute_theta(Direction::L, i, j);
        double theta_b  = compute_theta(Direction::B, i, j);
        double bot_x    = (theta_r + theta_l) / 2.0 * dx * dx;
        double bot_y    = (theta_t + theta_b) / 2.0 * dy * dy;

        double theta_rr = 0, theta_tt = 0, theta_ll = 0, theta_bb = 0;
        if (extra & Direction::R)
            theta_rr = compute_theta(Direction::R, i + 1, j);
        if (extra & Direction::T)
            theta_tt = compute_theta(Direction::T, i, j + 1);
        if (extra & Direction::L)
            theta_ll = compute_theta(Direction::L, i - 1, j);
        if (extra & Direction::B)
            theta_bb = compute_theta(Direction::B, i, j - 1);

        double x_r = x + theta_r * dx;
        double x_l = x - theta_l * dx;
        double y_t = y + theta_t * dy;
        double y_b = y - theta_b * dy;
        // half-cell faces lie in the cell's region; interpolate that region's
        // permittivity to the half-cell faces (theta/2 along each direction)
        const auto& eps_field = (eta > 0.0) ? eps_p_field : eps_m_field;
        double eps_r          = interp(Direction::R, theta_r / 2, i, j, eps_field);
        double eps_l          = interp(Direction::L, theta_l / 2, i, j, eps_field);
        double eps_t          = interp(Direction::T, theta_t / 2, i, j, eps_field);
        double eps_b          = interp(Direction::B, theta_b / 2, i, j, eps_field);

        size_t dir[3];
        double theta_arr[2], theta_extra[2];
        double x_I[3], y_I[3];
        double x_ext, y_ext;
        int offset_ext[2];

        // Dispatch the 8 sub-cases. dir[0] = non-extra, dir[1] = base with extra, dir[2] = extra
        // s[0]=dirsign(non-extra), s[1]=dirsign(base), s[2]=dirsign(extra, is_extra=true)
        if (direction == (Direction::T | Direction::R) && extra == Direction::R) {
            dir[0]         = Direction::T;
            dir[1]         = Direction::R;
            dir[2]         = Direction::R;
            theta_arr[0]   = theta_t;
            theta_arr[1]   = theta_r;
            theta_extra[0] = 0;
            theta_extra[1] = theta_rr;
            x_I[0]         = x;
            y_I[0]         = y + dirsign(Direction::T) * theta_t * dy;
            x_I[1]         = x + dirsign(Direction::R) * theta_r * dx;
            y_I[1]         = y;
            x_I[2]         = x + dirsign(Direction::R, true) * (theta_rr - 2) * dx;
            y_I[2]         = y;
            x_ext          = x - dx;
            y_ext          = y - dy;
            offset_ext[0]  = -1;
            offset_ext[1]  = -1;
        } else if (direction == (Direction::T | Direction::R) && extra == Direction::T) {
            dir[0]         = Direction::R;
            dir[1]         = Direction::T;
            dir[2]         = Direction::T;
            theta_arr[0]   = theta_r;
            theta_arr[1]   = theta_t;
            theta_extra[0] = 0;
            theta_extra[1] = theta_tt;
            x_I[0]         = x + dirsign(Direction::R) * theta_r * dx;
            y_I[0]         = y;
            x_I[1]         = x;
            y_I[1]         = y + dirsign(Direction::T) * theta_t * dy;
            x_I[2]         = x;
            y_I[2]         = y + dirsign(Direction::T, true) * (theta_tt - 2) * dy;
            x_ext          = x - dx;
            y_ext          = y - dy;
            offset_ext[0]  = -1;
            offset_ext[1]  = -1;
        } else if (direction == (Direction::T | Direction::L) && extra == Direction::T) {
            dir[0]         = Direction::L;
            dir[1]         = Direction::T;
            dir[2]         = Direction::T;
            theta_arr[0]   = theta_l;
            theta_arr[1]   = theta_t;
            theta_extra[0] = 0;
            theta_extra[1] = theta_tt;
            x_I[0]         = x + dirsign(Direction::L) * theta_l * dx;
            y_I[0]         = y;
            x_I[1]         = x;
            y_I[1]         = y + dirsign(Direction::T) * theta_t * dy;
            x_I[2]         = x;
            y_I[2]         = y + dirsign(Direction::T, true) * (theta_tt - 2) * dy;
            x_ext          = x + dx;
            y_ext          = y - dy;
            offset_ext[0]  = 1;
            offset_ext[1]  = -1;
        } else if (direction == (Direction::T | Direction::L) && extra == Direction::L) {
            dir[0]         = Direction::T;
            dir[1]         = Direction::L;
            dir[2]         = Direction::L;
            theta_arr[0]   = theta_t;
            theta_arr[1]   = theta_l;
            theta_extra[0] = 0;
            theta_extra[1] = theta_ll;
            x_I[0]         = x;
            y_I[0]         = y + dirsign(Direction::T) * theta_t * dy;
            x_I[1]         = x + dirsign(Direction::L) * theta_l * dx;
            y_I[1]         = y;
            x_I[2]         = x + dirsign(Direction::L, true) * (theta_ll - 2) * dx;
            y_I[2]         = y;
            x_ext          = x + dx;
            y_ext          = y - dy;
            offset_ext[0]  = 1;
            offset_ext[1]  = -1;
        } else if (direction == (Direction::L | Direction::B) && extra == Direction::L) {
            dir[0]         = Direction::B;
            dir[1]         = Direction::L;
            dir[2]         = Direction::L;
            theta_arr[0]   = theta_b;
            theta_arr[1]   = theta_l;
            theta_extra[0] = 0;
            theta_extra[1] = theta_ll;
            x_I[0]         = x;
            y_I[0]         = y + dirsign(Direction::B) * theta_b * dy;
            x_I[1]         = x + dirsign(Direction::L) * theta_l * dx;
            y_I[1]         = y;
            x_I[2]         = x + dirsign(Direction::L, true) * (theta_ll - 2) * dx;
            y_I[2]         = y;
            x_ext          = x + dx;
            y_ext          = y + dy;
            offset_ext[0]  = 1;
            offset_ext[1]  = 1;
        } else if (direction == (Direction::L | Direction::B) && extra == Direction::B) {
            dir[0]         = Direction::L;
            dir[1]         = Direction::B;
            dir[2]         = Direction::B;
            theta_arr[0]   = theta_l;
            theta_arr[1]   = theta_b;
            theta_extra[0] = 0;
            theta_extra[1] = theta_bb;
            x_I[0]         = x + dirsign(Direction::L) * theta_l * dx;
            y_I[0]         = y;
            x_I[1]         = x;
            y_I[1]         = y + dirsign(Direction::B) * theta_b * dy;
            x_I[2]         = x;
            y_I[2]         = y + dirsign(Direction::B, true) * (theta_bb - 2) * dy;
            x_ext          = x + dx;
            y_ext          = y + dy;
            offset_ext[0]  = 1;
            offset_ext[1]  = 1;
        } else if (direction == (Direction::R | Direction::B) && extra == Direction::B) {
            dir[0]         = Direction::R;
            dir[1]         = Direction::B;
            dir[2]         = Direction::B;
            theta_arr[0]   = theta_r;
            theta_arr[1]   = theta_b;
            theta_extra[0] = 0;
            theta_extra[1] = theta_bb;
            x_I[0]         = x + dirsign(Direction::R) * theta_r * dx;
            y_I[0]         = y;
            x_I[1]         = x;
            y_I[1]         = y + dirsign(Direction::B) * theta_b * dy;
            x_I[2]         = x;
            y_I[2]         = y + dirsign(Direction::B, true) * (theta_bb - 2) * dy;
            x_ext          = x - dx;
            y_ext          = y + dy;
            offset_ext[0]  = -1;
            offset_ext[1]  = 1;
        } else if (direction == (Direction::R | Direction::B) && extra == Direction::R) {
            dir[0]         = Direction::B;
            dir[1]         = Direction::R;
            dir[2]         = Direction::R;
            theta_arr[0]   = theta_b;
            theta_arr[1]   = theta_r;
            theta_extra[0] = 0;
            theta_extra[1] = theta_rr;
            x_I[0]         = x;
            y_I[0]         = y + dirsign(Direction::B) * theta_b * dy;
            x_I[1]         = x + dirsign(Direction::R) * theta_r * dx;
            y_I[1]         = y;
            x_I[2]         = x + dirsign(Direction::R, true) * (theta_rr - 2) * dx;
            y_I[2]         = y;
            x_ext          = x - dx;
            y_ext          = y + dy;
            offset_ext[0]  = -1;
            offset_ext[1]  = 1;
        } else {
            Kokkos::printf("case3: invalid direction/extra %d/%d\n", (int)direction, (int)extra);
            Kokkos::abort("case3: invalid sub-case");
        }

        double s[3] = {dirsign(dir[0]), dirsign(dir[1]), dirsign(dir[2], true)};

        bool is_x[3];
        double dr[3];
        for (int d = 0; d < 3; ++d) {
            is_x[d] = (dir[d] == Direction::R || dir[d] == Direction::L);
            dr[d]   = is_x[d] ? dx : dy;
        }

        // eps_p/m for 3 interfaces: interpolate the region permittivity fields to
        // each interface, then map to the far/near convention via interface_eps
        double eps_jump[3];
        double eps_p[3], eps_m[3];
        for (int d = 0; d < 2; ++d) {
            double eps_p_I = interp(dir[d], theta_arr[d], i, j, eps_p_field);
            double eps_m_I = interp(dir[d], theta_arr[d], i, j, eps_m_field);
            interface_eps(eta, eps_p_I, eps_m_I, eps_p[d], eps_m[d]);
            eps_jump[d] = eps_p_I - eps_m_I;
        }
        // Extra interface: interpolated at the shifted cell (same stencil as n1_I[2])
        {
            double eps_p_I, eps_m_I;
            if (extra == Direction::R) {
                eps_p_I = interp(Direction::L, theta_extra[1], i + 2, j, eps_p_field);
                eps_m_I = interp(Direction::L, theta_extra[1], i + 2, j, eps_m_field);
            } else if (extra == Direction::T) {
                eps_p_I = interp(Direction::B, theta_extra[1], i, j + 2, eps_p_field);
                eps_m_I = interp(Direction::B, theta_extra[1], i, j + 2, eps_m_field);
            } else if (extra == Direction::L) {
                eps_p_I = interp(Direction::R, theta_extra[1], i - 2, j, eps_p_field);
                eps_m_I = interp(Direction::R, theta_extra[1], i - 2, j, eps_m_field);
            } else { // B
                eps_p_I = interp(Direction::T, theta_extra[1], i, j - 2, eps_p_field);
                eps_m_I = interp(Direction::T, theta_extra[1], i, j - 2, eps_m_field);
            }
            interface_eps(eta, eps_p_I, eps_m_I, eps_p[2], eps_m[2]);
            eps_jump[2] = eps_p_I - eps_m_I;
        }

        // Interpolate: first two at (i,j), extra at shifted cell
        double n1_I[3], n2_I[3], a_tau_I[3], a_I[3], b_I[3];
        for (int d = 0; d < 2; ++d) {
            n1_I[d]    = interp(dir[d], theta_arr[d], i, j, n1);
            n2_I[d]    = interp(dir[d], theta_arr[d], i, j, n2);
            a_tau_I[d] = interp(dir[d], theta_arr[d], i, j, a_tau);
            a_I[d]     = interp(dir[d], theta_arr[d], i, j, a);
            b_I[d]     = interp(dir[d], theta_arr[d], i, j, b);
        }
        if (extra == Direction::R) {
            n1_I[2]    = interp(Direction::L, theta_extra[1], i + 2, j, n1);
            n2_I[2]    = interp(Direction::L, theta_extra[1], i + 2, j, n2);
            a_tau_I[2] = interp(Direction::L, theta_extra[1], i + 2, j, a_tau);
            a_I[2]     = interp(Direction::L, theta_extra[1], i + 2, j, a);
            b_I[2]     = interp(Direction::L, theta_extra[1], i + 2, j, b);
        } else if (extra == Direction::T) {
            n1_I[2]    = interp(Direction::B, theta_extra[1], i, j + 2, n1);
            n2_I[2]    = interp(Direction::B, theta_extra[1], i, j + 2, n2);
            a_tau_I[2] = interp(Direction::B, theta_extra[1], i, j + 2, a_tau);
            a_I[2]     = interp(Direction::B, theta_extra[1], i, j + 2, a);
            b_I[2]     = interp(Direction::B, theta_extra[1], i, j + 2, b);
        } else if (extra == Direction::L) {
            n1_I[2]    = interp(Direction::R, theta_extra[1], i - 2, j, n1);
            n2_I[2]    = interp(Direction::R, theta_extra[1], i - 2, j, n2);
            a_tau_I[2] = interp(Direction::R, theta_extra[1], i - 2, j, a_tau);
            a_I[2]     = interp(Direction::R, theta_extra[1], i - 2, j, a);
            b_I[2]     = interp(Direction::R, theta_extra[1], i - 2, j, b);
        } else { // B
            n1_I[2]    = interp(Direction::T, theta_extra[1], i, j - 2, n1);
            n2_I[2]    = interp(Direction::T, theta_extra[1], i, j - 2, n2);
            a_tau_I[2] = interp(Direction::T, theta_extra[1], i, j - 2, a_tau);
            a_I[2]     = interp(Direction::T, theta_extra[1], i, j - 2, a);
            b_I[2]     = interp(Direction::T, theta_extra[1], i, j - 2, b);
        }

        // B matrix (3x3) -- explicit algebraic coupling
        double B3[3][3] = {};
        double t0 = theta_arr[0], te0 = theta_extra[0];
        double t1 = theta_arr[1], te1 = theta_extra[1];
        B3[0][0] =
            s_eta * s[0] *
            (eps_p[0] * (3 - 2 * t0 - te0) / ((1 - t0) * (2 - t0 - te0)) + eps_m[0] * (2 * t0 + 1) / (t0 * (t0 + 1)));
        B3[1][1] =
            s_eta * s[1] *
            (eps_p[1] * (3 - 2 * t1 - te1) / ((1 - t1) * (2 - t1 - te1)) + eps_m[1] * (2 * t1 + 1) / (t1 * (t1 + 1)));
        B3[2][2] = s_eta * s[2] *
                   (eps_p[2] * (3 - 2 * te1 - t1) / ((1 - te1) * (2 - t1 - te1)) +
                    eps_m[2] * (2 * te1 + 1) / (te1 * (te1 + 1)));
        B3[1][2] = s_eta * s[1] * eps_p[1] * (1 - t1) / ((2 - t1 - te1) * (1 - te1));
        B3[2][1] = s_eta * s[2] * eps_p[2] * (1 - te1) / ((2 - t1 - te1) * (1 - t1));

        // C matrix (3x5) -- built directly from Python lines 910-954
        // Row 0: -s_eta * s[0] * [...]
        double C5[3][5] = {};
        C5[0][0]        = -s_eta * s[0] * (-eps_m[0] * t0 / (1 + t0));
        C5[0][1]        = -s_eta * s[0] * (eps_m[0] * (1 + t0) / t0);
        C5[0][2]        = -s_eta * s[0] * (eps_p[0] * (2 - t0) / (1 - t0));
        C5[0][3]        = -s_eta * s[0] * (-eps_p[0] * (1 - t0) / (2 - t0));
        C5[0][4]        = 0.0;

        // Row 1: -s_eta * s[1] * [...], with t0 in denominator for entry [1]
        C5[1][0] = -s_eta * s[1] * (-eps_m[1] * t1 / (1 + t1));
        C5[1][1] = -s_eta * s[1] * (eps_m[1] * (t1 + 1) / (t1 * (t0 + 1)));
        C5[1][2] = -s_eta * s[1] * (eps_p[1] * (2 - t1 - te1) / ((1 - t1) * (1 - te1)));
        C5[1][3] = 0.0;
        C5[1][4] = 0.0;

        // Row 2: +s_eta * s[2] * [...] (positive sign, NOT negative!)
        C5[2][0] = 0.0;
        C5[2][1] = 0.0;
        C5[2][2] = s_eta * s[2] * (-eps_p[2] * (2 - t1 - te1) / ((2 - t1) * (1 - te1)));
        C5[2][3] = s_eta * s[2] * (-eps_m[2] * (te1 + 1) / te1);
        C5[2][4] = s_eta * s[2] * (eps_m[2] * te1 / (te1 + 1));

        // a_term (3)
        double a_term3[3];
        a_term3[0] = -s[0] * a_I[0] * eps_p[0] * (3 - 2 * t0) / ((1 - t0) * (2 - t0));
        a_term3[1] = -s[1] * a_I[1] * eps_p[1] * (3 - 2 * t1 - te1) / ((1 - t1) * (2 - t1 - te1));
        a_term3[2] = -s[2] * a_I[2] * eps_p[1] * (3 - 2 * te1 - t1) / ((1 - te1) * (2 - t1 - te1));

        // geometric
        double P_inv[6][6];
        compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext, P_inv);
        double grad_coeff[3][6];
        for (int d = 0; d < 3; ++d) {
            double gc6[6];
            compute_grad_coeff(x_I[d], y_I[d], n1_I[d], n2_I[d], P_inv, gc6);
            double n_tang = is_x[d] ? -n2_I[d] : n1_I[d];
            for (int k = 0; k < 6; ++k)
                grad_coeff[d][k] = dr[d] * eps_jump[d] * n_tang * gc6[k];
        }

        double a_tau_term3[3], b_term3[3];
        for (int d = 0; d < 3; ++d) {
            double n_tang  = is_x[d] ? -n2_I[d] : n1_I[d];
            double n_norm  = is_x[d] ? n1_I[d] : n2_I[d];
            a_tau_term3[d] = dr[d] * a_tau_I[d] * eps_p[d] * n_tang;
            b_term3[d]     = dr[d] * b_I[d] * n_norm;
        }

        InterCaseResult r;
        r.n_intf       = 3;
        r.stencil_size = 49;
        // Use a custom assembly for case3 (full B, c_size=5)
        int grad_idx[9]        = {};
        grad_idx[Direction::R] = 0;
        grad_idx[Direction::L] = 1;
        grad_idx[Direction::T] = 2;
        grad_idx[Direction::B] = 3;
        int dx_dir[9] = {}, dy_dir[9] = {};
        dx_dir[Direction::R] = 1;
        dy_dir[Direction::R] = 0;
        dx_dir[Direction::L] = -1;
        dy_dir[Direction::L] = 0;
        dx_dir[Direction::T] = 0;
        dy_dir[Direction::T] = 1;
        dx_dir[Direction::B] = 0;
        dy_dir[Direction::B] = -1;

        double M3[3][3], N3[3][49], D3[3];
        for (int d = 0; d < 3; ++d)
            D3[d] = a_tau_term3[d] + b_term3[d] - a_term3[d];
        for (int d = 0; d < 3; ++d)
            for (int e = 0; e < 3; ++e)
                M3[d][e] = B3[d][e] - grad_coeff[d][grad_idx[dir[e]]];
        for (int d = 0; d < 3; ++d)
            for (int k = 0; k < 49; ++k)
                N3[d][k] = 0.0;

        for (int d = 0; d < 3; ++d) {
            N3[d][offset49(1, 0)] =
                (dir[0] == Direction::R || dir[1] == Direction::R || dir[2] == Direction::R) ? 0.0 : grad_coeff[d][0];
            N3[d][offset49(-1, 0)] =
                (dir[0] == Direction::L || dir[1] == Direction::L || dir[2] == Direction::L) ? 0.0 : grad_coeff[d][1];
            N3[d][offset49(0, 1)] =
                (dir[0] == Direction::T || dir[1] == Direction::T || dir[2] == Direction::T) ? 0.0 : grad_coeff[d][2];
            N3[d][offset49(0, -1)] =
                (dir[0] == Direction::B || dir[1] == Direction::B || dir[2] == Direction::B) ? 0.0 : grad_coeff[d][3];
            N3[d][offset49(0, 0)]                         = grad_coeff[d][4];
            N3[d][offset49(offset_ext[0], offset_ext[1])] = grad_coeff[d][5];
            int dx = dx_dir[dir[d]], dy = dy_dir[dir[d]];
            for (int k = 0; k < 5; ++k)
                N3[d][offset49((k - 1) * dx, (k - 1) * dy)] -= C5[d][k];
        }

        double invM3[3][3];
        invert3x3(M3, invM3);
        for (int d = 0; d < 3; ++d) {
            r.M_inv_d[d] = invM3[d][0] * D3[0] + invM3[d][1] * D3[1] + invM3[d][2] * D3[2];
            for (int k = 0; k < 49; ++k)
                r.M_inv_N[d][k] = invM3[d][0] * N3[0][k] + invM3[d][1] * N3[1][k] + invM3[d][2] * N3[2][k];
        }

        r.theta_r  = theta_r;
        r.theta_l  = theta_l;
        r.theta_t  = theta_t;
        r.theta_b  = theta_b;
        r.bot_x    = bot_x;
        r.bot_y    = bot_y;
        r.eps_r    = eps_r;
        r.eps_l    = eps_l;
        r.eps_t    = eps_t;
        r.eps_b    = eps_b;
        r.eps[0]   = (dir[0] == Direction::R)   ? eps_r
                     : (dir[0] == Direction::L) ? eps_l
                     : (dir[0] == Direction::T) ? eps_t
                                                : eps_b;
        r.eps[1]   = (dir[1] == Direction::R)   ? eps_r
                     : (dir[1] == Direction::L) ? eps_l
                     : (dir[1] == Direction::T) ? eps_t
                                                : eps_b;
        r.theta[0] = theta_arr[0];
        r.theta[1] = theta_arr[1];
        r.is_x[0]  = is_x[0];
        r.is_x[1]  = is_x[1];

        return r;
    }

    KOKKOS_INLINE_FUNCTION
    void coeff_case3(size_t direction, size_t extra, int i, int j) const {
        auto r      = case3(direction, extra, i, j);
        int row_idx = index(i, j);

        double sub_coeff[2];
        for (int d = 0; d < 2; ++d)
            sub_coeff[d] = r.eps[d] / r.theta[d] / (r.is_x[d] ? r.bot_x : r.bot_y);
        rhs(row_idx) -= r.M_inv_d[0] * sub_coeff[0] + r.M_inv_d[1] * sub_coeff[1];

        int p = row_idx * MAXNNZ;
        for (int ox = -3; ox <= 3; ++ox) {
            for (int oy = -3; oy <= 3; ++oy) {
                double value =
                    r.M_inv_N[0][offset49(ox, oy)] * sub_coeff[0] + r.M_inv_N[1][offset49(ox, oy)] * sub_coeff[1];
                if (ox == 0 && oy == 0) {
                    value += -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x -
                             (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y;
                } else if (ox == 1 && oy == 0 && !(direction & Direction::R)) {
                    value += r.eps_r / r.theta_r / r.bot_x;
                } else if (ox == -1 && oy == 0 && !(direction & Direction::L)) {
                    value += r.eps_l / r.theta_l / r.bot_x;
                } else if (ox == 0 && oy == 1 && !(direction & Direction::T)) {
                    value += r.eps_t / r.theta_t / r.bot_y;
                } else if (ox == 0 && oy == -1 && !(direction & Direction::B)) {
                    value += r.eps_b / r.theta_b / r.bot_y;
                }
                p = emit(p, row_idx, index(i + ox, j + oy), value);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Case 4 -- three interface cuts
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    InterCaseResult case4(size_t direction, int i, int j) const {
        auto [x, y] = center(i, j);
        // 2D component views of the unit normal so interp() can index them as field(i, j)
        auto n1        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 0);
        auto n2        = Kokkos::subview(normal, Kokkos::ALL, Kokkos::ALL, 1);
        double eta     = eta_field(i, j);
        double s_eta   = (eta > 0.0) ? 1.0 : -1.0;

        double theta_r = compute_theta(Direction::R, i, j);
        double theta_t = compute_theta(Direction::T, i, j);
        double theta_l = compute_theta(Direction::L, i, j);
        double theta_b = compute_theta(Direction::B, i, j);

        double x_r     = x + theta_r * dx;
        double x_l     = x - theta_l * dx;
        double y_t     = y + theta_t * dy;
        double y_b     = y - theta_b * dy;
        double bot_x   = (theta_r + theta_l) / 2.0 * dx * dx;
        double bot_y   = (theta_t + theta_b) / 2.0 * dy * dy;
        // half-cell faces lie in the cell's region; interpolate that region's
        // permittivity to the half-cell faces (theta/2 along each direction)
        const auto& eps_field = (eta > 0.0) ? eps_p_field : eps_m_field;
        double eps_r          = interp(Direction::R, theta_r / 2, i, j, eps_field);
        double eps_l          = interp(Direction::L, theta_l / 2, i, j, eps_field);
        double eps_t          = interp(Direction::T, theta_t / 2, i, j, eps_field);
        double eps_b          = interp(Direction::B, theta_b / 2, i, j, eps_field);

        size_t dir[3];
        double theta_arr[3], eps_arr[3];
        double x_I[3], y_I[3];
        bool is_x[3];
        double x_ext, y_ext;
        int offset_ext[2];

        if (direction == (Direction::R | Direction::T | Direction::L)) {
            dir[0]        = Direction::R;
            dir[1]        = Direction::T;
            dir[2]        = Direction::L;
            theta_arr[0]  = theta_r;
            theta_arr[1]  = theta_t;
            theta_arr[2]  = theta_l;
            eps_arr[0]    = eps_r;
            eps_arr[1]    = eps_t;
            eps_arr[2]    = eps_l;
            x_I[0]        = x + theta_r * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y + theta_t * dy;
            x_I[2]        = x - theta_l * dx;
            y_I[2]        = y;
            x_ext         = x - dx;
            y_ext         = y - dy;
            offset_ext[0] = -1;
            offset_ext[1] = -1;
        } else if (direction == (Direction::L | Direction::T | Direction::B)) {
            dir[0]        = Direction::L;
            dir[1]        = Direction::T;
            dir[2]        = Direction::B;
            theta_arr[0]  = theta_l;
            theta_arr[1]  = theta_t;
            theta_arr[2]  = theta_b;
            eps_arr[0]    = eps_l;
            eps_arr[1]    = eps_t;
            eps_arr[2]    = eps_b;
            x_I[0]        = x - theta_l * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y + theta_t * dy;
            x_I[2]        = x;
            y_I[2]        = y - theta_b * dy;
            x_ext         = x + dx;
            y_ext         = y - dy;
            offset_ext[0] = 1;
            offset_ext[1] = -1;
        } else if (direction == (Direction::L | Direction::B | Direction::R)) {
            dir[0]        = Direction::L;
            dir[1]        = Direction::B;
            dir[2]        = Direction::R;
            theta_arr[0]  = theta_l;
            theta_arr[1]  = theta_b;
            theta_arr[2]  = theta_r;
            eps_arr[0]    = eps_l;
            eps_arr[1]    = eps_b;
            eps_arr[2]    = eps_r;
            x_I[0]        = x - theta_l * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y - theta_b * dy;
            x_I[2]        = x + theta_r * dx;
            y_I[2]        = y;
            x_ext         = x + dx;
            y_ext         = y + dy;
            offset_ext[0] = 1;
            offset_ext[1] = 1;
        } else { // R|B|T
            dir[0]        = Direction::R;
            dir[1]        = Direction::B;
            dir[2]        = Direction::T;
            theta_arr[0]  = theta_r;
            theta_arr[1]  = theta_b;
            theta_arr[2]  = theta_t;
            eps_arr[0]    = eps_r;
            eps_arr[1]    = eps_b;
            eps_arr[2]    = eps_t;
            x_I[0]        = x + theta_r * dx;
            y_I[0]        = y;
            x_I[1]        = x;
            y_I[1]        = y - theta_b * dy;
            x_I[2]        = x;
            y_I[2]        = y + theta_t * dy;
            x_ext         = x - dx;
            y_ext         = y + dy;
            offset_ext[0] = -1;
            offset_ext[1] = 1;
        }
        for (int d = 0; d < 3; ++d)
            is_x[d] = (dir[d] == Direction::R || dir[d] == Direction::L);

        double s[3];
        for (int d = 0; d < 3; ++d)
            s[d] = dirsign(dir[d]);
        double dr[3];
        for (int d = 0; d < 3; ++d)
            dr[d] = is_x[d] ? dx : dy;

        double eps_p[3], eps_m[3];
        double eps_jump[3];
        for (int d = 0; d < 3; ++d) {
            double eps_p_I = interp(dir[d], theta_arr[d], i, j, eps_p_field);
            double eps_m_I = interp(dir[d], theta_arr[d], i, j, eps_m_field);
            interface_eps(eta, eps_p_I, eps_m_I, eps_p[d], eps_m[d]);
            eps_jump[d] = eps_p_I - eps_m_I;
        }

        double n1_I[3], n2_I[3], a_tau_I[3], a_I[3], b_I[3];
        for (int d = 0; d < 3; ++d) {
            n1_I[d]    = interp(dir[d], theta_arr[d], i, j, n1);
            n2_I[d]    = interp(dir[d], theta_arr[d], i, j, n2);
            a_tau_I[d] = interp(dir[d], theta_arr[d], i, j, a_tau);
            a_I[d]     = interp(dir[d], theta_arr[d], i, j, a);
            b_I[d]     = interp(dir[d], theta_arr[d], i, j, b);
        }

        // algebraic: diagonal B (no collinear interfaces in case4)
        double B[3];
        Kokkos::Array<double, 4> C[3];
        double a_term[3];
        for (int d = 0; d < 3; ++d)
            per_iface_algebraic(s_eta, s[d], eps_p[d], eps_m[d], theta_arr[d], a_I[d], B[d], C[d], a_term[d]);

        // geometric
        double P_inv[6][6];
        compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext, P_inv);
        double grad_coeff[3][6];
        for (int d = 0; d < 3; ++d) {
            double gc6[6];
            compute_grad_coeff(x_I[d], y_I[d], n1_I[d], n2_I[d], P_inv, gc6);
            double n_tang = is_x[d] ? -n2_I[d] : n1_I[d];
            for (int k = 0; k < 6; ++k)
                grad_coeff[d][k] = dr[d] * eps_jump[d] * n_tang * gc6[k];
        }

        double a_tau_term[3], b_term[3];
        for (int d = 0; d < 3; ++d) {
            double n_tang = is_x[d] ? -n2_I[d] : n1_I[d];
            double n_norm = is_x[d] ? n1_I[d] : n2_I[d];
            a_tau_term[d] = dr[d] * a_tau_I[d] * eps_p[d] * n_tang;
            b_term[d]     = dr[d] * b_I[d] * n_norm;
        }

        InterCaseResult r;
        r.n_intf       = 3;
        r.stencil_size = 25;
        double M[3][3], N[3][25], D[3];
        assemble_MND<25>(3, B, nullptr, C, 4, a_term, grad_coeff, a_tau_term, b_term, dir, offset_ext, M, N, D);

        double invM[3][3];
        invert3x3(M, invM);
        for (int d = 0; d < 3; ++d) {
            r.M_inv_d[d] = invM[d][0] * D[0] + invM[d][1] * D[1] + invM[d][2] * D[2];
            for (int k = 0; k < 25; ++k)
                r.M_inv_N[d][k] = invM[d][0] * N[0][k] + invM[d][1] * N[1][k] + invM[d][2] * N[2][k];
        }

        r.theta_r = theta_r;
        r.theta_l = theta_l;
        r.theta_t = theta_t;
        r.theta_b = theta_b;
        r.bot_x   = bot_x;
        r.bot_y   = bot_y;
        r.eps_r   = eps_r;
        r.eps_l   = eps_l;
        r.eps_t   = eps_t;
        r.eps_b   = eps_b;
        for (int d = 0; d < 3; ++d) {
            r.eps[d]   = eps_arr[d];
            r.theta[d] = theta_arr[d];
            r.is_x[d]  = is_x[d];
            r.dir[d]   = dir[d];
        }

        return r;
    }

    KOKKOS_INLINE_FUNCTION
    void coeff_case4(size_t direction, int i, int j) const {
        auto r      = case4(direction, i, j);
        int row_idx = index(i, j);

        double sub_coeff[3];
        for (int d = 0; d < 3; ++d)
            sub_coeff[d] = r.eps[d] / r.theta[d] / (r.is_x[d] ? r.bot_x : r.bot_y);
        rhs(row_idx) -= r.M_inv_d[0] * sub_coeff[0] + r.M_inv_d[1] * sub_coeff[1] + r.M_inv_d[2] * sub_coeff[2];

        int p = row_idx * MAXNNZ;
        for (int ox = -2; ox <= 2; ++ox) {
            for (int oy = -2; oy <= 2; ++oy) {
                double value = r.M_inv_N[0][offset25(ox, oy)] * sub_coeff[0] +
                               r.M_inv_N[1][offset25(ox, oy)] * sub_coeff[1] +
                               r.M_inv_N[2][offset25(ox, oy)] * sub_coeff[2];
                if (ox == 0 && oy == 0) {
                    value += -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x -
                             (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y;
                } else if (ox == 1 && oy == 0 && !(direction & Direction::R)) {
                    value += r.eps_r / r.theta_r / r.bot_x;
                } else if (ox == -1 && oy == 0 && !(direction & Direction::L)) {
                    value += r.eps_l / r.theta_l / r.bot_x;
                } else if (ox == 0 && oy == 1 && !(direction & Direction::T)) {
                    value += r.eps_t / r.theta_t / r.bot_y;
                } else if (ox == 0 && oy == -1 && !(direction & Direction::B)) {
                    value += r.eps_b / r.theta_b / r.bot_y;
                }
                p = emit(p, row_idx, index(i + ox, j + oy), value);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Interface value functions (for electric field computation)
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    InterfaceValue interface_value_case0(int i, int j, const auto& u) const {
        return {u(i - 1, j), u(i + 1, j), u(i, j - 1), u(i, j + 1), 1.0, 1.0, 1.0, 1.0};
    }

    KOKKOS_INLINE_FUNCTION
    InterfaceValue interface_value_case1(size_t direction, int i, int j, const auto& u) const {
        auto r = case1(direction, i, j);
        double u_arr[25];
        for (int ox = -2; ox <= 2; ++ox)
            for (int oy = -2; oy <= 2; ++oy)
                u_arr[offset25(ox, oy)] = u(i + ox, j + oy);
        double ghost = r.M_inv_d[0];
        for (int k = 0; k < 25; ++k)
            ghost += r.M_inv_N[0][k] * u_arr[k];
        return {
            (direction & Direction::L) ? ghost : u(i - 1, j),
            (direction & Direction::R) ? ghost : u(i + 1, j),
            (direction & Direction::B) ? ghost : u(i, j - 1),
            (direction & Direction::T) ? ghost : u(i, j + 1),
            r.theta_l,
            r.theta_r,
            r.theta_b,
            r.theta_t,
        };
    }

    KOKKOS_INLINE_FUNCTION
    InterfaceValue interface_value_case2(size_t direction, int i, int j, const auto& u) const {
        auto r = case2(direction, i, j);
        double u_arr[25];
        for (int ox = -2; ox <= 2; ++ox)
            for (int oy = -2; oy <= 2; ++oy)
                u_arr[offset25(ox, oy)] = u(i + ox, j + oy);
        double ghosts[2] = {r.M_inv_d[0], r.M_inv_d[1]};
        for (int k = 0; k < 25; ++k) {
            ghosts[0] += r.M_inv_N[0][k] * u_arr[k];
            ghosts[1] += r.M_inv_N[1][k] * u_arr[k];
        }
        return {
            (direction & Direction::L) ? ghosts[0] : u(i - 1, j),
            (direction & Direction::R) ? ghosts[0] : u(i + 1, j),
            (direction & Direction::B) ? ghosts[1] : u(i, j - 1),
            (direction & Direction::T) ? ghosts[1] : u(i, j + 1),
            r.theta_l,
            r.theta_r,
            r.theta_b,
            r.theta_t,
        };
    }

    KOKKOS_INLINE_FUNCTION
    InterfaceValue interface_value_case3(size_t direction, size_t extra, int i, int j, const auto& u) const {
        auto r = case3(direction, extra, i, j);
        double u_arr[49];
        for (int ox = -3; ox <= 3; ++ox)
            for (int oy = -3; oy <= 3; ++oy)
                u_arr[offset49(ox, oy)] = u(i + ox, j + oy);
        double ghosts[2] = {r.M_inv_d[0], r.M_inv_d[1]};
        for (int k = 0; k < 49; ++k) {
            ghosts[0] += r.M_inv_N[0][k] * u_arr[k];
            ghosts[1] += r.M_inv_N[1][k] * u_arr[k];
        }
        return {
            (direction & Direction::L) ? ghosts[0] : u(i - 1, j),
            (direction & Direction::R) ? ghosts[0] : u(i + 1, j),
            (direction & Direction::B) ? ghosts[1] : u(i, j - 1),
            (direction & Direction::T) ? ghosts[1] : u(i, j + 1),
            r.theta_l,
            r.theta_r,
            r.theta_b,
            r.theta_t,
        };
    }

    KOKKOS_INLINE_FUNCTION
    InterfaceValue interface_value_case4(size_t direction, int i, int j, const auto& u) const {
        auto r = case4(direction, i, j);
        double u_arr[25];
        for (int ox = -2; ox <= 2; ++ox)
            for (int oy = -2; oy <= 2; ++oy)
                u_arr[offset25(ox, oy)] = u(i + ox, j + oy);
        double ghosts[3] = {r.M_inv_d[0], r.M_inv_d[1], r.M_inv_d[2]};
        for (int k = 0; k < 25; ++k) {
            ghosts[0] += r.M_inv_N[0][k] * u_arr[k];
            ghosts[1] += r.M_inv_N[1][k] * u_arr[k];
            ghosts[2] += r.M_inv_N[2][k] * u_arr[k];
        }
        // Map ghosts to u_l/r/b/t using dir index
        int idx_l = -1, idx_r = -1, idx_b = -1, idx_t = -1;
        for (int d = 0; d < 3; ++d) {
            if (r.dir[d] == Direction::L)
                idx_l = d;
            else if (r.dir[d] == Direction::R)
                idx_r = d;
            else if (r.dir[d] == Direction::B)
                idx_b = d;
            else if (r.dir[d] == Direction::T)
                idx_t = d;
        }
        return {
            (direction & Direction::L) ? ghosts[idx_l] : u(i - 1, j),
            (direction & Direction::R) ? ghosts[idx_r] : u(i + 1, j),
            (direction & Direction::B) ? ghosts[idx_b] : u(i, j - 1),
            (direction & Direction::T) ? ghosts[idx_t] : u(i, j + 1),
            r.theta_l,
            r.theta_r,
            r.theta_b,
            r.theta_t,
        };
    }

    // -----------------------------------------------------------------------
    // COO to CRS conversion (device): the fixed-stride COO arrays are turned into
    // a CRS matrix by coo2crs, which sums duplicate (row,col) entries (so the
    // zero-padding slots fold harmlessly into the diagonal). The result is then
    // sorted within each row as required by the par-ILUT preconditioner.
    // -----------------------------------------------------------------------
    void coo2crs() {
        int n = nx * ny;
        A     = KokkosSparse::coo2crs(n, n, rows_coo, cols_coo, vals_coo);
        KokkosSparse::sort_crs_matrix(A);
    }

    void construct_fields() {
        world.poisson_jump_conditions(); // fills world.jump_a/jump_b (a/b alias them)
        int ngc = grid.ngc;

        // tangential derivative of the jump condition a; reads the unit normal from
        // world.normal (component 0 is n1, component 1 is n2)
        Kokkos::parallel_for(
            "poisson2nd_a_tau", Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double dx_a = (-a(i + 2, j) + 8 * a(i + 1, j) - 8 * a(i - 1, j) + a(i - 2, j)) / (12 * dx);
                double dy_a = (-a(i, j + 2) + 8 * a(i, j + 1) - 8 * a(i, j - 1) + a(i, j - 2)) / (12 * dy);
                a_tau(i, j) = -dx_a * normal(i, j, 1) + dy_a * normal(i, j, 0);
            });
    }

    void construct_matrix() {
        Kokkos::deep_copy(rhs, 0.0);
        int ngc             = grid.ngc;
        auto poisson_bc_map = world.poisson_bc_map; // device handle (do not deref world in-kernel)

        // Pre-fill every COO slot with an inert (row=c, col=c, val=0) entry so that
        // cells which emit fewer than MAXNNZ entries leave well-formed padding.
        Kokkos::parallel_for(
            "poisson2nd_coo_pad", Kokkos::RangePolicy<EXSP>(0, nx * ny * MAXNNZ), KOKKOS_CLASS_LAMBDA(const int s) {
                int c       = s / MAXNNZ;
                rows_coo(s) = c;
                cols_coo(s) = c;
                vals_coo(s) = 0.0;
            });

        Kokkos::parallel_for(
            "poisson2nd_assemble", Kokkos::MDRangePolicy({0, 0}, {nx, ny}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                int row_idx          = index(i, j);
                int base             = row_idx * MAXNNZ;
                PoissonBCPair bc_map = poisson_bc_map(i, j);

                if (bc_map.type == PoissonBCType::Dirichlet) {
                    emit(base, row_idx, row_idx, 1.0);
                    rhs(row_idx) = bc_map.val;
                } else if (bc_map.type == PoissonBCType::Neumann) {
                    if (i < ngc) {
                        emit(emit(base, row_idx, row_idx, -1.0), row_idx, index(i + 1, j), 1.0);
                        rhs(row_idx) = bc_map.val;
                    } else if (i >= nx - ngc) {
                        emit(emit(base, row_idx, row_idx, -1.0), row_idx, index(i - 1, j), 1.0);
                        rhs(row_idx) = -bc_map.val;
                    } else if (j < ngc) {
                        emit(emit(base, row_idx, row_idx, -1.0), row_idx, index(i, j + 1), 1.0);
                        rhs(row_idx) = bc_map.val;
                    } else if (j >= ny - ngc) {
                        emit(emit(base, row_idx, row_idx, -1.0), row_idx, index(i, j - 1), 1.0);
                        rhs(row_idx) = -bc_map.val;
                    } else {
                        Kokkos::printf("Neumann BC can only be applied at ghost cells");
                        Kokkos::abort("Terminated");
                    }
                } else if (bc_map.type == PoissonBCType::Periodic) {
                    if (i < ngc) {
                        emit(emit(base, row_idx, row_idx, 1.0), row_idx, index(nx - 2 * ngc + i, j), -1.0);
                        rhs(row_idx) = 0.0;
                    } else if (i >= nx - ngc) {
                        emit(emit(base, row_idx, row_idx, 1.0), row_idx, index(i - nx + ngc, j), -1.0);
                        rhs(row_idx) = 0.0;
                    } else if (j < ngc) {
                        emit(emit(base, row_idx, row_idx, 1.0), row_idx, index(i, ny - 2 * ngc + j), -1.0);
                        rhs(row_idx) = 0.0;
                    } else if (j >= ny - ngc) {
                        emit(emit(base, row_idx, row_idx, 1.0), row_idx, index(i, j - nx + ngc), -1.0);
                        rhs(row_idx) = 0.0;
                    } else {
                        Kokkos::printf("Periodic BC can only be applied at ghost cells");
                        Kokkos::abort("Terminated");
                    }
                } else {
                    double eta       = eta_field(i, j);
                    double eta_l     = eta_field(i - 1, j);
                    double eta_r     = eta_field(i + 1, j);
                    double eta_b     = eta_field(i, j - 1);
                    double eta_t     = eta_field(i, j + 1);

                    size_t direction = 0;
                    if (eta * eta_l < 0)
                        direction |= Direction::L;
                    if (eta * eta_r < 0)
                        direction |= Direction::R;
                    if (eta * eta_b < 0)
                        direction |= Direction::B;
                    if (eta * eta_t < 0)
                        direction |= Direction::T;

                    int ncuts = kk_popcount4(direction);
                    if (ncuts == 0) {
                        coeff_case0(i, j);
                    } else if (ncuts == 1) {
                        coeff_case1(direction, i, j);
                    } else if (ncuts == 2) {
                        size_t extra = case3_extra_dir(direction, i, j);
                        int nextra   = kk_popcount4(extra);
                        if (nextra == 0) {
                            coeff_case2(direction, i, j);
                        } else if (nextra == 1) {
                            coeff_case3(direction, extra, i, j);
                        } else {
                            Kokkos::printf("Too many extra cuts at (%d,%d), use finer grid.\n", i, j);
                            Kokkos::abort("Terminated");
                        }
                    } else if (ncuts == 3) {
                        coeff_case4(direction, i, j);
                    } else {
                        Kokkos::printf("All four sides cut at (%d,%d), use finer grid.\n", i, j);
                        Kokkos::abort("Terminated");
                    }
                }
            });
        coo2crs();
    }

    void construct_preconditioner() {
        auto row_map = A.graph.row_map;
        auto entries = A.graph.entries;
        auto values  = A.values;
        int nrows    = A.numRows();

        // Symmetric diagonal scaling (Jacobi equilibration) for better ILU quality
        auto D_inv_sqrt_h = Kokkos::create_mirror_view(D_inv_sqrt);
        auto row_map_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), row_map);
        auto entries_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), entries);
        auto values_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
        for (int i = 0; i < nrows; ++i) {
            double diag = 0.0;
            for (int k = row_map_h(i); k < row_map_h(i + 1); ++k) {
                if (entries_h(k) == i) {
                    diag = Kokkos::abs(values_h(k));
                    break;
                }
            }
            D_inv_sqrt_h(i) = (diag > 1e-30) ? 1.0 / Kokkos::sqrt(diag) : 1.0;
        }
        Kokkos::deep_copy(D_inv_sqrt, D_inv_sqrt_h);

        // Symmetrically scale the CRS matrix in-place on host
        for (int i = 0; i < nrows; ++i) {
            double di = D_inv_sqrt_h(i);
            for (int k = row_map_h(i); k < row_map_h(i + 1); ++k) {
                int j     = entries_h(k);
                double dj = D_inv_sqrt_h(j);
                values_h(k) *= di * dj;
            }
        }
        Kokkos::deep_copy(values, values_h);

        kh->create_par_ilut_handle();
        auto par_ilut_handle = kh->get_par_ilut_handle();
        par_ilut_handle->set_max_iter(ilut_max_iter);
        par_ilut_handle->set_residual_norm_delta_stop(ilut_drop_tol);
        par_ilut_handle->set_fill_in_limit(ilut_fill_limit);
        par_ilut_handle->set_verbose(verbose);

        Kokkos::View<int*> L_row_map("L_row_map", nrows + 1);
        Kokkos::View<int*> U_row_map("U_row_map", nrows + 1);
        KokkosSparse::Experimental::par_ilut_symbolic(kh.get(), row_map, entries, L_row_map, U_row_map);

        const int nnzL_est = par_ilut_handle->get_nnzL();
        const int nnzU_est = par_ilut_handle->get_nnzU();
        Kokkos::View<int*> L_entries("L_entries", nnzL_est);
        Kokkos::View<double*> L_values("L_values", nnzL_est);
        Kokkos::View<int*> U_entries("U_entries", nnzU_est);
        Kokkos::View<double*> U_values("U_values", nnzU_est);

        KokkosSparse::Experimental::par_ilut_numeric(kh.get(), row_map, entries, values, L_row_map, L_entries, L_values,
                                                     U_row_map, U_entries, U_values);
        const int nnzL      = L_values.extent(0);
        const int nnzU      = U_values.extent(0);
        CRS L               = CRS("L", nrows, A.numCols(), nnzL, L_values, L_row_map, L_entries);
        CRS U               = CRS("U", nrows, A.numCols(), nnzU, U_values, U_row_map, U_entries);
        prec                = std::make_shared<KokkosSparse::Experimental::LUPrec<CRS, KernelHandle>>(L, U);

        const auto iters    = par_ilut_handle->get_num_iters();
        const auto residual = par_ilut_handle->get_end_rel_res();
        Kokkos::printf("par ILU status: iters=%d, residual=%e\n", iters, residual);
    }

    void construct_rhs() {
        // Runs after construct_matrix, which already set rhs for BC rows and
        // subtracted the M_inv_d contribution on cut cells. Here we add the source
        // term (-rho) on interior rows and (idempotently) re-set the BC rows. rhs
        // must NOT be re-zeroed.
        auto rho            = world.rho;
        auto poisson_bc_map = world.poisson_bc_map;
        Kokkos::parallel_for(
            "poisson2nd_rhs", Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                PoissonBCPair bc_map = poisson_bc_map(i, j);
                if (bc_map.type == PoissonBCType::None)
                    rhs(index(i, j)) -= rho(i, j);
                else
                    rhs(index(i, j)) = bc_map.val;
            });
    }

    void solve() {
        world.potential_boundary_conditions();
        construct_fields();
        construct_matrix();
        construct_preconditioner();
        construct_rhs();

        // Apply diagonal scaling to RHS
        {
            auto rhs_h = Kokkos::create_mirror_view(rhs);
            Kokkos::deep_copy(rhs_h, rhs);
            auto D_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), D_inv_sqrt);
            for (int i = 0; i < nx * ny; ++i)
                rhs_h(i) *= D_h(i);
            Kokkos::deep_copy(rhs, rhs_h);
        }

        KokkosSparse::Experimental::gmres(kh.get(), A, rhs, u, prec.get());

        // Undo diagonal scaling on solution
        {
            auto u_h = Kokkos::create_mirror_view(u);
            Kokkos::deep_copy(u_h, u);
            auto D_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), D_inv_sqrt);
            for (int i = 0; i < nx * ny; ++i)
                u_h(i) *= D_h(i);
            Kokkos::deep_copy(u, u_h);
        }

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u_2d(u.data(), nx, ny);
        Kokkos::deep_copy(world.phi, u_2d);

        auto gmres_handle      = kh->get_gmres_handle();
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

    void compute_electric_field() {
        auto phi = world.phi;
        auto E   = world.E;
        int ngc  = grid.ngc;

        Kokkos::parallel_for(
            "poisson2nd_efield", Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                using Kokkos::pow;
                double eta       = eta_field(i, j);
                double eta_l     = eta_field(i - 1, j);
                double eta_r     = eta_field(i + 1, j);
                double eta_b     = eta_field(i, j - 1);
                double eta_t     = eta_field(i, j + 1);

                size_t direction = 0;
                if (eta * eta_l < 0)
                    direction |= Direction::L;
                if (eta * eta_r < 0)
                    direction |= Direction::R;
                if (eta * eta_b < 0)
                    direction |= Direction::B;
                if (eta * eta_t < 0)
                    direction |= Direction::T;

                int ncuts = kk_popcount4(direction);
                InterfaceValue ival;
                if (ncuts == 0) {
                    ival = interface_value_case0(i, j, phi);
                } else if (ncuts == 1) {
                    ival = interface_value_case1(direction, i, j, phi);
                } else if (ncuts == 2) {
                    size_t extra = case3_extra_dir(direction, i, j);
                    int nextra   = kk_popcount4(extra);
                    if (nextra == 0) {
                        ival = interface_value_case2(direction, i, j, phi);
                    } else if (nextra == 1) {
                        ival = interface_value_case3(direction, extra, i, j, phi);
                    } else {
                        Kokkos::abort("compute_electric_field: too many extra cuts");
                    }
                } else if (ncuts == 3) {
                    ival = interface_value_case4(direction, i, j, phi);
                } else {
                    Kokkos::abort("compute_electric_field: all four sides cut");
                }

                double u_c     = phi(i, j);
                double u_l     = ival.u_l;
                double u_r     = ival.u_r;
                double u_b     = ival.u_b;
                double u_t     = ival.u_t;
                double theta_l = ival.theta_l;
                double theta_r = ival.theta_r;
                double theta_b = ival.theta_b;
                double theta_t = ival.theta_t;

                E(i, j, 0) =
                    -(-pow(theta_r, 2) * u_l + (pow(theta_r, 2) - pow(theta_l, 2)) * u_c + pow(theta_l, 2) * u_r) /
                    (theta_l * theta_r * (theta_l + theta_r) * dx);
                E(i, j, 1) =
                    -(-pow(theta_t, 2) * u_b + (pow(theta_t, 2) - pow(theta_b, 2)) * u_c + pow(theta_b, 2) * u_t) /
                    (theta_b * theta_t * (theta_b + theta_t) * dy);
            });
    }
};
