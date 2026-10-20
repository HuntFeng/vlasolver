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
#include "linalg.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_MinMax.hpp>
#include <vector>

// Role of a physical-space cell in the GCIB-PFC scheme (papers/2026VlasovIBM.pdf, Section 3.1).
enum IBCellType : int {
    IB_PLASMA = 0, // interior cell with eta >= 0, updated by the PFC scheme
    IB_GHOST  = 1, // object cell in the PFC stencil of a plasma cell, filled by extrapolate_distribution_3rd_order()
    IB_OUTER  = 2, // outer ghost cell, filled by World::particle_boundary_conditions()
    IB_OBJECT = 3, // object cell outside every PFC stencil, never read
};

template <typename World, typename PoissonSolver>
class Vlasolver {
  private:
    World& world;
    PoissonSolver& poisson_solver;
    Writer<World>& writer;

    static constexpr int N              = World::nspecies;
    static constexpr ElectronModel EM   = World::electron_model;

    // ---- ghost-cell reconstruction of the GCIB-PFC scheme (Section 3.1.2) ----
    // The least-squares fit of Eqs. (28)-(31) only depends on the geometry, so the ghost-cell value is a
    // fixed linear combination of the VDF in the stencil S_G. The weights are precomputed once on the host.
    static constexpr int ghost_half_width = 5;                                                 // w in the paper
    static constexpr int ghost_max_stencil = (2 * ghost_half_width + 1) * (2 * ghost_half_width + 1);
    int n_ghost = 0;
    Kokkos::View<int**> cell_type;       // IBCellType of every physical-space cell (nx, ny)
    Kokkos::View<int**> ghost_id;        // index into the ghost arrays below, -1 if not a ghost cell (nx, ny)
    Kokkos::View<int**> ghost_index;     // (i, j) of each ghost cell (n_ghost, 2)
    Kokkos::View<double**> ghost_normal; // unit normal n_I at the boundary intercept (n_ghost, 2)
    Kokkos::View<int*> ghost_nstencil;   // number of plasma cells in S_G (n_ghost)
    Kokkos::View<int***> ghost_stencil;  // (I, J) of the cells in S_G (n_ghost, ghost_max_stencil, 2)
    Kokkos::View<double**> ghost_w_out;  // weights for outflow velocities, mu < 0 (n_ghost, ghost_max_stencil)
    Kokkos::View<double**> ghost_w_in;   // weights for inflow velocities, mu >= 0 (n_ghost, ghost_max_stencil)

  public:
    Vlasolver(World& world, PoissonSolver& poisson_solver, Writer<World>& writer)
        : world(world),
          poisson_solver(poisson_solver),
          writer(writer) {
        build_ghost_reconstruction();
    }

  private:
    /**
     * Interpolate the level set eta and its gradient at an arbitrary point (x, y) with tensor-product
     * cubic Lagrange interpolation of the cell-centre values. Host only.
     */
    template <typename EtaHost>
    void interpolate_eta(const EtaHost& eta_h, double x, double y, double& e, double& ex, double& ey) const {
        auto& grid              = world.grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        auto [dx, dy]           = grid.spacing(0, 0);
        auto [x0, y0]           = grid.center(0, 0);
        double xi               = (x - x0) / dx;
        double yi               = (y - y0) / dy;
        int i0                  = Kokkos::clamp(static_cast<int>(Kokkos::floor(xi)) - 1, 0, nx - 4);
        int j0                  = Kokkos::clamp(static_cast<int>(Kokkos::floor(yi)) - 1, 0, ny - 4);
        double tx               = xi - i0; // local coordinate, nodes at 0, 1, 2, 3
        double ty               = yi - j0;

        // Lagrange basis on nodes {0, 1, 2, 3} and its derivative
        auto lagrange = [](double t, double* L, double* dL) {
            for (int m = 0; m < 4; ++m) {
                double denom = 1.0, val = 1.0, der = 0.0;
                for (int k = 0; k < 4; ++k) {
                    if (k == m)
                        continue;
                    denom *= (m - k);
                    val *= (t - k);
                    double prod = 1.0;
                    for (int l = 0; l < 4; ++l)
                        if (l != m && l != k)
                            prod *= (t - l);
                    der += prod;
                }
                L[m]  = val / denom;
                dL[m] = der / denom;
            }
        };
        double Lx[4], dLx[4], Ly[4], dLy[4];
        lagrange(tx, Lx, dLx);
        lagrange(ty, Ly, dLy);
        e = ex = ey = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                double v = eta_h(i0 + a, j0 + b);
                e += Lx[a] * Ly[b] * v;
                ex += dLx[a] * Ly[b] * v / dx;
                ey += Lx[a] * dLy[b] * v / dy;
            }
    }

    /**
     * Closest point (xc, yc) on Gamma (eta = 0) to the point (x, y): project onto eta = 0 along the gradient
     * (Newton step), then remove the tangential component of (x, y) - (xc, yc), until convergence.
     * Uses only the zero level set of eta, so it works for any level-set function. Host only.
     */
    template <typename EtaHost>
    void closest_point(const EtaHost& eta_h, double x, double y, double& xc, double& yc) const {
        auto [dx, dy] = world.grid.spacing(0, 0);
        double h      = Kokkos::min(dx, dy);
        double e, ex, ey;
        xc = x;
        yc = y;
        for (int it = 0; it < 100; ++it) {
            interpolate_eta(eta_h, xc, yc, e, ex, ey);
            double g2 = ex * ex + ey * ey;
            double p1 = e * ex / g2, p2 = e * ey / g2; // Newton step onto eta = 0
            xc -= p1;
            yc -= p2;
            interpolate_eta(eta_h, xc, yc, e, ex, ey);
            double gn = Kokkos::sqrt(ex * ex + ey * ey);
            double n1 = ex / gn, n2 = ey / gn;
            double d1 = x - xc, d2 = y - yc;
            double dn = d1 * n1 + d2 * n2;
            double t1 = d1 - dn * n1, t2 = d2 - dn * n2; // tangential correction
            xc += t1;
            yc += t2;
            if (Kokkos::sqrt(p1 * p1 + p2 * p2 + t1 * t1 + t2 * t2) < 1e-13 * h)
                break;
        }
        interpolate_eta(eta_h, xc, yc, e, ex, ey);
        xc -= e * ex / (ex * ex + ey * ey);
        yc -= e * ey / (ex * ex + ey * ey);
    }

    /**
     * Signed distance from (x, y) to Gamma: positive in the plasma (eta > 0), negative in the object.
     * Host only.
     */
    template <typename EtaHost>
    double signed_distance(const EtaHost& eta_h, double x, double y) const {
        double e, ex, ey, xc, yc;
        interpolate_eta(eta_h, x, y, e, ex, ey);
        closest_point(eta_h, x, y, xc, yc);
        double d = Kokkos::sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));
        return (e >= 0.0) ? d : -d;
    }

    /**
     * Precompute the ghost cells and the weights of the ghost-cell reconstruction, Eqs. (26)-(31).
     *
     * For each ghost cell with centre x_G:
     *  - x_I is the closest point on Gamma (eta = 0), found by projecting onto eta = 0 along the normal and
     *    correcting along the tangent until convergence; n_I, tau_I are the unit normal/tangent at x_I.
     *  - S_G contains the plasma cells in the (2w+1)x(2w+1) box around the ghost cell whose centre "sees"
     *    x_I, i.e. the segment from the cell centre to x_I does not cross the object (plasma-facing cells).
     *  - outflow (mu = v.n_I < 0): p = c0 + c1 s + c2 tau + c3 s^2 + c4 s tau + c5 tau^2, s = (x-x_I).n_I,
     *    tau = (x-x_I).tau_I, fitted in the least-squares sense to the cell averages in S_G; f_G = <p>_G.
     *  - inflow (mu >= 0): p = s (c1 + c3 s + c4 tau) with s the signed distance to Gamma, so that p = 0 on
     *    Gamma; f_G is the cell average of the mirror image of the fit (|s| in place of s).
     * eta only enters through its zero level set and its sign: x_I and s are computed from closest points,
     * so eta may be any level-set function (it does not need to be a signed distance).
     * Cell averages of p are evaluated with 3x3 Gauss quadrature. Since f_G is linear in the data,
     * f_G = sum_k w_k f_k, with w = e^T (A^T A)^{-1} A^T, where A holds the cell averages of the basis in S_G
     * and e holds the (mirrored) cell average of the basis over the ghost cell.
     */
    void build_ghost_reconstruction() {
        using Kokkos::abs;
        using Kokkos::sqrt;
        auto& grid              = world.grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        auto [dx, dy]           = grid.spacing(0, 0);
        double h                = Kokkos::min(dx, dy); // length scale of the local coordinates
        const int w             = ghost_half_width;

        auto eta_h       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.eta);
        cell_type        = Kokkos::View<int**>("ib_cell_type", nx, ny);
        ghost_id         = Kokkos::View<int**>("ib_ghost_id", nx, ny);
        auto cell_type_h = Kokkos::create_mirror_view(cell_type);
        auto ghost_id_h  = Kokkos::create_mirror_view(ghost_id);

        auto interior = [&](int i, int j) { return i >= ngc && i < nx - ngc && j >= ngc && j < ny - ngc; };
        auto plasma   = [&](int i, int j) {
            return i >= 0 && i < nx && j >= 0 && j < ny && interior(i, j) && eta_h(i, j) >= 0.0;
        };

        // classify cells: a ghost cell is an object cell within two cells (along x or y) of a plasma cell,
        // which is exactly the set of object cells read by the PFC stencil of the plasma cells.
        std::vector<int> gi, gj;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j) {
                ghost_id_h(i, j) = -1;
                if (!interior(i, j)) {
                    cell_type_h(i, j) = IB_OUTER;
                } else if (eta_h(i, j) >= 0.0) {
                    cell_type_h(i, j) = IB_PLASMA;
                } else {
                    bool near = false;
                    for (int d = 1; d <= 2; ++d)
                        near = near || plasma(i - d, j) || plasma(i + d, j) || plasma(i, j - d) || plasma(i, j + d);
                    cell_type_h(i, j) = near ? IB_GHOST : IB_OBJECT;
                    if (near) {
                        ghost_id_h(i, j) = static_cast<int>(gi.size());
                        gi.push_back(i);
                        gj.push_back(j);
                    }
                }
            }
        n_ghost = static_cast<int>(gi.size());

        ghost_index         = Kokkos::View<int**>("ib_ghost_index", n_ghost, 2);
        ghost_normal        = Kokkos::View<double**>("ib_ghost_normal", n_ghost, 2);
        ghost_nstencil      = Kokkos::View<int*>("ib_ghost_nstencil", n_ghost);
        ghost_stencil       = Kokkos::View<int***>("ib_ghost_stencil", n_ghost, ghost_max_stencil, 2);
        ghost_w_out         = Kokkos::View<double**>("ib_ghost_w_out", n_ghost, ghost_max_stencil);
        ghost_w_in          = Kokkos::View<double**>("ib_ghost_w_in", n_ghost, ghost_max_stencil);
        auto ghost_index_h  = Kokkos::create_mirror_view(ghost_index);
        auto ghost_normal_h = Kokkos::create_mirror_view(ghost_normal);
        auto nstencil_h     = Kokkos::create_mirror_view(ghost_nstencil);
        auto stencil_h      = Kokkos::create_mirror_view(ghost_stencil);
        auto w_out_h        = Kokkos::create_mirror_view(ghost_w_out);
        auto w_in_h         = Kokkos::create_mirror_view(ghost_w_in);
        Kokkos::deep_copy(w_out_h, 0.0);
        Kokkos::deep_copy(w_in_h, 0.0);

        // 3-point Gauss-Legendre rule on [-1/2, 1/2]
        const double gq[3] = {-0.5 * sqrt(0.6), 0.0, 0.5 * sqrt(0.6)};
        const double gw[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};

        int n_degenerate = 0;
        for (int g = 0; g < n_ghost; ++g) {
            const int i = gi[g], j = gj[g];
            auto [xG, yG] = grid.center(i, j);

            // boundary intercept x_I: closest point on Gamma
            double xI, yI, e, ex, ey;
            closest_point(eta_h, xG, yG, xI, yI);
            interpolate_eta(eta_h, xI, yI, e, ex, ey);
            double gnI = sqrt(ex * ex + ey * ey);
            double n1 = ex / gnI, n2 = ey / gnI; // n_I, pointing into the plasma
            double t1 = -n2, t2 = n1;            // tau_I

            // basis functions in local coordinates (scaled by h)
            auto basis_out = [&](double x, double y, double* b) {
                double s = ((x - xI) * n1 + (y - yI) * n2) / h;
                double t = ((x - xI) * t1 + (y - yI) * t2) / h;
                b[0] = 1.0, b[1] = s, b[2] = t, b[3] = s * s, b[4] = s * t, b[5] = t * t;
            };
            auto basis_in = [&](double x, double y, bool mirror, double* b) {
                double s = signed_distance(eta_h, x, y) / h;
                if (mirror)
                    s = abs(s);
                double t = ((x - xI) * t1 + (y - yI) * t2) / h;
                b[0] = s, b[1] = s * s, b[2] = s * t;
            };
            // cell averages of the basis over cell (I, J)
            auto average = [&](int I, int J, int nb, auto&& basis, double* avg) {
                auto [xc, yc] = grid.center(I, J);
                double b[6];
                for (int m = 0; m < nb; ++m)
                    avg[m] = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int c = 0; c < 3; ++c) {
                        basis(xc + gq[a] * dx, yc + gq[c] * dy, b);
                        for (int m = 0; m < nb; ++m)
                            avg[m] += gw[a] * gw[c] * b[m];
                    }
            };
            // plasma-facing test: the segment from the cell centre to x_I must not cross the object
            const double tol = 1e-8 * gnI * h;
            auto visible     = [&](double xc, double yc) {
                const int nsamp = 40;
                for (int k = 0; k < nsamp; ++k) {
                    double s = static_cast<double>(k) / nsamp;
                    double ee, eex, eey;
                    interpolate_eta(eta_h, xc + s * (xI - xc), yc + s * (yI - yc), ee, eex, eey);
                    if (ee < -tol)
                        return false;
                }
                return true;
            };

            // stencil S_G
            int ns = 0;
            for (int I = i - w; I <= i + w; ++I)
                for (int J = j - w; J <= j + w; ++J) {
                    if (!plasma(I, J))
                        continue;
                    auto [xc, yc] = grid.center(I, J);
                    if (!visible(xc, yc))
                        continue;
                    stencil_h(g, ns, 0) = I;
                    stencil_h(g, ns, 1) = J;
                    ++ns;
                }
            nstencil_h(g)        = ns;
            ghost_index_h(g, 0)  = i;
            ghost_index_h(g, 1)  = j;
            ghost_normal_h(g, 0) = n1;
            ghost_normal_h(g, 1) = n2;
            if (ns < 6) { // not enough cells for the quadratic fit, leave the weights at zero
                ++n_degenerate;
                continue;
            }

            // normal equations (A^T A) z = e, then w_k = A_k . z
            auto weights = [&](int nb, auto&& basis_fit, auto&& basis_ghost, auto& w_h) {
                Kokkos::Array<Kokkos::Array<double, 6>, 6> M{};
                Kokkos::Array<double, 6> z{};
                double rows[ghost_max_stencil][6];
                for (int k = 0; k < ns; ++k) {
                    average(stencil_h(g, k, 0), stencil_h(g, k, 1), nb, basis_fit, rows[k]);
                    for (int a = 0; a < nb; ++a)
                        for (int c = 0; c < nb; ++c)
                            M[a][c] += rows[k][a] * rows[k][c];
                }
                double eg[6];
                average(i, j, nb, basis_ghost, eg);
                // pad unused unknowns with an identity block so one 6x6 solver serves both fits
                for (int a = nb; a < 6; ++a)
                    M[a][a] = 1.0;
                for (int a = 0; a < nb; ++a)
                    z[a] = eg[a];
                solve_linear_system(M, z);
                for (int k = 0; k < ns; ++k) {
                    double wk = 0.0;
                    for (int a = 0; a < nb; ++a)
                        wk += rows[k][a] * z[a];
                    w_h(g, k) = wk;
                }
            };
            weights(6, basis_out, basis_out, w_out_h);
            weights(
                3, [&](double x, double y, double* b) { basis_in(x, y, false, b); },
                [&](double x, double y, double* b) { basis_in(x, y, true, b); }, w_in_h);
        }

        Kokkos::deep_copy(cell_type, cell_type_h);
        Kokkos::deep_copy(ghost_id, ghost_id_h);
        Kokkos::deep_copy(ghost_index, ghost_index_h);
        Kokkos::deep_copy(ghost_normal, ghost_normal_h);
        Kokkos::deep_copy(ghost_nstencil, nstencil_h);
        Kokkos::deep_copy(ghost_stencil, stencil_h);
        Kokkos::deep_copy(ghost_w_out, w_out_h);
        Kokkos::deep_copy(ghost_w_in, w_in_h);
        Kokkos::printf("(VlasovSolver) GCIB ghost cells: %d", n_ghost);
        if (n_degenerate > 0)
            Kokkos::printf(", WARNING: %d ghost cells have fewer than 6 cells in S_G and are set to 0", n_degenerate);
        Kokkos::printf("\n");
    }

    /**
     * Zhang-Shu scaling limiter theta_j, Eqs. (24)-(25): the largest theta in [0, 1] such that
     * f_j + theta (p_j - f_j) stays within [0, f_inf] on the cell, where p_j is the PFC quadratic
     * reconstruction from f_{j-1}, f_j, f_{j+1}.
     */
    KOKKOS_INLINE_FUNCTION
    static double scaling_limiter(double fm, double f0, double fp, double f_inf) {
        double a    = 0.5 * (fp - fm);
        double b    = 0.5 * (fp - 2.0 * f0 + fm);
        double pl   = f0 - 0.5 * a + b / 6.0; // p(-1/2)
        double pr   = f0 + 0.5 * a + b / 6.0; // p(+1/2)
        double pmin = Kokkos::min(pl, pr);
        double pmax = Kokkos::max(pl, pr);
        if (b != 0.0) {
            double xs = -a / (2.0 * b); // vertex of the parabola
            if (Kokkos::abs(xs) <= 0.5) {
                double pv = f0 + a * xs + b * (xs * xs - 1.0 / 12.0);
                pmin      = Kokkos::min(pmin, pv);
                pmax      = Kokkos::max(pmax, pv);
            }
        }
        double theta = 1.0;
        if (pmin < 0.0)
            theta = (f0 - pmin > 0.0) ? Kokkos::min(theta, Kokkos::max(f0, 0.0) / (f0 - pmin)) : 0.0;
        if (pmax > f_inf)
            theta = (pmax - f0 > 0.0) ? Kokkos::min(theta, Kokkos::max(f_inf - f0, 0.0) / (pmax - f0)) : 0.0;
        return Kokkos::clamp(theta, 0.0, 1.0);
    }

  public:

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

                        // (x_b, y_b) is the intersection point between the normal line and the boundary.
                        // Coordinates are measured relative to it, i.e. x = X - x_b, y = Y - y_b.
                        double x_b = x0 - eta * n1;
                        double y_b = y0 - eta * n2;

                        // We assume f near the boundary can be approximated by a plane: f(x,y) = c0 * x + c1 * y + c2
                        // solve least square problem A @ c = b (where c = [c0, c1, c2])
                        // the normal equation is A^T @ A @ c = A^T @ b
                        // sum x_i^2 * c0 + sum x_i*y_i * c1 + sum x_i * c2 = sum x_i*f_i
                        // sum x_i*y_i * c0 + sum y_i^2 * c1 + sum y_i * c2 = sum y_i*f_i
                        // sum x_i * c0 + sum y_i * c1 + sum 1 * c2 = sum f_i
                        for (int x_offset = -offset_range; x_offset <= offset_range; ++x_offset) {
                            for (int y_offset = -offset_range; y_offset <= offset_range; ++y_offset) {
                                int I = i + x_offset;
                                int J = j + y_offset;
                                // skip ghost cells
                                if (I < ngc || I > nx - ngc - 1 || J < ngc || J > ny - ngc - 1 || eta_field(I, J) < 0)
                                    continue;

                                // only extrapolate using fluid cells
                                auto [X, Y, VX, VY] = grid.center(I, J, iv, jv, sp);
                                double x            = X - x_b;
                                double y            = Y - y_b;
                                double z            = f(I, J, iv, jv, sp);
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
                        // For inflow, we force zero-inflow condition, i.e. f(x_b, y_b) = 0.
                        // In boundary-relative coordinates this is the hard constraint c2 = 0, so the plane
                        // becomes f(x,y) = c0 * x + c1 * y and the least square problem only has two unknowns.
                        // Its normal equation is the upper-left 2x2 block of the full system:
                        // sum x_i^2 * c0 + sum x_i*y_i * c1 = sum x_i*f_i
                        // sum x_i*y_i * c0 + sum y_i^2 * c1 = sum y_i*f_i
                        if (v_dot_n >= 0.0) {
                            Kokkos::Array<Kokkos::Array<double, 2>, 2> A2{{{A[0][0], A[0][1]}, {A[1][0], A[1][1]}}};
                            Kokkos::Array<double, 2> b2{b[0], b[1]};
                            solve_linear_system(A2, b2);
                            b[0] = b2[0];
                            b[1] = b2[1];
                            b[2] = 0.0;
                        } else {
                            solve_linear_system(A, b);
                        }

                        // extrapolate using best-fit plane, evaluated at (x0, y0) in boundary-relative coordinates
                        f(i, j, iv, jv, sp) = b[0] * (x0 - x_b) + b[1] * (y0 - y_b) + b[2];
                    }
                }
            });
    }

    /**
     * Ghost-cell reconstruction of the GCIB-PFC scheme (Section 3.1.2), third-order accurate for outflow.
     * Fills every ghost cell (object cells in the PFC stencil of a plasma cell, two layers deep) for each
     * discrete velocity, using the weights precomputed by build_ghost_reconstruction():
     *   mu = v.n_I <  0 (outflow): quadratic least-squares fit, Eqs. (28)-(30)
     *   mu = v.n_I >= 0 (inflow):  fit vanishing on Gamma, mirrored across Gamma, Eq. (31)
     * The result is limited to [0, max_{S_G} f] so that the ghost cells obey the same bounds as the plasma.
     */
    void extrapolate_distribution_3rd_order() const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        auto ghost_index        = this->ghost_index;
        auto ghost_normal       = this->ghost_normal;
        auto ghost_nstencil     = this->ghost_nstencil;
        auto ghost_stencil      = this->ghost_stencil;
        auto ghost_w_out        = this->ghost_w_out;
        auto ghost_w_in         = this->ghost_w_in;
        if (n_ghost == 0)
            return;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, ngc, ngc}, {n_ghost, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int g, const int iv, const int jv) {
                int i  = ghost_index(g, 0);
                int j  = ghost_index(g, 1);
                int ns = ghost_nstencil(g);
                for (int sp = 0; sp < N; ++sp) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);
                    bool inflow         = vx * ghost_normal(g, 0) + vy * ghost_normal(g, 1) >= 0.0;
                    double value = 0.0, f_max = 0.0;
                    for (int k = 0; k < ns; ++k) {
                        double fk = f(ghost_stencil(g, k, 0), ghost_stencil(g, k, 1), iv, jv, sp);
                        value += (inflow ? ghost_w_in(g, k) : ghost_w_out(g, k)) * fk;
                        f_max = Kokkos::max(f_max, fk);
                    }
                    f(i, j, iv, jv, sp) = Kokkos::clamp(value, 0.0, f_max);
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

    void pfc_update_deprecated(double dt, int axis, int sp) const {
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
        auto& df                = world.df;
        auto& m                 = world.m;
        auto& q                 = world.q;

        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::deep_copy(flux, 0.0);
        Kokkos::deep_copy(flux_1st, 0.0);

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

        // Diagnostic guard: the negative-distribution report below aborts the run, so we
        // only need one thread's output. This scalar View (zero-initialized) is claimed
        // atomically by the first offending thread; every other thread skips the report.
        // Without this, splitting the report into several printf calls would let the
        // messages from concurrent threads interleave. Captured by value into the device
        // lambda via KOKKOS_CLASS_LAMBDA (= [=, *this]).
        Kokkos::View<int> report_claimed("pfc_report_claimed");

        // Step 2: Compute the limited PFC update increment for every fluid cell.
        //
        // IMPORTANT: this pass only READS f (never writes it) and writes the increment
        // into the scratch view `df`. The Xiong2014 limiters couple neighboring cells via
        // min(), so compute_ep for a cell reads f at that cell to form `delta`. If the
        // limiter computation and the `f += ...` update were fused into a single kernel
        // (as an earlier memory-saving version did), a thread's compute_ep could read a
        // neighbor cell whose own thread had already applied its in-place update. That
        // stale/mixed read makes the neighbor's `delta` come out positive, its limiter go
        // negative, and (through the min() coupling) drives a good cell's distribution
        // negative. Separating "compute increment" from "apply increment" into two passes
        // guarantees every compute_ep sees the pristine pre-update f. We store a single
        // per-cell increment (df) rather than the two full-domain limiter views (ep_l,
        // ep_r), so this stays cheaper than the classic three-pass layout.
        Kokkos::deep_copy(df, 0.0);
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

                // Xiong2014 (http://dx.doi.org/10.1016/j.jcp.2014.05.033) flux limiters
                // for the cell (I,J,Iv,Jv), returned as {ep_l, ep_r}.
                auto compute_ep = [&](int I, int J, int Iv, int Jv) -> Kokkos::pair<double, double> {
                    double fl_l  = (axis == 0)   ? flux(I - 1, J, Iv, Jv)
                                   : (axis == 1) ? flux(I, J - 1, Iv, Jv)
                                   : (axis == 2) ? flux(I, J, Iv - 1, Jv)
                                                 : flux(I, J, Iv, Jv - 1);
                    double fl_r  = flux(I, J, Iv, Jv);
                    double f1_l  = (axis == 0)   ? flux_1st(I - 1, J, Iv, Jv)
                                   : (axis == 1) ? flux_1st(I, J - 1, Iv, Jv)
                                   : (axis == 2) ? flux_1st(I, J, Iv - 1, Jv)
                                                 : flux_1st(I, J, Iv, Jv - 1);
                    double f1_r  = flux_1st(I, J, Iv, Jv);
                    double d_l   = fl_l - f1_l;                        // high order correction, left
                    double d_r   = fl_r - f1_r;                        // high order correction, right
                    double delta = -f(I, J, Iv, Jv, sp) - f1_l + f1_r; // 0.0 - 1st_order_update, always <= 0
                    double p     = d_l - d_r - delta;                  // 3rd order update without limiter
                    double ep_l_val = 1.0, ep_r_val = 1.0;
                    if (d_l >= 0 && d_r <= 0.0) {
                        ep_l_val = 1.0;
                        ep_r_val = 1.0;
                    } else if (d_l >= 0.0 && d_r > 0.0) {
                        ep_l_val = 1.0;
                        ep_r_val = min(1.0, delta / (-d_r));
                    } else if (d_l < 0.0 && d_r <= 0.0) {
                        ep_l_val = min(1.0, delta / d_l);
                        ep_r_val = 1.0;
                    } else if (d_l < 0.0 && d_r > 0.0) {
                        if (p >= 0) {
                            ep_l_val = 1.0;
                            ep_r_val = 1.0;
                        } else {
                            ep_l_val = delta / (d_l - d_r);
                            ep_r_val = delta / (d_l - d_r);
                        }
                    }
                    return {ep_l_val, ep_r_val};
                };

                // Get fluxes at left and right interfaces
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

                // limiters for this cell and its two neighbors along the active axis
                Kokkos::pair<double, double> ep_c = compute_ep(i, j, iv, jv);
                Kokkos::pair<double, double> ep_lo, ep_hi;
                if (axis == 0) {
                    ep_lo = compute_ep(i - 1, j, iv, jv);
                    ep_hi = compute_ep(i + 1, j, iv, jv);
                } else if (axis == 1) {
                    ep_lo = compute_ep(i, j - 1, iv, jv);
                    ep_hi = compute_ep(i, j + 1, iv, jv);
                } else if (axis == 2) {
                    ep_lo = compute_ep(i, j, iv - 1, jv);
                    ep_hi = compute_ep(i, j, iv + 1, jv);
                } else {
                    ep_lo = compute_ep(i, j, iv, jv - 1);
                    ep_hi = compute_ep(i, j, iv, jv + 1);
                }

                double ep_left = 0.0, ep_right = 0.0;
                if (axis == 0) {
                    ep_left  = (eta_l < 0.0) ? ep_c.first : min(ep_lo.second, ep_c.first);
                    ep_right = (eta_r < 0.0) ? ep_c.second : min(ep_c.second, ep_hi.first);
                } else if (axis == 1) {
                    ep_left  = (eta_b < 0.0) ? ep_c.first : min(ep_lo.second, ep_c.first);
                    ep_right = (eta_t < 0.0) ? ep_c.second : min(ep_c.second, ep_hi.first);
                } else {
                    ep_left  = min(ep_lo.second, ep_c.first);
                    ep_right = min(ep_c.second, ep_hi.first);
                }

                double flux_hat_l = ep_left * (flux_left - flux_1st_left) + flux_1st_left;
                double flux_hat_r = ep_right * (flux_right - flux_1st_right) + flux_1st_right;
                df(i, j, iv, jv)  = flux_hat_l - flux_hat_r;
            });

        // Step 3: Apply the increment. f is read/written only for its own cell here, so
        // there is no cross-cell hazard.
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                double eta = eta_field(i, j);
                if (eta < 0.0)
                    return;

                f(i, j, iv, jv, sp) += df(i, j, iv, jv);

                // integrity check: a correct MPP update should never produce a
                // meaningfully negative distribution in a fluid cell
                if (eta > 0.0 && f(i, j, iv, jv, sp) < -1e-15 &&
                    Kokkos::atomic_compare_exchange(&report_claimed(), 0, 1) == 0) {
                    auto [x, y, vx, vy]     = grid.center(i, j, iv, jv, sp);
                    auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                    double f_old            = f(i, j, iv, jv, sp) - df(i, j, iv, jv);
                    Kokkos::printf("\nWarning: Negative distribution\n"
                                   "  sp = %d, axis = %d\n"
                                   "  i = %d, j = %d, iv = %d, jv = %d\n",
                                   sp, axis, i, j, iv, jv);
                    Kokkos::printf("  x = %e, y = %e, vx = %e, vy = %e\n"
                                   "  f = %e, f_old = %e, df = %e\n",
                                   x, y, vx, vy, f(i, j, iv, jv, sp), f_old, df(i, j, iv, jv));
                    Kokkos::abort("Abort: Negative distribution");
                }

                // fix any negative value due to numerical error
                f(i, j, iv, jv, sp) = max(f(i, j, iv, jv, sp), 0.0);
            });
    }

    /**
     * GCIB-PFC update along one axis (papers/2026VlasovIBM.pdf, Sections 3.1.1-3.1.2).
     *
     * f_i^{n+1} = f_i^n + F_{i-1/2} - F_{i+1/2},  F = F^L + alpha * theta_j * (F^H - F^L),       Eqs. (19), (23)
     *   F^L  : first-order flux, whole swept cells + r f_j,                                         Eq. (22)
     *   F^H  : third-order PFC flux,                                                                Eq. (21)
     *   theta_j : Zhang-Shu scaling limiter of the reconstruction in the upwind cell j,          Eqs. (24)-(25)
     *   alpha   : Xiong face limiter, used only on faces between a plasma and a ghost cell,       Eqs. (32)-(39)
     * At a plasma|ghost face whose upwind cell is the ghost cell with mu = v.n_I >= 0, the flux is zero
     * (no particles enter the plasma from the object). The update keeps f within [0, f_inf], where f_inf
     * is the maximum of f over the plasma, ghost and outer ghost cells.
     *
     * Requires extrapolate_distribution_3rd_order() to have filled the ghost cells. Along x and y the
     * displacement |v dt/dx| must stay below one cell next to the interface.
     */
    void pfc_update(double dt, int axis, int sp) const {
        using Kokkos::abs;
        using Kokkos::floor;
        using Kokkos::max;
        using Kokkos::min;

        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& flux              = world.flux;     // limited high-order correction dF_{k+1/2} = theta_j (F^H - F^L)
        auto& flux_1st          = world.flux_1st; // first-order flux F^L_{k+1/2}
        auto& df                = world.df;
        auto& m                 = world.m;
        auto& q                 = world.q;
        auto cell_type          = this->cell_type;
        auto ghost_id           = this->ghost_id;
        auto ghost_normal       = this->ghost_normal;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        const bool spatial      = axis < 2;

        if (spatial && n_ghost > 0) {
            auto o      = grid.origin[sp];
            auto L      = grid.size[sp];
            double vmax = max(abs(o[axis + 2]), abs(o[axis + 2] + L[axis + 2]));
            if (vmax * dt / grid.spacing(sp)[axis] >= 1.0) {
                static bool warned = false;
                if (!warned)
                    Kokkos::printf("(VlasovSolver) WARNING: |v dt/dx| >= 1 along axis %d, the GCIB-PFC scheme "
                                   "requires a displacement below one cell next to the interface\n",
                                   axis);
                warned = true;
            }
        }

        // upper bound f_inf of Eqs. (25) and (35)
        double f_inf = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, double& lmax) {
                if (cell_type(i, j) != IB_OBJECT)
                    lmax = max(lmax, f(i, j, iv, jv, sp));
            },
            Kokkos::Max<double>(f_inf));

        Kokkos::deep_copy(flux, 0.0);
        Kokkos::deep_copy(flux_1st, 0.0);

        // Step 1: F^L and dF at the face k+1/2 between cells k and k+1 along `axis`, k = index of (i,j,iv,jv)
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc - 1, ngc - 1, ngc - 1, ngc - 1}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy]     = grid.center(i, j, iv, jv, sp);
                auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                double adv              = (axis == 0)   ? vx * dt / dx
                                          : (axis == 1) ? vy * dt / dy
                                          : (axis == 2) ? q[sp] / m[sp] * E(i, j, 0) * dt / dvx
                                                        : q[sp] / m[sp] * E(i, j, 1) * dt / dvy;
                int k                   = (axis == 0) ? i : (axis == 1) ? j : (axis == 2) ? iv : jv;
                int ext                 = f.extent_int(axis);
                // f along the line, zero outside the array (vacuum)
                auto F = [&](int n) -> double {
                    if (n < 0 || n >= ext)
                        return 0.0;
                    return (axis == 0)   ? f(n, j, iv, jv, sp)
                           : (axis == 1) ? f(i, n, iv, jv, sp)
                           : (axis == 2) ? f(i, j, n, jv, sp)
                                         : f(i, j, iv, n, sp);
                };

                // v dt/dx = +-(Nc + r): Nc whole cells and a fraction r of the upwind cell jc
                double a  = abs(adv);
                int Nc    = static_cast<int>(floor(a));
                double r  = a - Nc;
                int jc    = (adv >= 0.0) ? k - Nc : k + 1 + Nc;
                double fl = 0.0; // whole swept cells
                if (adv >= 0.0)
                    for (int n = jc + 1; n <= k; ++n)
                        fl += F(n);
                else
                    for (int n = k + 1; n <= jc - 1; ++n)
                        fl += F(n);
                double fj  = F(jc);
                double fjp = F(jc + 1);
                double fjm = F(jc - 1);
                // F^H - F^L: integral of p_j - f_j over the swept part of cell jc, Eq. (21)
                double d_down = (adv >= 0.0) ? fjp - fj : fjm - fj;
                double d_up   = (adv >= 0.0) ? fj - fjm : fj - fjp;
                double corr   = r * ((1.0 - r) * (2.0 - r) / 6.0 * d_down + (1.0 - r) * (1.0 + r) / 6.0 * d_up);
                double theta  = scaling_limiter(fjm, fj, fjp, f_inf);
                double sign   = (adv >= 0.0) ? 1.0 : -1.0;
                double FL     = sign * (fl + r * fj);
                double dF     = sign * theta * corr;

                // zero inflow from the object: plasma|ghost face with the ghost cell upwind and mu >= 0
                if (spatial) {
                    int up = (adv >= 0.0) ? k : k + 1;
                    int dn = (adv >= 0.0) ? k + 1 : k;
                    if (up >= 0 && dn >= 0 && up < ext && dn < ext) {
                        int Iu = (axis == 0) ? up : i, Ju = (axis == 0) ? j : up;
                        int Id = (axis == 0) ? dn : i, Jd = (axis == 0) ? j : dn;
                        if (cell_type(Iu, Ju) == IB_GHOST && cell_type(Id, Jd) == IB_PLASMA) {
                            int g = ghost_id(Iu, Ju);
                            if (vx * ghost_normal(g, 0) + vy * ghost_normal(g, 1) >= 0.0) {
                                FL = 0.0;
                                dF = 0.0;
                            }
                        }
                    }
                }
                flux_1st(i, j, iv, jv) = FL;
                flux(i, j, iv, jv)     = dF;
            });

        // Step 2: limited update increment of every plasma cell. Reads f/fluxes only and writes df, so no
        // thread sees a neighbour that was already updated.
        Kokkos::deep_copy(df, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                if (cell_type(i, j) != IB_PLASMA)
                    return;
                int c   = (axis == 0) ? i : (axis == 1) ? j : (axis == 2) ? iv : jv;
                int ext = f.extent_int(axis);
                auto F  = [&](int n) -> double {
                    return (axis == 0)   ? f(n, j, iv, jv, sp)
                           : (axis == 1) ? f(i, n, iv, jv, sp)
                           : (axis == 2) ? f(i, j, n, jv, sp)
                                         : f(i, j, iv, n, sp);
                };
                auto FL = [&](int n) -> double { // F^L at face n+1/2
                    return (axis == 0)   ? flux_1st(n, j, iv, jv)
                           : (axis == 1) ? flux_1st(i, n, iv, jv)
                           : (axis == 2) ? flux_1st(i, j, n, jv)
                                         : flux_1st(i, j, iv, n);
                };
                auto DF = [&](int n) -> double { // dF at face n+1/2
                    return (axis == 0)   ? flux(n, j, iv, jv)
                           : (axis == 1) ? flux(i, n, iv, jv)
                           : (axis == 2) ? flux(i, j, n, jv)
                                         : flux(i, j, iv, n);
                };
                auto type = [&](int n) -> int { // IBCellType of cell n along the line
                    if (n < ngc || n >= ext - ngc)
                        return IB_OUTER;
                    return (axis == 0) ? cell_type(n, j) : (axis == 1) ? cell_type(i, n) : IB_PLASMA;
                };
                // largest alpha admitted by cell n for corrections that raise (Lp) and lower (Lm) f_n, Eq. (38)
                auto admitted = [&](int n, double& Lp, double& Lm) {
                    Lp = Lm = 1.0;
                    if (type(n) != IB_PLASMA)
                        return; // ghost and outer cells are not updated
                    double dl = DF(n - 1), dr = DF(n);
                    double fL = F(n) + FL(n - 1) - FL(n); // first-order update, Eq. (32)
                    double Dp = max(dl, 0.0) + max(-dr, 0.0);
                    double Dm = max(-dl, 0.0) + max(dr, 0.0);
                    if (Dp > 0.0)
                        Lp = Kokkos::clamp((f_inf - fL) / Dp, 0.0, 1.0);
                    if (Dm > 0.0)
                        Lm = Kokkos::clamp(fL / Dm, 0.0, 1.0);
                };
                // alpha at face n+1/2, Eq. (39); only applied on faces touching a ghost cell
                auto alpha = [&](int n) -> double {
                    double d = DF(n);
                    if (!spatial || d == 0.0 || (type(n) != IB_GHOST && type(n + 1) != IB_GHOST))
                        return 1.0;
                    double Lp0, Lm0, Lp1, Lm1;
                    admitted(n, Lp0, Lm0);
                    admitted(n + 1, Lp1, Lm1);
                    return (d > 0.0) ? min(Lm0, Lp1) : min(Lp0, Lm1);
                };

                double flux_l    = FL(c - 1) + alpha(c - 1) * DF(c - 1);
                double flux_r    = FL(c) + alpha(c) * DF(c);
                df(i, j, iv, jv) = flux_l - flux_r;
            });

        // Diagnostic guard, see pfc_update_deprecated()
        Kokkos::View<int> report_claimed("pfc_report_claimed");

        // Step 3: apply the increment
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                if (cell_type(i, j) != IB_PLASMA)
                    return;

                f(i, j, iv, jv, sp) += df(i, j, iv, jv);

                // integrity check: the update must stay in [0, f_inf] up to round-off
                if (f(i, j, iv, jv, sp) < -1e-13 * f_inf &&
                    Kokkos::atomic_compare_exchange(&report_claimed(), 0, 1) == 0) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);
                    double f_old        = f(i, j, iv, jv, sp) - df(i, j, iv, jv);
                    Kokkos::printf("\nWarning: Negative distribution\n"
                                   "  sp = %d, axis = %d\n"
                                   "  i = %d, j = %d, iv = %d, jv = %d\n",
                                   sp, axis, i, j, iv, jv);
                    Kokkos::printf("  x = %e, y = %e, vx = %e, vy = %e\n"
                                   "  f = %e, f_old = %e, df = %e\n",
                                   x, y, vx, vy, f(i, j, iv, jv, sp), f_old, df(i, j, iv, jv));
                    Kokkos::abort("Abort: Negative distribution");
                }

                // remove round-off below zero
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
        extrapolate_distribution_3rd_order();
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
        extrapolate_distribution_3rd_order();
        Kokkos::printf("(VlasovSolver) PFC update along space by dt/2\n");
        for (int sp = 0; sp < N; ++sp) {
            pfc_update(dt / 2.0, 0, sp);
            pfc_update(dt / 2.0, 1, sp);
        }
        Kokkos::printf("(VlasovSolver) Applying Particle BC\n");
        world.particle_boundary_conditions();
        Kokkos::printf("(VlasovSolver) IB Extrapolation\n");
        extrapolate_distribution_3rd_order();
        Kokkos::printf("(VlasovSolver) Computing charge density\n");
        compute_charge_density();
    }

    void solve() {
        Kokkos::printf("Step %zu:\n", 0);
        world.initialize_distribution();
        world.particle_boundary_conditions();
        extrapolate_distribution_3rd_order();
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
