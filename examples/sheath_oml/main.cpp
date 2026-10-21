#include "grid.hpp"
#include "poisson_2nd_order.hpp"
#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct ImmersedWorld : World<ImmersedWorld, 2, ElectronModel::Kinetic> {
    static constexpr double PI = Kokkos::numbers::pi;

    // ---------------------------- geometry ----------------------------
    static constexpr double R_cyl    = 1.0; // cylinder radius, in Debye lengths
    static constexpr double eps_out  = 1.0; // permittivity of the plasma (eta > 0)
    static constexpr double eps_in   = 5.0; // permittivity of the dielectric (eta < 0)

    // Number of angular bins holding the accumulated surface charge density
    // sigma(theta) on the quarter circle theta in [0, pi/2]. The physical problem
    // here is axisymmetric, so all bins carry essentially the same value; binning
    // (instead of a single scalar) keeps the setup usable for an asymmetric case,
    // and (instead of a per-cell field) averages out the cell-to-cell noise of the
    // collected flux.
    static constexpr int nbins       = 8;
    static constexpr double theta_max = PI / 2.0;

    // ---------------------------- species -----------------------------
    static constexpr double Te = 1.0;
    static constexpr double Ti = 1.0;
    static constexpr double me = 1.0;
    static constexpr double mi = 100.0;

    double v_th_e = Kokkos::sqrt(Te / me);
    double v_th_i = Kokkos::sqrt(Ti / mi);

    // 2D Maxwellian normalization: int A_s exp(-v^2 / (2 v_th_s^2)) d^2v = 1,
    // so both species start quasineutral at the reference density n_0 = 1.
    double A_e = 1.0 / (2.0 * PI * v_th_e * v_th_e);
    double A_i = 1.0 / (2.0 * PI * v_th_i * v_th_i);

    // OML profile phi(r) on a uniform grid r = oml_r0 + k * oml_dr, loaded by
    // load_oml_profile(); phi = oml_phi_d for r < oml_r0, 0 beyond the table.
    Kokkos::View<double*> oml_phi;
    double oml_r0    = 0.0;
    double oml_dr    = 1.0;
    double oml_phi_d = 0.0;

    // Density of bound ions, int_{E<0} f_i d^2v, written as the "ni_bound" field.
    Kokkos::View<double**> ni_bound;

    // Accumulated surface charge density and the per-bin collected current
    // densities (kept as diagnostics: steady state <=> je + ji == 0).
    Kokkos::View<double*> sigma;
    Kokkos::View<double*> je;
    Kokkos::View<double*> ji;
    Kokkos::View<double*> cnt;

    // guard so the time-dependent jump condition is advanced once per step
    long long last_step = -1;

    ImmersedWorld(Grid<2>& grid)
        : World<ImmersedWorld, 2, ElectronModel::Kinetic>(grid) {
        sigma = Kokkos::View<double*>("sigma", nbins);
        je    = Kokkos::View<double*>("je", nbins);
        ji    = Kokkos::View<double*>("ji", nbins);
        cnt   = Kokkos::View<double*>("cnt", nbins);

        auto [nx, ny, nvx, nvy] = grid.ncells;
        ni_bound                = Kokkos::View<double**>("ni_bound", nx, ny);
        extra_fields["ni_bound"] = ni_bound;

        construct_surface();      // fill eta
        construct_permittivity(); // fill eps
        construct_normal_field(); // base method, reads eta
    }

    // ---------------------- surface charge helpers ----------------------

    /// Angle of (x, y) folded into the simulated quarter [0, pi/2]. Points in the
    /// mirror ghost cells (x < 0 or y < 0) map onto their image.
    KOKKOS_INLINE_FUNCTION
    double theta_of(double x, double y) const { return Kokkos::atan2(Kokkos::abs(y), Kokkos::abs(x)); }

    /// Angular bin owning the point (x, y).
    KOKKOS_INLINE_FUNCTION
    int bin_of(double x, double y) const {
        double t = theta_of(x, y) / theta_max * nbins;
        return Kokkos::min(Kokkos::max((int)Kokkos::floor(t), 0), nbins - 1);
    }

    /// sigma(theta) linearly interpolated between bin centres, so that the jump
    /// condition seen by the Poisson stencil has no staircase in theta. Beyond the
    /// first/last bin centre it is held constant, which is the even extension
    /// across the symmetry planes theta = 0 and theta = pi/2.
    KOKKOS_INLINE_FUNCTION
    double sigma_at(double x, double y) const {
        double t = Kokkos::clamp(theta_of(x, y) / theta_max * nbins - 0.5, 0.0, nbins - 1.0);
        int b0   = Kokkos::min((int)Kokkos::floor(t), nbins - 2);
        double w = t - b0;
        return (1.0 - w) * sigma(b0) + w * sigma(b0 + 1);
    }

    // ------------------------- OML initial state -------------------------

    /// Read the CSV profile "r,phi,dphi_dr" written by sheath_oml.py --write-init
    /// (uniform r grid starting at the cylinder surface) and set the initial
    /// surface charge from its slope at the cylinder.
    void load_oml_profile(const std::string& path) {
        std::ifstream in(path);
        std::string header;
        if (!std::getline(in, header))
            Kokkos::abort(("cannot read OML profile " + path).c_str());
        std::vector<double> r, p, dp;
        double rk, pk, dk;
        char comma;
        while (in >> rk >> comma >> pk >> comma >> dk) {
            r.push_back(rk);
            p.push_back(pk);
            dp.push_back(dk);
        }
        if (r.size() < 2 || !isclose(r[0], R_cyl, 1e-9))
            Kokkos::abort(("OML profile must start at r = R_cyl: " + path).c_str());

        oml_r0    = r[0];
        oml_dr    = r[1] - r[0];
        oml_phi_d = p[0];
        oml_phi   = Kokkos::View<double*>("oml_phi", p.size());
        Kokkos::deep_copy(oml_phi, Kokkos::View<double*, Kokkos::HostSpace>(p.data(), p.size()));

        // no field inside the dielectric (axisymmetric), so Gauss's law at the
        // surface gives sigma = -eps_out * dphi/dr(a)
        Kokkos::deep_copy(sigma, -eps_out * dp[0]);
        Kokkos::printf("(OML init) %s: phi_d = %f, sigma = %f, r in [%f, %f]\n", path.c_str(), oml_phi_d,
                       -eps_out * dp[0], r.front(), r.back());
    }

    /// OML potential at radius r, linearly interpolated.
    KOKKOS_INLINE_FUNCTION
    double phi_oml(double r) const {
        int nr = oml_phi.extent_int(0);
        double t = (r - oml_r0) / oml_dr;
        if (t <= 0.0)
            return oml_phi_d;
        if (t >= nr - 1)
            return 0.0;
        int k = (int)Kokkos::floor(t);
        double w = t - k;
        return (1.0 - w) * oml_phi(k) + w * oml_phi(k + 1);
    }

    /// OML distribution of species sp at (x, y, vx, vy): A_s exp(-E / T_s) on the
    /// part of velocity space reachable from the reservoir (E >= 0), outgoing only
    /// if the particle is turned back before it reaches the cylinder.
    KOKKOS_INLINE_FUNCTION
    double f_oml(int sp, double x, double y, double vx, double vy, double phi_r) const {
        double r   = Kokkos::sqrt(x * x + y * y);
        double ms  = (sp == 0) ? me : mi;
        double Ts  = (sp == 0) ? Te : Ti;
        double A   = (sp == 0) ? A_e : A_i;
        double vr  = (x * vx + y * vy) / r;
        double vt  = (x * vy - y * vx) / r;
        double E   = 0.5 * ms * (vx * vx + vy * vy) + q[sp] * phi_r;
        double z   = r / R_cyl;
        double U_a = 0.5 * ms * vt * vt * z * z + q[sp] * oml_phi_d; // effective potential at r = a
        bool allowed = E >= 0.0 && (vr < 0.0 || E < U_a);
        return allowed ? A * Kokkos::exp(-E / Ts) : 0.0;
    }

    // Fill the surface field eta(i,j) = S(x,y) over the full domain (including ghost cells).
    void construct_surface() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                eta(i, j)   = x * x + y * y - R_cyl * R_cyl;
            });
    }

    void construct_permittivity() {
        auto& eps_p             = this->eps_p;
        auto& eps_m             = this->eps_m;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                eps_p(i, j) = eps_out; // permittivity in the eta>0 region (plasma)
                eps_m(i, j) = eps_in;  // permittivity in the eta<0 region (dielectric)
            });
    }

    void initialize_distribution() {
        using Kokkos::sqrt;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        if (oml_phi.extent(0) == 0)
            Kokkos::abort("load_oml_profile() must be called before the run");

        // Initial sheath potential from the OML solution. Only the prefill uses it;
        // the step-0 Poisson solve overwrites phi.
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                phi(i, j)   = phi_oml(sqrt(x * x + y * y));
            });

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                if (eta(i, j) <= 0.0) {
                    // immersed wall absorbs, emits nothing back into the plasma
                    f(i, j, iv, jv, 0) = 0.0;
                    f(i, j, iv, jv, 1) = 0.0;
                    return;
                }
                // average over the velocity cell: f_oml has sharp loss-cone
                // edges, and sampling only the cell centre leaves a striped
                // density error where the cone is narrow (large r)
                constexpr int ns = 4;
                for (int sp = 0; sp < 2; ++sp) {
                    auto [x, y, vx, vy]     = grid.center(i, j, iv, jv, sp);
                    auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                    double avg              = 0.0;
                    for (int a = 0; a < ns; ++a)
                        for (int b = 0; b < ns; ++b)
                            avg += f_oml(sp, x, y, vx + ((a + 0.5) / ns - 0.5) * dvx,
                                         vy + ((b + 0.5) / ns - 0.5) * dvy, phi(i, j));
                    f(i, j, iv, jv, sp) = avg / (ns * ns);
                }
            });
    };

    void particle_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        // immersed wall emits nothing back (interior cells only)
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                if (eta(i, j) > 0.0)
                    return;
                double n1 = normal(i, j, 0);
                double n2 = normal(i, j, 1);
                for (int sp = 0; sp < 2; ++sp) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);
                    if (vx * n1 + vy * n2 > 0.0)
                        f(i, j, iv, jv, sp) = 0.0;
                }
            });

        // Outer ghost cells. Separate launch: the mirror ghosts read interior cells
        // that the kernel above writes.
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                bool right = i >= nx - ngc, top = j >= ny - ngc;
                bool left = i < ngc, bottom = j < ngc;
                if (!(left || right || bottom || top))
                    return;
                for (int sp = 0; sp < 2; ++sp) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);
                    if (right || top) {
                        // reservoir injected on the inflow half of velocity space: the
                        // OML distribution at the ghost cell, consistent with the
                        // Dirichlet value phi_oml(r) set in potential_boundary_conditions,
                        // so the initial state is not kicked by the wall
                        bool inflow = right ? vx < 0.0 : vy < 0.0;
                        if (inflow) {
                            f(i, j, iv, jv, sp) = f_oml(sp, x, y, vx, vy, phi_oml(Kokkos::sqrt(x * x + y * y)));
                        } else {
                            // outflow: zero-gradient copy of the nearest interior cell. A
                            // zero here is a cliff the third-order PFC reconstruction sees,
                            // which cuts the outgoing flux and leaves an over/undershoot in
                            // the last interior cells. Corner ghosts that also lie beyond
                            // a symmetry plane read the mirror image of that cell.
                            int I  = right ? nx - ngc - 1 : (left ? 2 * ngc - 1 - i : i);
                            int J  = top ? ny - ngc - 1 : (bottom ? 2 * ngc - 1 - j : j);
                            int IV = (left && !right) ? nvx - 1 - iv : iv;
                            int JV = (bottom && !top) ? nvy - 1 - jv : jv;
                            f(i, j, iv, jv, sp) = f(I, J, IV, JV, sp);
                        }
                    } else {
                        // symmetry planes x = 0 / y = 0: specular reflection. The image
                        // of ghost index k is 2 ngc - 1 - k in space and the velocity
                        // component normal to the plane flips (v grids are symmetric).
                        int I  = left ? 2 * ngc - 1 - i : i;
                        int J  = bottom ? 2 * ngc - 1 - j : j;
                        int IV = left ? nvx - 1 - iv : iv;
                        int JV = bottom ? nvy - 1 - jv : jv;
                        f(i, j, iv, jv, sp) = f(I, J, IV, JV, sp);
                    }
                }
            });
    };

    /// ni_bound(i, j) = int_{E<0} f_i d^2v with E = m_i v^2 / 2 + e phi, plasma cells only.
    void compute_diagnostics() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double nb = 0.0;
                if (i >= ngc && i < nx - ngc && j >= ngc && j < ny - ngc && eta(i, j) >= 0.0) {
                    auto [dx, dy, dvx, dvy] = grid.spacing(1);
                    for (int iv = ngc; iv < nvx - ngc; ++iv)
                        for (int jv = ngc; jv < nvy - ngc; ++jv) {
                            auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                            if (0.5 * mi * (vx * vx + vy * vy) + q[1] * phi(i, j) < 0.0)
                                nb += f(i, j, iv, jv, 1) * dvx * dvy;
                        }
                }
                ni_bound(i, j) = nb;
            });
    }

    void potential_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // Reservoir on the right and top sides (this also takes the two
                // corners shared with a symmetry plane), held at the OML potential.
                // Symmetry planes x = 0 and y = 0 on the left and bottom.
                auto [x, y]     = grid.center(i, j);
                double phi_wall = phi_oml(Kokkos::sqrt(x * x + y * y));
                if (i >= nx - ngc || j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, phi_wall);
                else if (i < ngc || j < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Symmetric, 0.0);
                else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
            });
    }

    void poisson_jump_conditions() {
        using Kokkos::max;
        // advance sigma once per time step (the solver may call this more than once)
        bool advance_sigma = (long long)current_step != last_step;
        last_step          = (long long)current_step;

        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        if (advance_sigma) {
            Kokkos::deep_copy(je, 0.0);
            Kokkos::deep_copy(ji, 0.0);
            Kokkos::deep_copy(cnt, 0.0);

            // Collect the charge flux onto the surface from the first *fluid* cell
            // layer only. Cells inside the object hold extrapolated values and
            // cells far from the interface see the unperturbed distribution, so
            // neither may contribute to sigma.
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                    if (eta(i, j) <= 0.0)
                        return;
                    bool at_surface = eta(i - 1, j) <= 0.0 || eta(i + 1, j) <= 0.0 || eta(i, j - 1) <= 0.0 ||
                                      eta(i, j + 1) <= 0.0;
                    if (!at_surface)
                        return;

                    auto [x, y] = grid.center(i, j);
                    double n1   = normal(i, j, 0);
                    double n2   = normal(i, j, 1);

                    // Net charge flux towards the wall through this cell, both halves
                    // of velocity space. The cell centre sits up to ~1.5 dx off the
                    // surface, where part of the inward-moving particles are still
                    // turned back before reaching it (repelled electrons, ions with
                    // too much angular momentum); counting only n.v < 0 overestimates
                    // the electron current relative to the ion current there and
                    // shifts the floating potential. The net flux is what reaches the
                    // wall: in steady state the radial current through a circle is
                    // independent of r, so the flux density at r maps to the surface
                    // by the factor r / R_cyl (cylinder geometry).
                    double flux[2] = {0.0, 0.0};
                    double to_wall = Kokkos::sqrt(x * x + y * y) / R_cyl;
                    for (int sp = 0; sp < 2; ++sp) {
                        auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                        for (int iv = ngc; iv < nvx - ngc; ++iv) {
                            for (int jv = ngc; jv < nvy - ngc; ++jv) {
                                auto [_x, _y, vx, vy] = grid.center(i, j, iv, jv, sp);
                                double v_dot_n        = vx * n1 + vy * n2;
                                flux[sp] += q[sp] * (-v_dot_n) * f(i, j, iv, jv, sp) * dvx * dvy;
                            }
                        }
                        flux[sp] *= to_wall;
                    }

                    int b = bin_of(x, y);
                    Kokkos::atomic_add(&je(b), flux[0]);
                    Kokkos::atomic_add(&ji(b), flux[1]);
                    Kokkos::atomic_add(&cnt(b), 1.0);
                });

            // per-bin averages -> surface charge density increment (host side,
            // nbins is tiny)
            auto sigma_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sigma);
            auto je_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), je);
            auto ji_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ji);
            auto cnt_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cnt);
            std::vector<int> filled;
            for (int b = 0; b < nbins; ++b) {
                if (cnt_h(b) == 0.0)
                    continue;
                je_h(b) /= cnt_h(b);
                ji_h(b) /= cnt_h(b);
                sigma_h(b) += (je_h(b) + ji_h(b)) * dt;
                filled.push_back(b);
            }
            if (filled.empty())
                Kokkos::abort("no surface cell in any sigma bin");

            // A bin without any first-layer cell (the grid has only a handful of
            // them per quarter) would otherwise keep its initial sigma forever.
            // Interpolate it from the nearest filled bins; beyond the first/last
            // filled bin hold the value, the even extension across the symmetry
            // planes.
            for (int b = 0; b < nbins; ++b) {
                if (cnt_h(b) != 0.0)
                    continue;
                int lo = -1, hi = -1;
                for (int k : filled) {
                    if (k < b)
                        lo = k;
                    if (k > b && hi < 0)
                        hi = k;
                }
                if (lo < 0)
                    sigma_h(b) = sigma_h(hi);
                else if (hi < 0)
                    sigma_h(b) = sigma_h(lo);
                else
                    sigma_h(b) = sigma_h(lo) + (sigma_h(hi) - sigma_h(lo)) * (b - lo) / double(hi - lo);
            }
            Kokkos::deep_copy(sigma, sigma_h);

            // convergence monitor over the filled bins: steady state <=> J_e + J_i -> 0
            // and sigma flat
            double s_sum = 0.0, je_sum = 0.0, ji_sum = 0.0;
            for (int b = 0; b < nbins; ++b)
                s_sum += sigma_h(b) / nbins;
            for (int b : filled) {
                je_sum += je_h(b) / filled.size();
                ji_sum += ji_h(b) / filled.size();
            }
            Kokkos::printf("(Surface) sigma = %.6e, J_e = %.6e, J_i = %.6e, J_net = %.6e, filled bins = %zu/%d\n",
                           s_sum, je_sum, ji_sum, je_sum + ji_sum, filled.size(), nbins);
        }

        // [phi] = 0, [eps dphi/dn] = -sigma
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y]  = grid.center(i, j);
                jump_a(i, j) = 0.0;
                jump_b(i, j) = -sigma_at(x, y);
            });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_file_path>\n";
        return 1;
    }
    std::string input_file_path = argv[1];
    INIReader reader(input_file_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load input file, please specify path\n";
        return 1;
    }
    double x_min_e            = reader.GetReal("grid_electron", "x_min", 0.0);
    double y_min_e            = reader.GetReal("grid_electron", "y_min", 0.0);
    double vx_min_e           = reader.GetReal("grid_electron", "vx_min", 0.0);
    double vy_min_e           = reader.GetReal("grid_electron", "vy_min", 0.0);
    double Lx_e               = reader.GetReal("grid_electron", "Lx", 1.0);
    double Ly_e               = reader.GetReal("grid_electron", "Ly", 1.0);
    double Lvx_e              = reader.GetReal("grid_electron", "Lvx", 1.0);
    double Lvy_e              = reader.GetReal("grid_electron", "Lvy", 1.0);

    double x_min_i            = reader.GetReal("grid_ion", "x_min", 0.0);
    double y_min_i            = reader.GetReal("grid_ion", "y_min", 0.0);
    double vx_min_i           = reader.GetReal("grid_ion", "vx_min", 0.0);
    double vy_min_i           = reader.GetReal("grid_ion", "vy_min", 0.0);
    double Lx_i               = reader.GetReal("grid_ion", "Lx", 1.0);
    double Ly_i               = reader.GetReal("grid_ion", "Ly", 1.0);
    double Lvx_i              = reader.GetReal("grid_ion", "Lvx", 1.0);
    double Lvy_i              = reader.GetReal("grid_ion", "Lvy", 1.0);

    int nx_intr               = reader.GetInteger("grid", "nx", 128);
    int ny_intr               = reader.GetInteger("grid", "ny", 128);
    int nvx_intr              = reader.GetInteger("grid", "nvx", 128);
    int nvy_intr              = reader.GetInteger("grid", "nvy", 128);
    int ngc                   = reader.GetInteger("grid", "ngc", 3);
    double dt                 = reader.GetReal("world", "dt", 1e-3);
    double total_time         = reader.GetReal("world", "total_time", 1.0);
    int total_steps           = reader.GetInteger("world", "total_steps", 1000);
    int diag_steps            = reader.GetInteger("world", "diag_steps", 10);
    std::string oml_profile   = reader.Get("init", "oml_profile", "examples/sheath_oml/oml_init.csv");
    std::string output_folder = reader.Get("output", "folder", "data/sheath_oml");
    std::string output_prefix = reader.Get("output", "prefix", "output");

    Kokkos::printf("Input parameters:\n");
    Kokkos::printf("Phase space (x,y,vx,vy):\nElectron: [%f, %f, %f, %f]x[%f, %f, %f, %f]\nIon: [%f, %f, %f, %f]x[%f, "
                   "%f, %f, %f]\n",
                   x_min_e, y_min_e, vx_min_e, vy_min_e, x_min_e + Lx_e, y_min_e + Ly_e, vx_min_e + Lvx_e,
                   vy_min_e + Lvy_e, x_min_i, y_min_i, vx_min_i, vy_min_i, x_min_i + Lx_i, y_min_i + Ly_i,
                   vx_min_i + Lvx_i, vy_min_i + Lvy_i);
    Kokkos::printf("Grid size, interior (nx,ny,nvx,nvy): [%d, %d, %d, %d]\n", nx_intr, ny_intr, nvx_intr, nvy_intr);
    Kokkos::printf("Simulation control: dt: %f, total_time: %f, total_steps: %d, diag_steps: %d\n", dt, total_time,
                   total_steps, diag_steps);

    double Te     = ImmersedWorld::Te;
    double Ti     = ImmersedWorld::Ti;
    double me     = ImmersedWorld::me;
    double mi     = ImmersedWorld::mi;
    double v_th_e = Kokkos::sqrt(Te / me);
    double v_th_i = Kokkos::sqrt(Ti / mi);

    // Resolution / stability report. The spatial CFL number is quoted for the
    // half step actually used by the PFC sweeps in Vlasolver::advance().
    double dx     = Lx_e / nx_intr;
    double v_max  = Kokkos::max(Kokkos::abs(vx_min_e), Kokkos::abs(vx_min_e + Lvx_e));
    Kokkos::printf("Derived: dx = %f lambda_D, dv_e = %f v_th_e, dv_i = %f v_th_i\n", dx, Lvx_e / nvx_intr,
                   Lvx_i / nvx_intr);
    Kokkos::printf("Derived: omega_pe*dt = %f, spatial CFL = %f, ion plasma period = %f, ion transit = %f\n", dt,
                   v_max * dt / 2.0 / dx, 2.0 * Kokkos::numbers::pi * Kokkos::sqrt(mi), Lx_e / v_th_i);

    // the quarter-domain symmetry planes need x = y = 0 at the lower edges and
    // velocity grids symmetric about v = 0 (see particle_boundary_conditions)
    if (x_min_e != 0.0 || y_min_e != 0.0 || x_min_i != 0.0 || y_min_i != 0.0 || vx_min_e != -0.5 * Lvx_e ||
        vy_min_e != -0.5 * Lvy_e || vx_min_i != -0.5 * Lvx_i || vy_min_i != -0.5 * Lvy_i) {
        std::cout << "Quarter domain requires x_min = y_min = 0 and v_min = -Lv / 2 for both species\n";
        return 1;
    }

    Grid<2> grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
    grid.set_grid({x_min_e, y_min_e, vx_min_e * v_th_e, vy_min_e * v_th_e},
                  {Lx_e, Ly_e, Lvx_e * v_th_e, Lvy_e * v_th_e}, 0); // electrons
    grid.set_grid({x_min_i, y_min_i, vx_min_i * v_th_i, vy_min_i * v_th_i},
                  {Lx_i, Ly_i, Lvx_i * v_th_i, Lvy_i * v_th_i}, 1); // ions

    ImmersedWorld world(grid);
    world.dt            = dt;                                  // time step size
    world.total_time    = total_time;                          // total simulation time
    world.total_steps   = total_steps;                         // number of total_steps
    world.diag_steps    = diag_steps;                          // number of steps between diagnostics
    world.m             = Kokkos::Array<double, 2>{me, mi};    // relative mass of electrons and ions
    world.q             = Kokkos::Array<double, 2>{-1.0, 1.0}; // charge number of electrons and ions
    world.T             = Kokkos::Array<double, 2>{Te, Ti};    // relative temperature of electrons and ions
    world.species_names = {"e", "i"};                          // electron (sp0), ion (sp1)
    world.load_oml_profile(oml_profile);                       // also sets the initial surface charge

    PoissonSolver2ndOrder poisson_solver(world);
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi", "ni_bound"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
