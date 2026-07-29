/*
 * Sheath around a floating dielectric cylinder immersed in a quiescent plasma.
 *
 * Normalization (see vlasov.hpp): x / lambda_D, v / v_th_e, t * omega_pe,
 * f / (n_0 / v_th_e), and laplacian(phi) = -(n_i - n_e). For lambda_D and
 * omega_pe to actually be 1 in these units the *initial* plasma must have
 * n_e = n_i = 1 and T_e = 1, which fixes the Maxwellian prefactor to
 * A_s = 1 / (2 pi v_th_s^2) (2D Maxwellian, exponent -v^2 / (2 v_th_s^2)).
 *
 * Relevant time scales for mi/me = 100, n_0 = 1, T_e = T_i = 1:
 *   electron plasma period   2 pi / omega_pe        =  6.3
 *   ion plasma period        2 pi / omega_pi        = 62.8   <-- startup ringing
 *   ion transit of a 20 lambda_D domain  L / v_th_i = 200
 *   surface charging time    |sigma_w| / Gamma_e    ~ 5
 * The surface charge, and therefore the potential, cannot reach a steady state
 * before the ion distribution has been refreshed a few times, i.e. not before
 * t ~ several hundred. Running only a fraction of an ion transit time leaves
 * the run in the middle of an undamped ion plasma oscillation (the "potential
 * oscillates up and down" symptom) plus a slow monotonic charging drift.
 * Two things are done here to shorten that transient:
 *   1. the surface charge starts at its estimated floating value sigma_0
 *      instead of 0, and the electrons are prefilled with the matching
 *      Boltzmann factor, so the abrupt t = 0 kick that rings the ions is small;
 *   2. sigma is stored per surface element and is built from the flux in the
 *      first fluid cell layer only, so the jump condition is not polluted by
 *      cells far from the interface.
 * The net collected current is printed every step: the run has converged when
 * J_e + J_i -> 0 and sigma is flat, not merely when the fields look smooth.
 */
#include "grid.hpp"
#include "poisson_2nd_order.hpp"
#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld, 2, ElectronModel::Kinetic> {
    static constexpr double PI = Kokkos::numbers::pi;

    // ---------------------------- geometry ----------------------------
    static constexpr double R_cyl    = 1.0; // cylinder radius, in Debye lengths
    static constexpr double eps_out  = 1.0; // permittivity of the plasma (eta > 0)
    static constexpr double eps_in   = 5.0; // permittivity of the dielectric (eta < 0)

    // Number of angular bins holding the accumulated surface charge density
    // sigma(theta). The physical problem here is axisymmetric, so all bins carry
    // essentially the same value; binning (instead of a single scalar) keeps the
    // setup usable for an asymmetric case, and (instead of a per-cell field)
    // averages out the cell-to-cell noise of the collected flux.
    static constexpr int nbins       = 32;

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

    // Planar floating-potential estimate, Gamma_e exp(phi_w / Te) = Gamma_i:
    // phi_w = -Te * ln(sqrt(mi / (2 pi me))) = -1.38 Te for mi / me = 100.
    // Used only to build the initial state; the simulation is free to drift away
    // from it (in cylindrical geometry the orbital-motion-limited ion collection
    // makes the true floating potential somewhat less negative).
    double phi_w    = -Te * Kokkos::log(Kokkos::sqrt(mi / (2.0 * PI * me)));
    double L_sheath = 2.0; // assumed initial sheath thickness, in Debye lengths

    // Initial surface charge. With no charge inside the dielectric and an
    // axisymmetric surface the interior field vanishes, so Gauss's law gives
    // sigma = -eps_out * dphi/dn = eps_out * phi_w / L_sheath < 0.
    double sigma_0 = eps_out * phi_w / L_sheath;

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
        Kokkos::deep_copy(sigma, sigma_0);

        construct_surface();      // fill eta
        construct_permittivity(); // fill eps
        construct_normal_field(); // base method, reads eta
    }

    // ---------------------- surface charge helpers ----------------------

    /// Angular bin owning the point (x, y).
    KOKKOS_INLINE_FUNCTION
    int bin_of(double x, double y) const {
        double t = (Kokkos::atan2(y, x) + PI) / (2.0 * PI) * nbins;
        return Kokkos::min(Kokkos::max((int)Kokkos::floor(t), 0), nbins - 1);
    }

    /// sigma(theta) linearly interpolated between bin centres, so that the jump
    /// condition seen by the Poisson stencil has no staircase in theta.
    KOKKOS_INLINE_FUNCTION
    double sigma_at(double x, double y) const {
        double t   = (Kokkos::atan2(y, x) + PI) / (2.0 * PI) * nbins - 0.5;
        int b0     = (int)Kokkos::floor(t);
        double w   = t - b0;
        b0         = ((b0 % nbins) + nbins) % nbins;
        int b1     = (b0 + 1) % nbins;
        return (1.0 - w) * sigma(b0) + w * sigma(b1);
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
        using Kokkos::exp;
        using Kokkos::pow;
        using Kokkos::sqrt;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        // Initial guess for the sheath potential, consistent with sigma_0. Only the
        // electron prefill uses it; the step-0 Poisson solve overwrites phi.
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                double r    = sqrt(x * x + y * y);
                phi(i, j)   = (eta(i, j) > 0.0) ? phi_w * exp(-(r - R_cyl) / L_sheath) : phi_w;
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
                // electron: Boltzmann-depleted Maxwellian, n_e = exp(phi / Te).
                // The absorbing-surface loss cone is deliberately *not* imposed
                // here: in cylindrical geometry the depleted region of velocity
                // space shrinks like R/r, and applying the planar (1D) loss cone
                // everywhere would remove a large part of the far-field
                // electrons and drive exactly the global oscillation this setup
                // is trying to avoid. It fills in self-consistently within a few
                // electron plasma periods.
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    f(i, j, iv, jv, 0) =
                        A_e * exp(phi(i, j) / Te) * exp(-(vx * vx + vy * vy) / (2.0 * v_th_e * v_th_e));
                }
                // ion: unperturbed Maxwellian. The ion sheath depletion develops
                // on the (fast, local) ion sheath transit time.
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    f(i, j, iv, jv, 1)  = A_i * exp(-(vx * vx + vy * vy) / (2.0 * v_th_i * v_th_i));
                }
            });
    };

    void particle_boundary_conditions() {
        using Kokkos::exp;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                double n1 = normal(i, j, 0);
                double n2 = normal(i, j, 1);
                for (int sp = 0; sp < 2; ++sp) {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, sp);
                    double v_th         = (sp == 0) ? v_th_e : v_th_i;
                    double A            = (sp == 0) ? A_e : A_i;
                    // unperturbed Maxwellian reservoir at the outer boundary
                    // (phi = 0 there), injected on the inflow half of velocity space
                    double f_inj        = A * exp(-(vx * vx + vy * vy) / (2.0 * v_th * v_th));

                    if (i < ngc)
                        f(i, j, iv, jv, sp) = (vx > 0.0) ? f_inj : 0.0; // left
                    else if (i >= nx - ngc)
                        f(i, j, iv, jv, sp) = (vx < 0.0) ? f_inj : 0.0; // right
                    else if (j < ngc)
                        f(i, j, iv, jv, sp) = (vy > 0.0) ? f_inj : 0.0; // bottom
                    else if (j >= ny - ngc)
                        f(i, j, iv, jv, sp) = (vy < 0.0) ? f_inj : 0.0; // top
                    else if (eta(i, j) <= 0.0 && vx * n1 + vy * n2 > 0.0)
                        f(i, j, iv, jv, sp) = 0.0; // immersed wall emits nothing back
                }
            });
    };

    void potential_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // Grounded reservoir on all four sides. NOTE: in 2D the potential
                // of a charged cylinder decays only like ln(r), so phi = 0 at
                // 10 lambda_D is an approximation; if the far-field potential in
                // the diagnostics is not small compared to |phi_w|, enlarge the
                // domain rather than trusting the sheath amplitude.
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0);
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

                    // charge flux density onto the wall, only the n.v < 0 half space
                    double flux[2] = {0.0, 0.0};
                    for (int sp = 0; sp < 2; ++sp) {
                        auto [dx, dy, dvx, dvy] = grid.spacing(sp);
                        for (int iv = ngc; iv < nvx - ngc; ++iv) {
                            for (int jv = ngc; jv < nvy - ngc; ++jv) {
                                auto [_x, _y, vx, vy] = grid.center(i, j, iv, jv, sp);
                                double v_dot_n        = vx * n1 + vy * n2;
                                flux[sp] += q[sp] * max(-v_dot_n, 0.0) * f(i, j, iv, jv, sp) * dvx * dvy;
                            }
                        }
                    }

                    int b = bin_of(x, y);
                    Kokkos::atomic_add(&je(b), flux[0]);
                    Kokkos::atomic_add(&ji(b), flux[1]);
                    Kokkos::atomic_add(&cnt(b), 1.0);
                });

            // per-bin averages -> surface charge density increment
            Kokkos::parallel_for(
                Kokkos::RangePolicy<>(0, nbins), KOKKOS_CLASS_LAMBDA(const int b) {
                    if (cnt(b) == 0.0)
                        return;
                    je(b) /= cnt(b);
                    ji(b) /= cnt(b);
                    sigma(b) += (je(b) + ji(b)) * dt;
                });

            // convergence monitor: steady state <=> J_e + J_i -> 0 and sigma flat
            auto sigma_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sigma);
            auto je_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), je);
            auto ji_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ji);
            double s_sum = 0.0, je_sum = 0.0, ji_sum = 0.0;
            for (int b = 0; b < nbins; ++b) {
                s_sum += sigma_h(b) / nbins;
                je_sum += je_h(b) / nbins;
                ji_sum += ji_h(b) / nbins;
            }
            Kokkos::printf("(Surface) sigma = %.6e, J_e = %.6e, J_i = %.6e, J_net = %.6e\n", s_sum, je_sum, ji_sum,
                           je_sum + ji_sum);
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
    std::string output_folder = reader.Get("output", "folder", "data/sheath_cylinder");
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

    Kokkos::printf("Derived: phi_w (planar estimate) = %f, initial sigma = %f\n", world.phi_w, world.sigma_0);

    PoissonSolver2ndOrder poisson_solver(world);
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
