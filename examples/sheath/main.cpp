#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include "grid.hpp"
#include "poisson_1st_order.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld, 2, ElectronModel::Kinetic> {
    // all quantities are normalized by electron parameters. Species constants are
    // defined here (not read from the base m/q/T defaults) so quantities computed
    // in the member initializers below use the actual masses at construction time.
    static constexpr double me = 1.0;        // electron mass (normalization)
    static constexpr double mi = 2 * 1836.0; // ion mass (deuterium), normalized to me
    static constexpr double Te = 1.0;        // electron temperature (normalization)
    static constexpr double Ti = 0.1;        // ion temperature, normalized to Te

    double E_w    = 0.0;                                                             // wall electric field
    double phi_w  = -Kokkos::log(Kokkos::sqrt(mi / (2 * Kokkos::numbers::pi * me))); // wall potential
    double v_th_e = Kokkos::sqrt(Te / me);                                           // electron thermal velocity
    double v_th_i = Kokkos::sqrt(Ti / mi);                                           // ion thermal velocity
    double u0     = Kokkos::sqrt(Te / mi);                                           // Bohm velocity

    ImmersedWorld(Grid<2>& grid)
        : World<ImmersedWorld, 2, ElectronModel::Kinetic>(grid) {
        construct_surface();      // fill eta
        construct_permittivity(); // fill eps
        construct_normal_field(); // base method, reads eta
    }

    // Fill the surface field eta(i,j) = S(x,y) over the full domain (including ghost cells).
    void construct_surface() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y]     = grid.center(i, j);
                eta(i, j) = y + 1.0;
            });
    }

    // Uniform permittivity.
    void construct_permittivity() {
        auto [nx, ny, nvx, nvy] = this->grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                eps_p(i, j) = 1.0;
                eps_m(i, j) = 1.0;
            });
    }

    void initialize_distribution() {
        using Kokkos::abs;
        using Kokkos::exp;
        using Kokkos::log;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        // Initialize potential with Debye-shielded profile for faster relaxation
        {
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                    auto [x, y]     = grid.center(i, j);
                    phi(i, j) = phi_w * Kokkos::exp(-y / 2.5);
                });
        }

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // electron
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    double v_ce         = sqrt(2 * (phi(i, j) - phi_w) / m[0]);
                    f(i, j, iv, jv, 0) =
                        (vy <= v_ce)
                            ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                  (2.0 * pi * pow(v_th_e, 2))
                            : 0.0;
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    double v_ci         = -sqrt(2 * abs(phi(i, j)) / m[1]); // ion cutoff velocity
                    f(i, j, iv, jv, 1) =
                        (vy <= v_ci) ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
                                           (2.0 * pow(v_th_i, 2))) /
                                           (2.0 * pi * pow(v_th_i, 2))
                                     : 0.0;
                };
            });

        // periodic boundary conditions for left and right boundaries
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(0, ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL));
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    };

    void particle_boundary_conditions() {
        using Kokkos::abs;
        using Kokkos::exp;
        using Kokkos::log;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // electron
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    double v_ce         = sqrt(2 * (0.0 - phi_w) / m[0]); // electron cutoff velocity
                    if (j < ngc && vy > 0.0) {
                        f(i, j, iv, jv, 0) = 0.0; // bottom boundary, zero-inflow
                    } else if (j >= ny - ngc) {
                        // dynamic electron density adjustment
                        double ne = (n(i, ny - ngc - 1, 0) > 0.0) ? n(i, ny - ngc - 1, 1) / n(i, ny - ngc - 1, 0) : 1.0;
                        f(i, j, iv, jv, 0) =
                            (vy <= v_ce) ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                               (2.0 * pi * pow(v_th_e, 2)) * ne
                                         : 0.0;
                    }
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    double v_ci         = 0.0; // ion cutoff velocity, since phi(y=Lx) = 0
                    if (j < ngc && vy > 0.0) {
                        f(i, j, iv, jv, 1) = 0.0; // bottom boundary, zero-inflow
                    } else if (j >= ny - ngc) {
                        f(i, j, iv, jv, 1) = (vy <= v_ci)
                                                 ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
                                                       (2.0 * pow(v_th_i, 2))) /
                                                       (2.0 * pi * pow(v_th_i, 2))
                                                 : 0.0;
                    }
                };
            });

        // periodic boundary conditions for left and right boundaries
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(0, ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL));
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    };

    // No jump in the potential or its normal derivative across the interface.
    void poisson_jump_conditions() {
        auto [nx, ny, nvx, nvy] = this->grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                jump_a(i, j) = 0.0;
                jump_b(i, j) = 0.0;
            });
    }

    void potential_boundary_conditions() {
        using Kokkos::abs;
        using Kokkos::exp;
        using Kokkos::log;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dy               = grid.spacing(0, 0)[1];

        // bottom boundary, floating potential
        // electron flux to wall: Boltzmann approximation ne * v_th_e / sqrt(2π)
        int nx_mid        = nx / 2;
        auto phi_mid      = Kokkos::subview(phi, nx_mid, ngc);
        auto phi_mid_host = Kokkos::create_mirror_view(phi_mid);
        Kokkos::deep_copy(phi_mid_host, phi_mid);
        double flux_e = exp(phi_mid_host()) * v_th_e / sqrt(2 * pi);

        // ion flux: conserved from source (Bohm flux, n_0 * u0 ≈ u0)
        double flux_i = u0;
        E_w += (flux_i - flux_e) * dt;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // need to make sure the left and right boundaries are periodic
                // must check this before checking top and bottom boundaries
                if (i < ngc || i >= nx - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Periodic);
                else if (j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0);
                else if (j < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, -E_w);
                else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None);
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
    std::string output_folder = reader.Get("output", "folder", "data/plasma_past_charged_cylinder");
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

    double Te     = ImmersedWorld::Te;     // electron temperature
    double Ti     = ImmersedWorld::Ti;     // ion temperature normalized to Te
    double me     = ImmersedWorld::me;     // electron mass
    double mi     = ImmersedWorld::mi;     // ion mass, normalized to me
    double v_th_e = Kokkos::sqrt(Te / me); // electron thermal velocity
    double v_th_i = Kokkos::sqrt(Ti / mi); // ion thermal velocity, normalized to v_th_e
    double u0     = Kokkos::sqrt(Te / mi); // Bohm velocity, normalized to v_th_e

    Grid<2> grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
    grid.set_grid({x_min_e, y_min_e, vx_min_e, vy_min_e}, {Lx_e, Ly_e, Lvx_e, Lvy_e}, 0); // electrons
    grid.set_grid({x_min_i, y_min_i, vx_min_i * v_th_i, vy_min_i * v_th_i},
                  {Lx_i, Ly_i, Lvx_i * v_th_i, Lvy_i * v_th_i}, 1); // ions

    ImmersedWorld world(grid);
    world.dt          = dt;                                  // time step size
    world.total_time  = total_time;                          // total simulation time
    world.total_steps = total_steps;                         // number of total_steps
    world.diag_steps  = diag_steps;                          // number of steps between diagnostics
    world.m           = Kokkos::Array<double, 2>{me, mi};    // relative mass of electrons and ions
    world.q           = Kokkos::Array<double, 2>{-1.0, 1.0}; // charge number of electrons and ions
    world.T           = Kokkos::Array<double, 2>{Te, Ti};    // relative temperature of electrons and ions
    world.species_names = {"e", "i"};                        // electron (sp0), ion (sp1)
    world.v_th_e      = v_th_e;
    world.v_th_i      = v_th_i;
    world.u0          = u0;

    PoissonSolver1stOrder poisson_solver(world, 1e-6);
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi", "fi", "fe"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
