#include "full/grid.hpp"
#include "full/poisson.hpp"
#include "full/vlasov.hpp"
#include "full/world.hpp"
#include "full/writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    // all quantities are normalized by electron parameters
    double E_w    = 0.0;                                                                 // wall electric field
    double phi_w  = -Kokkos::log(Kokkos::sqrt(m[1] / (2 * Kokkos::numbers::pi * m[0]))); // wall potential
    double v_th_e = Kokkos::sqrt(T[0] / m[0]);                                           // electron thermal velocity
    double v_th_i = Kokkos::sqrt(T[1] / m[1]);                                           // ion thermal velocity
    double u0     = Kokkos::sqrt(T[0] / m[1]);                                           // Bohm velocity

    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {
        load_initial_potential();
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return y + 1.0; }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const { return {0.0, 1.0}; }

    void load_initial_potential() {
        Kokkos::printf("Loading initial potential from initial_potential.csv\n");
        std::ifstream file("examples/sheath/initial_potential.csv");
        std::string line;
        int j = 0;
        while (std::getline(file, line)) {
            try {
                double value = std::stod(line);
                Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::ALL, j), value);
                j++;
            } catch (const std::invalid_argument& e) {
                // Handle the case where conversion fails
                Kokkos::printf("Invalid line in initial_potential.csv: %s\n", line.c_str());
            }
        }
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

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // electron
                {
                    auto [x, y, vx, vy] = grid.center({i, j, iv, jv}, 0);
                    double v_ce         = sqrt(2 * (phi(i, j) - phi_w) / m[0]);
                    f(i, j, iv, jv, 0)  = (vy <= v_ce)
                                              ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                                   (2.0 * pi * pow(v_th_e, 2))
                                              : 0.0;
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center({i, j, iv, jv}, 1);
                    double v_ci         = -sqrt(2 * abs(phi(i, j)) / m[1]); // ion cutoff velocity
                    f(i, j, iv, jv, 1)  = (vy <= v_ci)
                                              ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
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
                    auto [x, y, vx, vy] = grid.center({i, j, iv, jv}, 0);
                    double v_ce         = sqrt(2 * (phi(i, j) - phi_w) / m[0]); // electron cutoff velocity
                    if (j < ngc && vy > 0.0) {
                        f(i, j, iv, jv, 0) = 0.0; // bottom boundary, zero-inflow
                    } else if (j >= ny - ngc) {
                        f(i, j, iv, jv, 0) =
                            (vy <= v_ce) ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                               (2.0 * pi * pow(v_th_e, 2))
                                         : 0.0;
                    }
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center({i, j, iv, jv}, 1);
                    double v_ci         = -sqrt(2 * abs(phi(i, j)) / m[1]); // ion cutoff velocity
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

    void poisson_jump_conditions() {}

    void potential_boundary_conditions() {
        using Kokkos::abs;
        using Kokkos::log;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dy               = grid.spacing[0][1]; // species does not matter here

        // top boundary, dirichlet
        Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::ALL, Kokkos::make_pair(ny - ngc, ny)), 0.0);
        // bottom boundary, floating potential
        int nx_mid = nx / 2;
        // auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
        // double flux_e = exp(phi_host(nx_mid, ngc)) * v_th_e / sqrt(2 * Kokkos::numbers::pi);
        // double flux_i = 1.0 / sqrt(1 - 2.0 * phi_host(nx_mid, ngc)) * u0;
        // E_w += (flux_i - flux_e) * dt;

        double dE_w = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy({ngc, ngc}, {nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int iv, const int jv, double& local_E_w) {
                {
                    // electron
                    auto [x, y, vx, vy]     = grid.center({nx_mid, ngc, iv, jv}, 0);
                    auto [dx, dy, dvx, dvy] = grid.spacing[0];
                    double v_ce             = sqrt(2 * (phi(nx_mid, ngc) - phi_w) / m[0]);
                    double f_e              = (vy <= v_ce)
                                                  ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(nx_mid, ngc)) /
                                           (2.0 * pi * pow(v_th_e, 2))
                                                  : 0.0;
                    local_E_w += (vy < 0.0) ? -abs(vy) * f_e * dvx * dvy * dt : 0.0;
                };
                {
                    // ion
                    auto [x, y, vx, vy]     = grid.center({nx_mid, ngc, iv, jv}, 1);
                    auto [dx, dy, dvx, dvy] = grid.spacing[1];
                    double v_ci             = -sqrt(2 * abs(phi(nx_mid, ngc)) / m[1]); // ion cutoff velocity
                    double f_i = (vy <= v_ci) ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
                                                    (2.0 * pow(v_th_i, 2))) /
                                                    (2.0 * pi * pow(v_th_i, 2))
                                              : 0.0;
                    local_E_w += (vy < 0.0) ? abs(vy) * f_i * dvx * dvy * dt : 0.0;
                };
            },
            dE_w);
        E_w += dE_w;
        Kokkos::parallel_for(
            Kokkos::RangePolicy(0, nx), KOKKOS_CLASS_LAMBDA(const int i) {
                // Ey = (flux_i - flux_e) = -dphi/dy
                for (int j = 0; j < ngc; ++j) {
                    phi(i, j) = phi(i, ngc + 1) + E_w * 2 * dy;
                }
            });

        // left and right boundary, periodic
        Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::make_pair(0, ngc), Kokkos::ALL),
                          Kokkos::subview(phi, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL));
        Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL),
                          Kokkos::subview(phi, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL));
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

    double Te     = 1.0;                   // electron temperature
    double Ti     = 0.1;                   // ion temperature normalized to Te
    double me     = 1.0;                   // electron mass
    double mi     = 2 * 1836.0;            // ion mass, normalized to me
    double v_th_e = Kokkos::sqrt(Te / me); // electron thermal velocity
    double v_th_i = Kokkos::sqrt(Ti / mi); // ion thermal velocity, normalized to v_th_e
    double u0     = Kokkos::sqrt(Te / mi); // Bohm velocity, normalized to v_th_e

    Grid grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
    grid.set_grid({x_min_e, y_min_e, vx_min_e, vy_min_e}, {Lx_e, Ly_e, Lvx_e, Lvy_e}, 0); // electrons
    grid.set_grid({x_min_i * v_th_i, y_min_i * v_th_i, vx_min_i * v_th_i, vy_min_i * v_th_i},
                  {Lx_i * v_th_i, Ly_i * v_th_i, Lvx_i * v_th_i, Lvy_i * v_th_i}, 1); // ions

    ImmersedWorld world(grid);
    world.dt          = dt;                                  // time step size
    world.total_time  = total_time;                          // total simulation time
    world.total_steps = total_steps;                         // number of total_steps
    world.diag_steps  = diag_steps;                          // number of steps between diagnostics
    world.m           = Kokkos::Array<double, 2>{me, mi};    // relative mass of electrons and ions
    world.q           = Kokkos::Array<double, 2>{-1.0, 1.0}; // charge number of electrons and ions
    world.T           = Kokkos::Array<double, 2>{Te, Ti};    // relative temperature of electrons and ions
    world.v_th_e      = v_th_e;
    world.v_th_i      = v_th_i;
    world.u0          = u0;

    PoissonSolver poisson_solver(world, 1e-6, 5e3);
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi", "fi", "fe"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    // Kokkos::Timer timer;
    // double start_time = timer.seconds();
    // // solve the potential first
    // auto& phi    = world.phi;
    // double phi_w = world.phi_w;
    // Kokkos::parallel_for(
    //     Kokkos::MDRangePolicy({0, 0}, {grid.ncells[0], grid.ncells[1]}), KOKKOS_LAMBDA(const int i, const int j) {
    //         auto [x, y, vx, vy] = grid.center({i, j, 0, 0}, 0); // species does not matter here
    //         phi(i, j)           = phi_w * Kokkos::exp(-y / 2.5);
    //     });
    world.initialize_distribution();
    world.particle_boundary_conditions();
    vlasolver.compute_charge_density();
    writer.write(0);
    // poisson_solver.solve();
    //
    // // initialize distribution using the solved potential
    // vlasolver.solve();
    // double end_time = timer.seconds();
    // Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
