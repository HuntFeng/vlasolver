#include "full/grid.hpp"
#include "full/poisson.hpp"
#include "full/vlasov.hpp"
#include "full/world.hpp"
#include "full/writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    Kokkos::View<double*> E_w;

    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {
        E_w = Kokkos::View<double*>("E_w", grid.ncells[0]);
        Kokkos::deep_copy(E_w, 0.0);
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return x + 1.0; }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const { return {1.0, 0.0}; }

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

        // all quantities are normalized by electron parameters
        double phi_w  = -log(sqrt(m[1] / (2 * pi * m[0])));
        double v_th_e = sqrt(T[0] / m[0]); // electron thermal velocity
        double v_th_i = sqrt(T[1] / m[1]); // ion thermal velocity
        double u0     = sqrt(T[0] / m[1]); // Bohm velocity

        // initialize potential profile to a linear function from phi_w at the bottom to 0 at the top
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // linear profile
                // phi(i, j) = phi_w - phi_w * (j - ngc) / (ny - ngc);

                // exponential profile
                double y  = grid.center({i, j, 0, 0}, 0)[1];
                phi(i, j) = -exp(-y / 0.2);
            });

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
    };

    void particle_boundary_conditions() {
        using Kokkos::exp;
        using Kokkos::log;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double phi_w            = -log(sqrt(m[1] / (2 * pi * m[0])));
        double v_th_e           = sqrt(T[0] / m[0]); // electron thermal velocity
        double v_th_i           = sqrt(T[1] / m[1]); // ion thermal velocity
        double u0               = sqrt(T[0] / m[1]); // Bohm velocity

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

    void poisson_jump_conditions() {
        // skip, since no jump conditions for immersed boundary
    }

    void potential_boundary_conditions() {
        using Kokkos::abs;
        using Kokkos::log;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double phi_w            = -log(sqrt(m[1] / (2 * pi * m[0])));
        double v_th_e           = sqrt(T[0] / m[0]);  // electron thermal velocity
        double v_th_i           = sqrt(T[1] / m[1]);  // ion thermal velocity
        double u0               = sqrt(T[0] / m[1]);  // Bohm velocity
        double dy               = grid.spacing[0][1]; // species does not matter here

        // top boundary, dirichlet
        Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::ALL, Kokkos::make_pair(ny - ngc, ny)), 0.0);
        // bottom boundary, floating potential
        Kokkos::parallel_for(
            Kokkos::RangePolicy(0, nx), KOKKOS_CLASS_LAMBDA(const int i) {
                double flux = 0.0;

                // estimate the floating potential by balancing the fluxes
                flux += q[0] * n(i, ngc, 0) * v_th_e / sqrt(2 * pi);
                flux += q[1] * n(i, ngc, 1) * (v_th_i / sqrt(2 * pi) + u0);

                // compute flux from distribution function
                // for (int sp = 0; sp < 2; ++sp) {
                //     auto [dx, dy, dvx, dvy] = grid.spacing[sp];
                //     for (int iv = ngc; iv < nvx - ngc; ++iv) {
                //         for (int jv = ngc; jv < nvy - ngc; ++jv) {
                //             auto [x, y, vx, vy] = grid.center({i, ngc, iv, jv}, sp);
                //             if (vy >= 0.0)
                //                 continue;
                //             flux += q[sp] * abs(vy) * f(i, ngc, iv, jv, sp) * dvx * dvy;
                //         }
                //     }
                // }

                E_w(i) += flux * dt;
                // E_w(i) = -50.0;
                // Ey = (flux_i - flux_e) = -dphi/dy
                for (int j = 0; j < ngc; ++j) {
                    phi(i, j) = phi(i, ngc + 1) + E_w(i) * 2 * dy;
                    // phi(i, j) = phi_w;
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

    Grid grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
    grid.set_grid({x_min_e, y_min_e, vx_min_e, vy_min_e}, {Lx_e, Ly_e, Lvx_e, Lvy_e}, 0); // electrons
    grid.set_grid({x_min_i, y_min_i, vx_min_i, vy_min_i}, {Lx_i, Ly_i, Lvx_i, Lvy_i}, 1); // electrons
    ImmersedWorld world(grid);

    world.dt          = dt;          // time step size
    world.total_time  = total_time;  // total simulation time
    world.total_steps = total_steps; // number of total_steps
    world.diag_steps  = diag_steps;  // number of steps between diagnostics
    // world.m           = Kokkos::Array<double, 2>{1.0, 1836.0};     // relative mass of electrons and ions
    world.m = Kokkos::Array<double, 2>{1.0, 100.0};      // relative mass of electrons and ions
    world.q = Kokkos::Array<double, 2>{-1.0, 1.0};       // charge number of electrons and ions
    world.T = Kokkos::Array<double, 2>{1.0, 1.0 / 10.0}; // relative temperature of electrons and ions

    PoissonSolver poisson_solver(world, 1e-6, 5e3);
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi", "fi", "fe"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
