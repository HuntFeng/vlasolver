#include "grid.hpp"
#include "poisson.hpp"
#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <INIReader.h>
#include <string>

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
    double x_min      = reader.GetReal("grid", "x_min", 0.0);
    double y_min      = reader.GetReal("grid", "y_min", 0.0);
    double vx_min     = reader.GetReal("grid", "vx_min", 0.0);
    double vy_min     = reader.GetReal("grid", "vy_min", 0.0);
    double Lx         = reader.GetReal("grid", "Lx", 1.0);
    double Ly         = reader.GetReal("grid", "Ly", 1.0);
    double Lvx        = reader.GetReal("grid", "Lvx", 1.0);
    double Lvy        = reader.GetReal("grid", "Lvy", 1.0);
    int nx            = reader.GetInteger("grid", "nx", 128);
    int ny            = reader.GetInteger("grid", "ny", 128);
    int nvx           = reader.GetInteger("grid", "nvx", 128);
    int nvy           = reader.GetInteger("grid", "nvy", 128);
    int ngc           = reader.GetInteger("grid", "ngc", 3);
    double dt         = reader.GetReal("world", "dt", 1e-3);
    double total_time = reader.GetReal("world", "total_time", 1.0);
    int total_steps   = reader.GetInteger("world", "total_steps", 1000);
    int diag_steps    = reader.GetInteger("world", "diag_steps", 10);

    Kokkos::printf("Input parameters:\n");
    Kokkos::printf("Phase space (x,y,vx,vy): [%f, %f, %f, %f]x[%f, %f, %f, %f]\n", x_min, y_min, vx_min, vy_min,
                   x_min + Lx, y_min + Ly, vx_min + Lvx, vy_min + Lvy);
    Kokkos::printf("Grid size (nx,ny,nvx,nvy): [%d, %d, %d, %d]\n", nx, ny, nvx, nvy);
    Kokkos::printf("Simulation control: dt: %f, total_time: %f, total_steps: %d, diag_steps: %d\n", dt, total_time,
                   total_steps, diag_steps);

    Kokkos::Array<double, DIM> origin       = {x_min, y_min, vx_min, vy_min}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {Lx, Ly, Lvx, Lvy};             // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx - 2 * ngc, ny - 2 * ngc, nvx - 2 * ngc,
                                               nvy - 2 * ngc}; // number of cells in the grid (excluding ghost cells)

    // int ngc                                 = 3;                        // number of ghost cells on
    // each side Kokkos::Array<double, DIM> origin       = {0.0, 0.0, -10.0, -10.0}; // origin of the
    // grid Kokkos::Array<double, DIM> size         = {1.0, 0.5, 20.0, 20.0};   // size of the grid
    // Kokkos::Array<int, DIM> ncells_interior = {160 - 2 * ngc, 50 - 2 * ngc, 120 - 2 * ngc,
    //                                            50 - 2 * ngc}; // number of cells in the grid
    //                                            (excluding ghost cells)
    Kokkos::Array<double, 2> charge_number = {-1.0, 1.0}; // charge number of the particle
    Kokkos::Array<double, 2> mass_ratio    = {1.0, 1836}; // mass ratio of the particle (relative to the electron mass)
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid, charge_number, mass_ratio);

    world.dt          = dt;          // time step size
    world.total_time  = total_time;  // total simulation time
    world.total_steps = total_steps; // number of total_steps
    world.diag_steps  = diag_steps;  // number of steps between diagnostics

    // world.dt          = 1e-3;                        // time step size
    // world.total_time  = 0.6;                         // total simulation time
    // world.total_steps = world.total_time / world.dt; // number of total_steps
    // world.diag_steps  = 10;                          // number of steps between diagnostics
    PoissonSolver poisson_solver(world);
    // poisson_solver.enable_debug();
    Writer writer(world, "data/plasma_past_charged_cylinder.hdf", {"ne", "ni", "phi", "Ex"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
