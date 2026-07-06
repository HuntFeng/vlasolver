#include "grid.hpp"
#include "poisson_1st_order.hpp"
#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld, 1, ElectronModel::Boltzmann> {
    ImmersedWorld(Grid<1>& grid)
        : World<ImmersedWorld, 1, ElectronModel::Boltzmann>(grid) {
        construct_surface();      // fill eta
        construct_permittivity(); // fill eps_p / eps_m
        construct_normal_field(); // base method, reads eta
    }

    // fill the level set field `eta` over the full domain (including ghost cells)
    void construct_surface() {
        auto& grid              = this->grid;
        auto& eta               = this->eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                eta(i, j)   = Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
            });
    }

    // fill the region permittivity fields over the full domain
    void construct_permittivity() {
        auto& grid              = this->grid;
        auto& eps_p             = this->eps_p;
        auto& eps_m             = this->eps_m;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                eps_p(i, j) = 1.0;    // permittivity in the eta>0 region
                eps_m(i, j) = 1000.0; // permittivity in the eta<0 region
            });
    }

    // fill the Poisson jump condition fields (no jumps for this case)
    void poisson_jump_conditions() {
        Kokkos::deep_copy(jump_a, 0.0);
        Kokkos::deep_copy(jump_b, 0.0);
    }

    void initialize_distribution() {
        // no particles initially
    };

    void particle_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                if (i < ngc) {
                    f(i, j, iv, jv, 0) =
                        (vx > 0.0) ? exp(-pow(vx - 5, 2)) * exp(-pow(vy, 2)) : 0.0; // left boundary, injection
                } else if (i >= nx - ngc) {
                    if (vx < 0.0)
                        f(i, j, iv, jv, 0) = 0.0; // right boundary, zero-inflow
                } else if (j < ngc) {
                    f(i, j, iv, jv, 0) = f(i, 2 * ngc - j - 1, iv, nvy - jv - 1, 0); // bottom boundary, reflective
                } else if (j >= ny - ngc) {
                    f(i, j, iv, jv, 0) = f(i, 2 * (ny - ngc) - j - 1, iv, nvy - jv - 1, 0); // top boundary, reflective
                }
            });
    };

    void potential_boundary_conditions() {
        double phi_w            = -20.0 / (2 * 0.15); // cylinder potential normalized to ion quantities
        int ngc                 = grid.ngc;
        int nx                  = grid.ncells[0];
        int ny                  = grid.ncells[1];
        auto& eta               = this->eta;
        auto& poisson_bc_map    = this->poisson_bc_map;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (i < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0); // left
                else if (i >= nx - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0); // right
                else if (j < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0); // bottom
                else if (j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0); // top
                else if (eta(i, j) <= 0.0)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, phi_w); // immersed object
                else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
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
    double x_min              = reader.GetReal("grid", "x_min", 0.0);
    double y_min              = reader.GetReal("grid", "y_min", 0.0);
    double vx_min             = reader.GetReal("grid", "vx_min", 0.0);
    double vy_min             = reader.GetReal("grid", "vy_min", 0.0);
    double Lx                 = reader.GetReal("grid", "Lx", 1.0);
    double Ly                 = reader.GetReal("grid", "Ly", 1.0);
    double Lvx                = reader.GetReal("grid", "Lvx", 1.0);
    double Lvy                = reader.GetReal("grid", "Lvy", 1.0);
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
    Kokkos::printf("Phase space (x,y,vx,vy): [%f, %f, %f, %f]x[%f, %f, %f, %f]\n", x_min, y_min, vx_min, vy_min,
                   x_min + Lx, y_min + Ly, vx_min + Lvx, vy_min + Lvy);
    Kokkos::printf("Grid size, interior (nx,ny,nvx,nvy): [%d, %d, %d, %d]\n", nx_intr, ny_intr, nvx_intr, nvy_intr);
    Kokkos::printf("Simulation control: dt: %f, total_time: %f, total_steps: %d, diag_steps: %d\n", dt, total_time,
                   total_steps, diag_steps);

    Kokkos::Array<double, DIM> origin   = {x_min, y_min, vx_min, vy_min};         // origin of the grid
    Kokkos::Array<double, DIM> size     = {Lx, Ly, Lvx, Lvy};                     // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {nx_intr, ny_intr, nvx_intr, nvy_intr}; // number of interior cells

    Grid<1> grid(ncells_intr, ngc);
    grid.set_grid(origin, size, 0);
    ImmersedWorld world(grid);

    world.dt            = dt;          // time step size
    world.total_time    = total_time;  // total simulation time
    world.total_steps   = total_steps; // number of total_steps
    world.diag_steps    = diag_steps;  // number of steps between diagnostics
    world.species_names = {"i"};       // single kinetic ion species
    // m, q, T are normalized to electron quantities. Here the single kinetic ion
    // is normalized to itself, so its mass/charge/temperature all equal 1.
    world.m = Kokkos::Array<double, 1>{1.0}; // ion mass (= electron mass)
    world.q = Kokkos::Array<double, 1>{1.0}; // ion charge (= electron charge)
    world.T = Kokkos::Array<double, 1>{1.0}; // ion temperature (= electron temperature)

    PoissonSolver1stOrder poisson_solver(world);
    Writer writer(world, output_folder, output_prefix, {"ni", "phi", "Ex"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
