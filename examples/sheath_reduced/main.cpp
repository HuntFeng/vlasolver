#include "reduced/grid.hpp"
#include "reduced/poisson.hpp"
#include "reduced/vlasov.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {}

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return x + 1.0; }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const { return {1.0, 0.0}; }

    void initialize_distribution() {
        // using Kokkos::exp;
        // using Kokkos::pow;
        //
        // // must assign grid and f here, otherwise, using world.grid.xxx in device region causes illegal memory access
        // auto& grid              = this->grid;
        // auto& f                 = this->f;
        // auto [nx, ny, nvx, nvy] = grid.ncells;
        //
        // Kokkos::parallel_for(
        //     Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
        //     KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
        //         auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
        //         if (y > 0.3)
        //             f(i, j, iv, jv) = (vy < 0.0) ? exp(-pow(vx, 2)) * exp(-pow(vy + 2, 2)) / 3.17 : 0.0;
        //     });
    };

    void particle_boundary_conditions() {
        using Kokkos::exp;
        using Kokkos::pow;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                if (j < ngc) {
                    if (vy > 0.0)
                        f(i, j, iv, jv) = 0.0; // bottom boundary, zero-inflow
                } else if (j >= ny - ngc) {
                    f(i, j, iv, jv) =
                        (vy < 0.0) ? exp(-pow(vx, 2)) * exp(-pow(vy + 2, 2)) / 3.0 : 0.0; // top boundary, injection
                }
            });

        // periodic boundary conditions for left and right boundaries
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(0, ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                          Kokkos::subview(f, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    };

    void poisson_jump_conditions() {
        // skip, since no jump conditions for immersed boundary
    }

    void potential_boundary_conditions(Kokkos::View<double**>& u) {
        using Kokkos::abs;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        auto [dx, dy, dvx, dvy] = grid.spacing;
        int ngc                 = grid.ngc;
        // double phi_w            = -Kokkos::log(Kokkos::sqrt(1836 / Kokkos::numbers::pi)); // wall potential
        double phi_w = -15.0; // wall potential

        // top boundary, dirichlet
        Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, Kokkos::make_pair(ny - ngc, ny)), 0.0);
        // bottom boundary, dirichlet
        // Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, Kokkos::make_pair(0, ngc)), phi_w);

        // bottom boundary, floating potential
        Kokkos::parallel_for(
            Kokkos::RangePolicy(0, nx), KOKKOS_CLASS_LAMBDA(const int i) {
                phi(i, ngc - 1) = 2 * phi_w - phi(i, ngc + 1);
                // double flux_e = Kokkos::exp(phi_w);
                // double flux_i = 0.0;
                // for (int iv = ngc; iv < nvx - ngc; ++iv) {
                //     for (int jv = ngc; jv < nvy - ngc; ++jv) {
                //         auto [x, y, vx, vy] = grid.center({i, ngc, iv, jv});
                //         if (vy >= 0.0)
                //             continue;
                //         flux_i += Kokkos::abs(vy) * f(i, ngc, iv, jv) * dvx * dvy;
                //     }
                // }
                // // Ey = (flux_i - flux_e) = -dphi/dy
                // phi(i, ngc - 1) = phi(i, ngc + 1) + (flux_i - flux_e) * dt * 2 * dy;
            });
        // left and right boundary, periodic
        Kokkos::deep_copy(Kokkos::subview(u, Kokkos::make_pair(0, ngc), Kokkos::ALL),
                          Kokkos::subview(u, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL));
        Kokkos::deep_copy(Kokkos::subview(u, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL),
                          Kokkos::subview(u, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL));
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

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);

    world.dt          = dt;          // time step size
    world.total_time  = total_time;  // total simulation time
    world.total_steps = total_steps; // number of total_steps
    world.diag_steps  = diag_steps;  // number of steps between diagnostics

    PoissonSolver poisson_solver(world, 1e-6, 1e6);
    Writer writer(world, output_folder, output_prefix, {"ni", "phi", "fi"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
