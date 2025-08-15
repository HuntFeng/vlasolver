#include "grid.hpp"
#include "poisson.hpp"
#include "vlasov.hpp"
#include "world.hpp"
#include "writer.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {}

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const {
        double norm = Kokkos::sqrt(Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2));
        return {(x - 0.375) / norm, y / norm};
    }

    void particle_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                if (i < ngc) {
                    f(i, j, iv, jv) =
                        (vx > 0.0) ? exp(-pow(vx - 5, 2)) * exp(-pow(vy, 2)) : 0.0; // left boundary, injection
                } else if (i >= nx - ngc) {
                    if (vx < 0.0)
                        f(i, j, iv, jv) = 0.0; // right boundary, zero-inflow
                } else if (j < ngc) {
                    f(i, j, iv, jv) = f(i, 2 * ngc - j - 1, iv, nvy - jv - 1); // bottom boundary, reflective
                } else if (j >= ny - ngc) {
                    f(i, j, iv, jv) = f(i, 2 * (ny - ngc) - j - 1, iv, nvy - jv - 1); // top boundary, reflective
                }
            });
    };

    void poisson_jump_conditions() {
        auto& grid = this->grid;
        int nx     = grid.ncells[0];
        int ny     = grid.ncells[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // potential and electric field are continuous
                a(i, j) = 0.0;
                b(i, j) = 0.0;

                // the immersed cylinder is a conductor, set a high permittivity
                auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
                eps(i, j)           = (surface(x, y) < 0.0) ? 1000.0 : 1.0;
            });
    }

    void potential_boundary_conditions(Kokkos::View<double**>& u) {
        using Kokkos::abs;
        auto& grid   = this->grid;
        int ngc      = grid.ngc;
        int nx       = u.extent(0);
        int ny       = u.extent(1);
        double dx    = grid.size[0] / (nx - 2 * ngc);
        double dy    = grid.size[1] / (ny - 2 * ngc);
        double phi_w = -66.67; // normalized potential at the wall of the charged cylinder
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double x   = (i - ngc + 0.5) * dx;
                double y   = (j - ngc + 0.5) * dy;
                double eta = surface(x, y);
                if (eta < 0.0) {
                    u(i, j) = phi_w; // inside the immersed object, set potential to a constant value
                }
            });

        for (int k = 0; k < ngc; ++k) {
            // left boundary, dirichlet
            Kokkos::deep_copy(Kokkos::subview(u, k, Kokkos::ALL), 0.0);
            // right boundary, neumann
            Kokkos::deep_copy(Kokkos::subview(u, nx - k - 1, Kokkos::ALL),
                              Kokkos::subview(u, nx - 2 * ngc + k, Kokkos::ALL));
            // bottom boundary, neumann
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, k), Kokkos::subview(u, Kokkos::ALL, 2 * ngc - k - 1));
            // top boundary, neumann
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, ny - k - 1),
                              Kokkos::subview(u, Kokkos::ALL, ny - 2 * ngc + k));
        }
    }

    // void potential_boundary_conditions() {
    //     using Kokkos::abs;
    //     auto& grid   = this->grid;
    //     int ngc      = grid.ngc;
    //     int nx       = grid.ncells[0];
    //     int ny       = grid.ncells[1];
    //     double phi_w = -66.67; // normalized potential at the wall of the charged cylinder
    //     Kokkos::parallel_for(
    //         Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //             auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
    //             if (surface(x, y) < 0.0) {
    //                 phi(i, j) = phi_w; // inside the immersed object, set potential to a constant value
    //             }
    //         });
    //
    //     for (int k = 0; k < ngc; ++k) {
    //         // left boundary, dirichlet
    //         Kokkos::deep_copy(Kokkos::subview(phi, k, Kokkos::ALL), 0.0);
    //         // right boundary, neumann
    //         Kokkos::deep_copy(Kokkos::subview(phi, nx - k - 1, Kokkos::ALL),
    //                           Kokkos::subview(phi, nx - 2 * ngc + k, Kokkos::ALL));
    //         // bottom boundary, neumann
    //         Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::ALL, k), Kokkos::subview(phi, Kokkos::ALL, 2 * ngc - k -
    //         1));
    //         // top boundary, neumann
    //         Kokkos::deep_copy(Kokkos::subview(phi, Kokkos::ALL, ny - k - 1),
    //                           Kokkos::subview(phi, Kokkos::ALL, ny - 2 * ngc + k));
    //     }
    // }
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
    Writer writer(world, output_folder, output_prefix, {"ni", "phi", "Ex"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
