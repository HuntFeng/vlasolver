#include "full/vlasov.hpp"
#include "full/world.hpp"
#include "full/writer.hpp"
#include "grid.hpp"
#include "poisson_2nd_order.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    double last_step = -1;

    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {
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
                auto [x, y] = grid.center(i, j);
                eta(i, j)   = Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.1, 2);
            });
    }

    void construct_permittivity() {
        auto& eps_p             = this->eps_p;
        auto& eps_m             = this->eps_m;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                eps_p(i, j) = 1.0; // permittivity in the eta>0 region
                eps_m(i, j) = 5.0; // permittivity in the eta<0 region
            });
    }

    void initialize_distribution() {};

    void particle_boundary_conditions() {
        using Kokkos::exp;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                double n1 = normal(i, j, 0);
                double n2 = normal(i, j, 1);
                // electron
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    if (i < ngc) {
                        f(i, j, iv, jv, 1) =
                            (vx > 0.0) ? exp(-pow(vx - 5, 2)) * exp(-pow(vy, 2)) : 0.0; // left boundary, injection
                    } else if (i >= nx - ngc) {
                        f(i, j, iv, jv, 0) = 0.0; // right domain boundary, zero-inflow
                    } else if (j < ngc) {
                        f(i, j, iv, jv, 0) = f(i, 2 * ngc - j - 1, iv, nvy - jv - 1, 0); // bottom boundary, reflective
                    } else if (j >= ny - ngc) {
                        f(i, j, iv, jv, 0) =
                            f(i, 2 * (ny - ngc) - j - 1, iv, nvy - jv - 1, 0); // top boundary, reflective
                    } else if (eta(i, j) < 0.0 && vx * n1 + vy * n2 > 0.0) {
                        f(i, j, iv, jv, 0) = 0.0; // immersed wall absorbs, emits nothing back into the plasma
                    }
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    if (i < ngc) {
                        f(i, j, iv, jv, 1) =
                            (vx > 0.0) ? exp(-pow(vx - 5, 2)) * exp(-pow(vy, 2)) : 0.0; // left boundary, injection
                    } else if (i >= nx - ngc && vx < 0.0) {
                        f(i, j, iv, jv, 1) = 0.0; // right domain boundary, zero-inflow
                    } else if (j < ngc) {
                        f(i, j, iv, jv, 1) = f(i, 2 * ngc - j - 1, iv, nvy - jv - 1, 0); // bottom boundary, reflective
                    } else if (j >= ny - ngc) {
                        f(i, j, iv, jv, 1) =
                            f(i, 2 * (ny - ngc) - j - 1, iv, nvy - jv - 1, 0); // top boundary, reflective
                    } else if (eta(i, j) < 0.0 && vx * n1 + vy * n2 > 0.0) {
                        f(i, j, iv, jv, 1) = 0.0; // immersed wall absorbs, emits nothing back into the plasma
                    }
                };
            });
    };

    void potential_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
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
                else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
            });
    }

    void poisson_jump_conditions() {
        using Kokkos::min;
        if (current_step == last_step)
            return;
        last_step               = current_step;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double n1      = normal(i, j, 0);
                double n2      = normal(i, j, 1);
                double d_sigma = 0.0;
                for (int s = 0; s < 2; ++s) {
                    for (int iv = 0; iv < nvx; ++iv) {
                        for (int jv = 0; jv < nvy; ++jv) {
                            auto [_x, _y, vx, vy]     = grid.center(i, j, iv, jv, s);
                            auto [_dx, _dy, dvx, dvy] = grid.spacing(s);
                            // only accumulate when n.v < 0
                            d_sigma += -q[s] * min(n1 * vx + n2 * vy, 0.0) * f(i, j, iv, jv, s) * dvx * dvy * dt;
                        }
                    }
                }
                jump_b(i, j) += -d_sigma;
                jump_a(i, j) = 0.0; // no jump in the potential across the interface
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

    double Te     = 1.0;                   // electron temperature
    double Ti     = 0.1;                   // ion temperature normalized to Te
    double me     = 1.0;                   // electron mass
    double mi     = 2 * 1836.0;            // ion mass, normalized to me
    double v_th_e = Kokkos::sqrt(Te / me); // electron thermal velocity
    double v_th_i = Kokkos::sqrt(Ti / mi); // ion thermal velocity, normalized to v_th_e

    Grid grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
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

    PoissonSolver2ndOrder poisson_solver(world); // set lower relaxation for convergence
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
