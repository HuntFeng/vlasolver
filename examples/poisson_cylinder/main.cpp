#include "reduced/grid.hpp"
#include "reduced/poisson_2nd_order.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <Kokkos_Core.hpp>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {

        double phi_w = -20.0 / 0.3;
        int ngc      = grid.ngc;
        int nx       = grid.ncells[0];
        int ny       = grid.ncells[1];
        double dx    = grid.spacing[0];
        double dy    = grid.spacing[1];
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
                double eta          = surface(x, y);
                double eta_l        = surface(x - dx, y);
                double eta_r        = surface(x + dx, y);
                double eta_b        = surface(x, y - dy);
                double eta_t        = surface(x, y + dy);

                if (i < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0);
                else if (i >= nx - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0);
                else if (j < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0);
                else if (j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0);
                else if (eta <= 0) {
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, phi_w);
                } else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None, 0.0);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
    }

    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return surface(x, y) <= 0.0 ? 1000.0 : 1.0; }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const { return 0.0; }

    KOKKOS_INLINE_FUNCTION double poisson_jump_condition_b(double x, double y) const { return 0.0; }
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);

    const int n                         = (argc == 2) ? std::stoi(argv[1]) : 64;

    Kokkos::Array<double, DIM> origin   = {0.0, 0.0, 0.0, 0.0}; // origin of the grid
    Kokkos::Array<double, DIM> size     = {1.0, 0.5, 1.0, 1.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {2 * n, n, 1, 1};     // number of interior cells
    const int ngc                       = 3;

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);
    double tol      = 1e-12;
    int gmres_m     = 100;
    int max_restart = 30;
    PoissonSolver poisson_solver(world, tol, gmres_m, max_restart);
    Writer writer(world, "data/poisson_cylinder", "output_" + std::to_string(n), {"phi", "Ex", "Ey"});

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    poisson_solver.compute_electric_field();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);

    writer.write(0.0);

    return 0;
}
