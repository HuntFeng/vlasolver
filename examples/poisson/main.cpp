#include "reduced/grid.hpp"
#include "reduced/poisson_2nd_order.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <Kokkos_Core.hpp>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {

        int ngc = grid.ngc;
        int nx  = grid.ncells[0];
        int ny  = grid.ncells[1];
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc) {
                    poisson_bc_map(i, j) = BCPair(BCType::Dirichlet, 0.0);
                } else {
                    poisson_bc_map(i, j) = BCPair(BCType::None, 0.0);
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        return Kokkos::pow(x - 0.5, 2) + Kokkos::pow(y - 0.5, 2) - Kokkos::pow(0.25, 2);
    }

    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return surface(x, y) <= 0.0 ? 2.0 : 1.0; }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const { return -Kokkos::exp(-(x * x + y * y)); }

    KOKKOS_INLINE_FUNCTION double poisson_jump_condition_b(double x, double y) const {
        return 8.0 * (2 * x * x + 2 * y * y - x - y) * Kokkos::exp(-(x * x + y * y));
    }
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);

    const int n                         = (argc == 2) ? std::stoi(argv[1]) : 64;

    Kokkos::Array<double, DIM> origin   = {0.0, 0.0, 0.0, 0.0}; // origin of the grid
    Kokkos::Array<double, DIM> size     = {1.0, 1.0, 1.0, 1.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {n, n, 1, 1};         // number of interior cells
    const int ngc                       = 3;

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);
    PoissonSolver poisson_solver(world);
    Writer writer(world, "data/poisson", "poisson_" + std::to_string(n), {"phi", "Ex", "Ey"});

    using Kokkos::sin;
    using Kokkos::numbers::pi;
    auto& rho    = world.rho;
    const int nx = grid.ncells[0];
    const int ny = grid.ncells[1];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            if (world.surface(x, y) <= 0.0)
                rho(i, j) = -8 * (x * x + y * y - 1.0) * Kokkos::exp(-(x * x + y * y));
            else
                rho(i, j) = 0.0;
        });

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    poisson_solver.compute_electric_field();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);

    writer.write(0.0);

    return 0;
}
