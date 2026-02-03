#include "reduced/grid.hpp"
#include "reduced/poisson_2nd_order.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Preconditioner.hpp>
#include <KokkosSparse_gmres.hpp>
#include <Kokkos_Core.hpp>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {}

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        return Kokkos::pow(x - 0.5, 2) + Kokkos::pow(y - 0.5, 2) - Kokkos::pow(0.25, 2);
    }

    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return surface(x, y) <= 0.0 ? 2.0 : 1.0; }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const { return -Kokkos::exp(-(x * x + y * y)); }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_b(double x, double y) const {
        return 8.0 * (2 * x * x + 2 * y * y - x - y) * Kokkos::exp(-(x * x + y * y));
    }

    void potential_boundary_conditions(Kokkos::View<double**>& u) {
        // TODO: not used in solver for now
        using Kokkos::abs;
        auto& grid = this->grid;
        int ngc    = grid.ngc;
        int nx     = u.extent(0);
        int ny     = u.extent(1);

        for (int k = 0; k < ngc; ++k) {
            // dirichlet
            Kokkos::deep_copy(Kokkos::subview(u, k, Kokkos::ALL), 0.0);
            Kokkos::deep_copy(Kokkos::subview(u, nx - k - 1, Kokkos::ALL), 0.0);
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, k), 0.0);
            Kokkos::deep_copy(Kokkos::subview(u, Kokkos::ALL, ny - k - 1), 0.0);
        }
    }
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);

    Kokkos::Array<double, DIM> origin   = {0.0, 0.0, 0.0, 0.0}; // origin of the grid
    Kokkos::Array<double, DIM> size     = {1.0, 1.0, 1.0, 1.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {64, 64, 1, 1};       // number of interior cells
    const int ngc                       = 3;

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);
    PoissonSolver poisson_solver(world);
    Writer writer(world, "data/poisson", "poisson", {"phi"});

    using Kokkos::sin;
    using Kokkos::numbers::pi;
    auto& rho    = world.rho;
    const int nx = grid.ncells[0];
    const int ny = grid.ncells[1];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            if (world.surface(x, y) <= 0.0)
                rho(i, j) = 8 * (x * x + y * y - 1.0) * Kokkos::exp(-(x * x + y * y));
            else
                rho(i, j) = 0.0;
        });

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);

    writer.write(0.0);

    return 0;
}
