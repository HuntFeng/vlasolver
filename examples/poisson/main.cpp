#include "reduced/grid.hpp"
#include "reduced/poisson_2nd_order.hpp"
#include "reduced/world.hpp"
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
        // return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
        return y - 2.5;
    }

    // KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const {
    //     double norm = Kokkos::sqrt(Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2));
    //     return {(x - 0.375) / norm, y / norm};
    // }
    //
    double permittivity(double x, double y) const { return 1.0; }

    void poisson_jump_conditions() {
        auto& grid = this->grid;
        int nx     = grid.ncells[0];
        int ny     = grid.ncells[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                // potential is continuous
                a(i, j) = 0.0;
                // electric field is not continuous but this works??
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
        double phi_w = -20.0 / (2 * 0.15); // cylinder potential normalized to electron quantities
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
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkosScopeGuard(argc, argv);

    Kokkos::Array<double, DIM> origin   = {0.0, 0.0, 0.0, 0.0}; // origin of the grid
    Kokkos::Array<double, DIM> size     = {1.0, 1.0, 1.0, 1.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {64, 64, 1, 1};       // number of interior cells
    const int ngc                       = 1;

    Grid grid(origin, size, ncells_intr, ngc);
    ImmersedWorld world(grid);
    PoissonSolver poisson_solver(world);

    using Kokkos::sin;
    using Kokkos::numbers::pi;
    auto& rho    = world.rho;
    const int nx = grid.ncells[0];
    const int ny = grid.ncells[1];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                rho(i, j) = 0.0;
            else
                rho(i, j) = -2.0 * pi * pi * sin(pi * x) * sin(pi * y);
        });

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);

    return 0;
}
