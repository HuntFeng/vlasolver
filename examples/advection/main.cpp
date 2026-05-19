#include "grid.hpp"
#include "poisson.hpp"
#include "reduced/vlasov.hpp"
#include "reduced/world.hpp"
#include "reduced/writer.hpp"
#include <Kokkos_Core.hpp>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {}

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const {
        // return 1.0;
        return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
    }

    void initialize_distribution() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        double x0               = 0.2;
        double sigma_x          = 0.05;
        int ivx_peak            = ngc + 1;
        int ivy_peak            = ngc;
        auto [x, y, vx, vy]     = grid.center(0, 0, ivx_peak, ivy_peak);
        Kokkos::printf("Velocity of peak: (%f, %f)\n", vx, vy);

        auto [dx, dy, dvx, dvy] = grid.spacing(0);

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                auto [x, y, vx, vy] = grid.center(i, j, iv, jv);
                if (surface(x, y) <= 0.0)
                    return;
                double f0 = Kokkos::exp(-Kokkos::pow((x - x0) / sigma_x, 2)) / (dvx * dvy);
                if (iv == ivx_peak && jv == ivy_peak) {
                    f(i, j, iv, jv) = f0;
                } else {
                    f(i, j, iv, jv) = 0.0;
                }
            });
    };

    void particle_boundary_conditions() {
        // Periodic BCs in all directions for pure advection test
        auto& grid              = this->grid;
        auto& f                 = this->f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        // x-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {ngc, ny, nvx, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // left ghost = right interior
                f(i, j, iv, jv) = f(nx - 2 * ngc + i, j, iv, jv);
                // right ghost = left interior
                f(nx - ngc + i, j, iv, jv) = f(ngc + i, j, iv, jv);
            });
        // y-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ngc, nvx, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                f(i, j, iv, jv)            = f(i, ny - 2 * ngc + j, iv, jv);
                f(i, ny - ngc + j, iv, jv) = f(i, ngc + j, iv, jv);
            });
        // vx-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, ngc, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                f(i, j, iv, jv)             = f(i, j, nvx - 2 * ngc + iv, jv);
                f(i, j, nvx - ngc + iv, jv) = f(i, j, ngc + iv, jv);
            });
        // vy-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                f(i, j, iv, jv)             = f(i, j, iv, nvy - 2 * ngc + jv);
                f(i, j, iv, nvy - ngc + jv) = f(i, j, iv, ngc + jv);
            });
        Kokkos::fence();
    }
};

class EmptyPoissonSolver : PoissonSolver<EmptyPoissonSolver> {
  public:
    EmptyPoissonSolver() {}
    void solve() {}
    void compute_electric_field() {}
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    const int n = (argc == 2) ? std::stoi(argv[1]) : 64;
    Kokkos::printf("Input parameters:\n");
    double x_min  = 0.0;
    double y_min  = 0.0;
    double Lx     = 1.0;
    double Ly     = 0.5;
    double vx_min = -0.05;
    double vy_min = -0.5;
    double Lvx    = 1.0;
    double Lvy    = 1.0;
    int nx        = 2 * n;
    int ny        = n;
    int nvx       = 10;
    int nvy       = 1;
    int ngc       = 3;
    Kokkos::printf("Phase space (x,y,vx,vy): [%f, %f, %f, %f]x[%f, %f, %f, %f]\n", x_min, y_min, vx_min, vy_min,
                   x_min + Lx, y_min + Ly, vx_min + Lvx, vy_min + Lvy);
    Kokkos::printf("Grid size, interior (nx,ny,nvx,nvy): [%d, %d, %d, %d]\n", nx, ny, nvx, nvy);

    Kokkos::Array<double, DIM> origin   = {x_min, y_min, vx_min, vy_min}; // origin of the grid
    Kokkos::Array<double, DIM> size     = {Lx, Ly, Lvx, Lvy};             // size of the grid
    Kokkos::Array<int, DIM> ncells_intr = {nx, ny, nvx, nvy};             // number of interior cells

    Grid grid(ncells_intr, ngc);
    grid.set_grid(origin, size, 0);
    ImmersedWorld world(grid);
    double total_time = 1.0;
    double CFL        = 1.0; // I know vx is 0.1, so CFL=1.0 is enough
    double dt         = CFL * grid.spacing(0, 0)[0] / Lvx;
    int total_steps   = total_time / dt;
    int diag_steps    = total_steps;
    Kokkos::printf("Simulation control: dt: %f, total_time: %f, total_steps: %d, diag_steps: %d\n", dt, total_time,
                   total_steps, diag_steps);

    world.dt          = dt;          // time step size
    world.total_time  = total_time;  // total simulation time
    world.total_steps = total_steps; // number of total_steps
    world.diag_steps  = diag_steps;  // number of steps between diagnostics

    EmptyPoissonSolver poisson_solver;
    Writer writer(world, "data/advection", "output_" + std::to_string(n), {"ni"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
