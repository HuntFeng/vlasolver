#include <KokkosBlas1_nrm2.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Timer.hpp>
#include <gridformat/gridformat.hpp>

class PoissonSolver {
    using Grid = GridFormat::ImageGrid<1, double>;
    using Cell = Grid::Cell;

  private:
    Grid& grid;
    std::vector<Cell> cells;
    double tol;
    Kokkos::View<double*> phi_old;
    Kokkos::View<double*> residual;

  public:
    Kokkos::View<double*> phi;
    Kokkos::View<double*> rho;
    Kokkos::View<double*> eps;
    Kokkos::View<double*> a;
    Kokkos::View<double*> b;

    PoissonSolver(Grid& grid, double tol = 1e-6) : grid(grid), tol(tol) {
        auto cell_range = grid.cells();
        cells = std::vector<Cell>(cell_range.begin(), cell_range.end());
        auto [nx] = grid.extents();
        phi = Kokkos::View<double*>("phi", nx);
        rho = Kokkos::View<double*>("rho", nx);
        eps = Kokkos::View<double*>("eps", nx);
        a = Kokkos::View<double*>("a", nx);
        b = Kokkos::View<double*>("b", nx);
        phi_old = Kokkos::View<double*>("phi", nx);
        residual = Kokkos::View<double*>("residual", nx);
    }

    double surface(double x) const {
        using Kokkos::abs;
        // example 1
        return x - 0.5;
        // example 2
        // return abs(x - 0.45) - 0.15;
    }

    std::array<double, 1> normal(double x) const {
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;
        auto [dx] = grid.spacing();
        double eta = surface(x);
        double eta_l = surface(x - dx);
        double eta_r = surface(x + dx);
        double eta_grad_x = (eta_r - eta_l) / (2 * dx);
        double eta_grad_norm = abs(eta_grad_x);
        double nx = eta_grad_x / eta_grad_norm;
        return {nx};
    }

    void init() {
        using Kokkos::exp;
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                auto cell = cells[k];
                auto [i] = cell.location;
                auto [dx] = grid.spacing();
                auto [x] = grid.center(cell);
                auto [nx] = grid.extents();
                double eta = surface(x);
                double eta_l = surface(x - dx);
                double eta_r = surface(x + dx);
                // example 1
                phi(i) = (i == nx - 1) ? 2 : 0.0;
                eps(i) = 1.0;
                rho(i) = 0.0;
                b(i) = 0.0;
                if (eta * eta_l <= 0.0) {
                    a(i) = 1.0;
                }
                if (eta * eta_r <= 0.0) {
                    a(i) = 1.0;
                }
                // example 2
                // phi(i) = 0.0;
                // b(i) = 0.0;
                // if (eta <= 0.0) {
                //     eps(i) = 2.0;
                //     rho(i) = (8 * x * x - 4) * exp(-x * x);
                // } else {
                //     eps(i) = 1.0;
                //     rho(i) = 0.0;
                // }
                // if (eta * eta_l <= 0.0) {
                //     if (x < 0.4) {
                //         a(i) = -exp(-0.09);
                //         b(i) = -1.2 * exp(-0.09);
                //     } else {
                //         a(i) = -exp(-0.36);
                //         b(i) = 2.4 * exp(-0.36);
                //     }
                // }
                // if (eta * eta_r <= 0.0) {
                //     if (x < 0.4) {
                //         a(i) = -exp(-0.09);
                //         b(i) = -1.2 * exp(-0.09);
                //     } else {
                //         a(i) = -exp(-0.36);
                //         b(i) = 2.4 * exp(-0.36);
                //     }
                // }
            });
    }

    void update_potential(Cell& cell) const {
        using Kokkos::abs;
        auto [i] = cell.location;
        auto [dx] = grid.spacing();
        auto [x] = grid.center(cell);

        // eps at left, right, top, bottom fluxes
        double eps_l = 0.5 * (eps(i - 1) + eps(i));
        double eps_r = 0.5 * (eps(i + 1) + eps(i));

        // jump condition term at left, right, top, bottom fluxes
        double F_l = 0.0, F_r = 0.0;

        // interior indicator
        double eta = surface(x);
        double eta_l = surface(x - dx);
        double eta_r = surface(x + dx);

        // modify eps and F if discontinuity is detected
        if (eta * eta_l <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_l;
            double eta_m = (eta <= 0.0) ? eta : eta_l;
            double eps_p = (eta > 0.0) ? eps(i) : eps(i - 1);
            double eps_m = (eta <= 0.0) ? eps(i) : eps(i - 1);
            eps_l = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx] = normal(x);
            auto [nx_l] = normal(x - dx);
            double theta = abs(eta_l) / (abs(eta) + abs(eta_l));
            double a_gamma = (a(i) * abs(eta_l) + a(i - 1) * abs(eta)) / (abs(eta) + abs(eta_l));
            double b_gamma = (b(i) * nx * abs(eta_l) + b(i - 1) * nx_l * abs(eta)) / (abs(eta) + abs(eta_l));
            if (eta <= 0.0)
                F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
            else
                F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
        }
        if (eta * eta_r <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_r;
            double eta_m = (eta <= 0.0) ? eta : eta_r;
            double eps_p = (eta > 0.0) ? eps(i) : eps(i + 1);
            double eps_m = (eta <= 0.0) ? eps(i) : eps(i + 1);
            eps_r = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx] = normal(x);
            auto [nx_r] = normal(x + dx);
            double theta = abs(eta_r) / (abs(eta) + abs(eta_r));
            double a_gamma = (a(i) * abs(eta_r) + a(i + 1) * abs(eta)) / (abs(eta) + abs(eta_r));
            double b_gamma = (b(i) * nx * abs(eta_r) + b(i + 1) * nx_r * abs(eta)) / (abs(eta) + abs(eta_r));
            if (eta <= 0.0)
                F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
            else
                F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
        }

        // update potential field
        double denom = (eps_l + eps_r) / (dx * dx);
        double average = (eps_l * phi(i - 1) + eps_r * phi(i + 1)) / (dx * dx);
        double Fx = F_l + F_r;

        phi(i) = (average - rho(i) - Fx) / denom;
    }

    void sor_update() {
        auto [nx] = grid.extents();
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                Cell cell = cells[k];
                auto [i] = cell.location;
                if (i < 1 || i >= nx - 1)
                    return; // skip boundary cells
                if (i % 2 != 0)
                    return; // skip red grid points
                update_potential(cell);
            });
        // update black grid points using already updated red grid points
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                Cell cell = cells[k];
                auto [i] = cell.location;
                if (i < 1 || i >= nx - 1)
                    return; // skip boundary cells
                if (i % 2 != 1)
                    return; // skip red grid points
                update_potential(cell);
            });
    }

    double compute_err() {
        using Kokkos::abs;
        auto [nx] = grid.extents();
        double err = 0.0;
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(int i) {
                if (i < 1 || i >= nx - 1)
                    return; // skip boundary cells
                residual(i) = abs(phi(i) - phi_old(i));
            });
        return KokkosBlas::nrm2(residual);
    }

    void solve() {
        GridFormat::VTKHDFImageGridTimeSeriesWriter writer{grid, "data_example1"};
        writer.set_meta_data(
            "X", std::ranges::views::transform(grid.cells(), [&](const auto& cell) { return grid.center(cell)[0]; }));
        writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0]); });
        writer.write(0);

        int iter = 0;
        Kokkos::deep_copy(phi_old, phi);
        sor_update();
        double err = compute_err();
        Kokkos::printf("Iteration %d: L2 error = %f\n", iter, err);
        while (err > tol) {
            iter++;
            Kokkos::deep_copy(phi_old, phi);
            sor_update();
            err = compute_err();
            if (iter % 1000 == 0) {
                Kokkos::printf("Iteration %d: L2 error = %f\n", iter, err);
            }
        }
        Kokkos::printf("Iteration %d: L2 error = %f\n", iter, err);

        writer.write(1);
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    // example 1
    GridFormat::ImageGrid<1, double> grid{
        {1},  // domain size
        {100} // number of cells (pixels) in each direction
    };
    PoissonSolver poisson_solver(grid);
    poisson_solver.init();
    Kokkos::Timer timer;
    double start_time = timer.seconds();
    poisson_solver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
    return 0;
}
