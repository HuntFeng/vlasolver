#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Timer.hpp>
#include <gridformat/gridformat.hpp>

class PoissonSolver {
    using Grid = GridFormat::ImageGrid<2, double>;
    using Cell = Grid::Cell;

  private:
    Grid& grid;
    std::vector<Cell> cells;
    Kokkos::View<double**> phi_old;
    double tol;
    double omega;

  public:
    Kokkos::View<double**> phi;
    Kokkos::View<double**> rho;
    Kokkos::View<double**> eps;
    Kokkos::View<double**> a;
    Kokkos::View<double**> b;

    PoissonSolver(Grid& grid, double tol = 1e-6) : grid(grid), tol(tol) {
        auto cell_range = grid.cells();
        cells = std::vector<Cell>(cell_range.begin(), cell_range.end());
        auto [nx, ny] = grid.extents();
        phi = Kokkos::View<double**>("phi", nx, ny);
        rho = Kokkos::View<double**>("rho", nx, ny);
        eps = Kokkos::View<double**>("eps", nx, ny);
        a = Kokkos::View<double**>("a", nx, ny);
        b = Kokkos::View<double**>("b", nx, ny);
        phi_old = Kokkos::View<double**>("phi_old", nx, ny);
        omega = 2 / (1 + Kokkos::sin(Kokkos::numbers::pi / (nx + 1)));
    }

    double surface(double x, double y) const {
        // example 3
        // return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - 0.25 * 0.25;
        // example 5
        // return x * x + y * y - 0.5 * 0.5;
        // example 7
        return x * x + y * y - 0.5 * 0.5;
    }

    std::array<double, 2> normal(double x, double y) const {
        using Kokkos::pow;
        using Kokkos::sqrt;
        auto [dx, dy] = grid.spacing();
        double eta = surface(x, y);
        double eta_l = surface(x - dx, y);
        double eta_r = surface(x + dx, y);
        double eta_b = surface(x, y - dy);
        double eta_t = surface(x, y + dy);
        double eta_grad_x = (eta_r - eta_l) / (2 * dx);
        double eta_grad_y = (eta_t - eta_b) / (2 * dy);
        double eta_grad_norm = sqrt(pow(eta_grad_x, 2) + pow(eta_grad_y, 2));
        double nx = eta_grad_x / eta_grad_norm;
        double ny = eta_grad_y / eta_grad_norm;
        return {nx, ny};
    }

    void init() {
        using Kokkos::exp;
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                auto cell = cells[k];
                auto [i, j] = cell.location;
                auto [dx, dy] = grid.spacing();
                auto [nx, ny] = grid.extents();
                auto [x, y] = grid.center(cell);
                double eta = surface(x, y);
                double eta_l = surface(x - dx, y);
                double eta_r = surface(x + dx, y);
                double eta_b = surface(x, y - dy);
                double eta_t = surface(x, y + dy);
                // example 3
                // phi(i, j) = 0.0;
                // if (eta <= 0.0) {
                //     eps(i, j) = 2.0;
                //     rho(i, j) = 8 * (x * x + y * y - 1) * exp(-x * x - y * y);
                // } else {
                //     eps(i, j) = 1.0;
                //     rho(i, j) = 0.0;
                // }
                // if (eta * eta_l <= 0.0) {
                //     a(i, j) = -exp(-x * x - y * y);
                //     b(i, j) = 8 * (2 * x * x + 2 * y * y - x - y) * exp(-x * x - y * y);
                // }
                // if (eta * eta_r <= 0.0) {
                //     a(i, j) = -exp(-x * x - y * y);
                //     b(i, j) = 8 * (2 * x * x + 2 * y * y - x - y) * exp(-x * x - y * y);
                // }
                // if (eta * eta_b <= 0.0) {
                //     a(i, j) = -exp(-x * x - y * y);
                //     b(i, j) = 8 * (2 * x * x + 2 * y * y - x - y) * exp(-x * x - y * y);
                // }
                // if (eta * eta_t <= 0.0) {
                //     a(i, j) = -exp(-x * x - y * y);
                //     b(i, j) = 8 * (2 * x * x + 2 * y * y - x - y) * exp(-x * x - y * y);
                // }

                // example 5
                // eps(i, j) = 1.0;
                // rho(i, j) = 0.0;
                // if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                //     phi(i, j) = 1 + Kokkos::log(2 * Kokkos::sqrt(x * x + y * y));
                // } else {
                //     phi(i, j) = 0.0;
                // }
                // a(i, j) = 0.0;
                // if (eta * eta_l <= 0.0) {
                //     b(i, j) = 2;
                // }
                // if (eta * eta_r <= 0.0) {
                //     b(i, j) = 2;
                // }
                // if (eta * eta_b <= 0.0) {
                //     b(i, j) = 2;
                // }
                // if (eta * eta_t <= 0.0) {
                //     b(i, j) = 2;
                // }

                // example 7
                eps(i, j) = 1.0;
                rho(i, j) = 0.0;
                phi(i, j) = 0.0;
                if (eta * eta_l <= 0.0) {
                    a(i, j) = y * y - x * x;
                    b(i, j) = 4 * (y * y - x * x);
                }
                if (eta * eta_r <= 0.0) {
                    a(i, j) = y * y - x * x;
                    b(i, j) = 4 * (y * y - x * x);
                }
                if (eta * eta_b <= 0.0) {
                    a(i, j) = y * y - x * x;
                    b(i, j) = 4 * (y * y - x * x);
                }
                if (eta * eta_t <= 0.0) {
                    a(i, j) = y * y - x * x;
                    b(i, j) = 4 * (y * y - x * x);
                }
            });
    }

    void update_potential(Cell& cell) const {
        using Kokkos::abs;
        auto [i, j] = cell.location;
        auto [dx, dy] = grid.spacing();
        auto [x, y] = grid.center(cell);

        // eps at left, right, top, bottom fluxes
        double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
        double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
        double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
        double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

        // jump condition term at left, right, top, bottom fluxes
        double F_l = 0.0, F_r = 0.0, F_b = 0.0, F_t = 0.0;

        // interior indicator
        double eta = surface(x, y);
        double eta_l = surface(x - dx, y);
        double eta_r = surface(x + dx, y);
        double eta_b = surface(x, y - dy);
        double eta_t = surface(x, y + dy);

        // modify eps and F if discontinuity is detected
        if (eta * eta_l <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_l;
            double eta_m = (eta <= 0.0) ? eta : eta_l;
            double eps_p = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
            double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
            eps_l = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx, ny] = normal(x, y);
            auto [nx_l, ny_l] = normal(x - dx, y);
            double theta = abs(eta_l) / (abs(eta) + abs(eta_l));
            double a_gamma = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));
            double b_gamma = (b(i, j) * nx * abs(eta_l) + b(i - 1, j) * nx_l * abs(eta)) / (abs(eta) + abs(eta_l));
            if (eta <= 0.0)
                F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
            else
                F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
        }
        if (eta * eta_r <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_r;
            double eta_m = (eta <= 0.0) ? eta : eta_r;
            double eps_p = (eta > 0.0) ? eps(i, j) : eps(i + 1, j);
            double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i + 1, j);
            eps_r = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx, ny] = normal(x, y);
            auto [nx_r, ny_r] = normal(x + dx, y);
            double theta = abs(eta_r) / (abs(eta) + abs(eta_r));
            double a_gamma = (a(i, j) * abs(eta_r) + a(i + 1, j) * abs(eta)) / (abs(eta) + abs(eta_r));
            double b_gamma = (b(i, j) * nx * abs(eta_r) + b(i + 1, j) * nx_r * abs(eta)) / (abs(eta) + abs(eta_r));
            if (eta <= 0.0)
                F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
            else
                F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
        }
        if (eta * eta_b <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_b;
            double eta_m = (eta <= 0.0) ? eta : eta_b;
            double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j - 1);
            double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j - 1);
            eps_b = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx, ny] = normal(x, y);
            auto [nx_b, ny_b] = normal(x, y - dy);
            double theta = abs(eta_b) / (abs(eta) + abs(eta_b));
            double a_gamma = (a(i, j) * abs(eta_b) + a(i, j - 1) * abs(eta)) / (abs(eta) + abs(eta_b));
            double b_gamma = (b(i, j) * ny * abs(eta_b) + b(i, j - 1) * ny_b * abs(eta)) / (abs(eta) + abs(eta_b));
            if (eta <= 0.0)
                F_b = eps_b * a_gamma / (dy * dy) - eps_b * b_gamma * theta / (eps_p * dy);
            else
                F_b = -eps_b * a_gamma / (dy * dy) + eps_b * b_gamma * theta / (eps_m * dy);
        }
        if (eta * eta_t <= 0.0) {
            double eta_p = (eta > 0.0) ? eta : eta_t;
            double eta_m = (eta <= 0.0) ? eta : eta_t;
            double eps_p = (eta > 0.0) ? eps(i, j) : eps(i, j + 1);
            double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i, j + 1);
            eps_t = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

            auto [nx, ny] = normal(x, y);
            auto [nx_t, ny_t] = normal(x, y + dy);
            double theta = abs(eta_t) / (abs(eta) + abs(eta_t));
            double a_gamma = (a(i, j) * abs(eta_t) + a(i, j + 1) * abs(eta)) / (abs(eta) + abs(eta_t));
            double b_gamma = (b(i, j) * ny * abs(eta_t) + b(i, j + 1) * ny_t * abs(eta)) / (abs(eta) + abs(eta_t));
            if (eta <= 0.0)
                F_t = eps_t * a_gamma / (dy * dy) + eps_t * b_gamma * theta / (eps_p * dy);
            else
                F_t = -eps_t * a_gamma / (dy * dy) - eps_t * b_gamma * theta / (eps_m * dy);
        }

        // update potential field
        double denom = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy);
        double average = (eps_l * phi(i - 1, j) + eps_r * phi(i + 1, j)) / (dx * dx) +
                         (eps_b * phi(i, j - 1) + eps_t * phi(i, j + 1)) / (dy * dy);
        double Fx = F_l + F_r;
        double Fy = F_b + F_t;

        // sor update
        phi(i, j) = phi(i, j) + omega * (average - rho(i, j) - Fx - Fy - denom * phi(i, j)) / denom;
    }

    void red_black_update() {
        auto [nx, ny] = grid.extents();
        // update red grid points using black grid points in old values
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                Cell cell = cells[k];
                auto [i, j] = cell.location;
                if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1)
                    return; // skip boundary cells
                if ((i + j) % 2 != 0)
                    return; // skip red grid points
                update_potential(cell);
            });
        // update black grid points using already updated red grid points
        Kokkos::parallel_for(
            cells.size(), KOKKOS_CLASS_LAMBDA(int k) {
                Cell cell = cells[k];
                auto [i, j] = cell.location;
                if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1)
                    return; // skip boundary cells
                if ((i + j) % 2 != 1)
                    return; // skip red grid points
                update_potential(cell);
            });
    }

    double compute_error() {
        using Kokkos::abs;
        double err;
        Kokkos::Max<double> max_reducer(err);
        Kokkos::parallel_reduce(
            cells.size(),
            KOKKOS_CLASS_LAMBDA(int k, double& err) {
                Cell cell = cells[k];
                auto [i, j] = cell.location;
                max_reducer.join(err, abs(phi(i, j) - phi_old(i, j)));
            },
            max_reducer);
        return err;
    }

    void solve() {
        // GridFormat::VTKHDFImageGridTimeSeriesWriter writer{grid, "data_example7"};
        // writer.set_meta_data(
        //     "X", std::ranges::views::transform(grid.cells(), [&](const auto& cell) { return grid.center(cell)[0];
        //     }));
        // writer.set_meta_data(
        //     "Y", std::ranges::views::transform(grid.cells(), [&](const auto& cell) { return grid.center(cell)[1];
        //     }));
        // writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0], cell.location[1]); });
        // writer.write(0);
        int iter = 0;
        Kokkos::deep_copy(phi_old, phi);
        red_black_update();
        double err = compute_error();
        Kokkos::printf("Iteration %d: L_inf error = %f\n", iter, err);
        while (err > tol) {
            iter++;
            Kokkos::deep_copy(phi_old, phi);
            red_black_update();
            err = compute_error();
            if (iter % 1000 == 0) {
                Kokkos::printf("Iteration %d: L_inf error = %f\n", iter, err);
            }
        }
        Kokkos::printf("Iteration %d: L_inf error = %f\n", iter, err);

        // writer.write(1);

        GridFormat::VTKHDFImageGridWriter writer{grid};
        writer.set_meta_data("iter", iter);
        writer.set_meta_data(
            "X", std::ranges::views::transform(grid.cells(), [&](const auto& cell) { return grid.center(cell)[0]; }));
        writer.set_meta_data(
            "Y", std::ranges::views::transform(grid.cells(), [&](const auto& cell) { return grid.center(cell)[1]; }));
        writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0], cell.location[1]); });
        writer.write("data_example7");
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    // example 3
    // GridFormat::ImageGrid<2, double> grid{
    //     {0, 0},     // origin
    //     {1.0, 1.0}, // domain size
    //     {61, 61}    // number of cells (pixels) in each direction
    // };
    // example 5
    // GridFormat::ImageGrid<2, double> grid{
    //     {-1.0, -1.0}, // origin
    //     {2.0, 2.0},   // domain size
    //     {61, 61}      // number of cells (pixels) in each direction
    // };
    // example 7
    GridFormat::ImageGrid<2, double> grid{
        {-1.0, -1.0}, // origin
        {2.0, 2.0},   // domain size
        {61, 61}      // number of cells (pixels) in each direction
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
