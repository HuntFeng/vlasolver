#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <gridformat/gridformat.hpp>

using Grid = GridFormat::ImageGrid<2, double>;
using Cell = Grid::Cell;

double surface(double x, double y) {
    double x_c = 0.5, y_c = 0.5;
    // return (x - x_c) * (x - x_c) + (y - y_c) * (y - y_c);
    return x - 1;
}

std::array<double, 2> normal(int i, int j, double dx, double dy) {
    double eta = surface(i * dx, j * dy);
    double eta_l = surface((i - 1) * dx, j * dy);
    double eta_r = surface((i + 1) * dx, j * dy);
    double eta_b = surface(i * dx, (j - 1) * dy);
    double eta_t = surface(i * dx, (j + 1) * dy);
    double eta_grad_x = (eta_r - eta_l) / dx;
    double eta_grad_y = (eta_t - eta_b) / dy;
    double eta_grad_norm = sqrt(pow(eta_grad_x, 2) + pow(eta_grad_y, 2));
    double nx = eta_grad_x / eta_grad_norm;
    double ny = eta_grad_y / eta_grad_norm;
    return {nx, ny};
}

void update_potential_at_cell(Kokkos::View<double**>& phi, Kokkos::View<double**>& rho, Kokkos::View<double**>& eps,
                              Grid& grid, Cell cell) {
    auto [i, j] = cell.location;
    auto [x, y] = grid.center(cell);
    auto [dx, dy] = grid.spacing();
}

int main(int argc, char* argv[]) {
    using Kokkos::abs;
    using Kokkos::pow;
    using Kokkos::sqrt;

    Kokkos::ScopeGuard guard(argc, argv);

    const int nx = 128, ny = 128;
    Grid grid{
        {1.0, 1.0}, // domain size
        {nx, ny}    // number of cells (pixels) in each direction
    };
    auto [dx, dy] = grid.spacing();
    GridFormat::VTKHDFTimeSeriesWriter writer{grid, "data"};

    // potential field
    Kokkos::View<double**> phi("phi", nx, ny);
    Kokkos::View<double**> rho("rho", nx, ny);
    Kokkos::View<double**> eps("eps", nx, ny);
    Kokkos::View<double**> a("a", nx, ny); // jump of phi
    Kokkos::View<double**> b("b", nx, ny); // jump of eps*phi

    // Initialize phi
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(int i, int j) {
            double x = i * dx; // y-coordinate
            if (j == 0) {
                phi(i, j) = x * (1.0 - x);
            } else {
                phi(i, j) = 0.0;
            }
        });
    Kokkos::deep_copy(rho, 0.0);
    Kokkos::deep_copy(eps, 1.0);

    writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0], cell.location[1]); });
    writer.write(0);

    // Jacobi iteration
    for (int iter = 1; iter < 100; ++iter) {
        // update red grid points using black grid points in old values
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({1, 1}, {nx - 1, ny - 1}), KOKKOS_LAMBDA(int i, int j) {
                if ((i + j) % 2 != 0)
                    return;
                // eps at left, right, top, bottom fluxes
                double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
                double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
                double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
                double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

                // jump condition term at left, right, top, bottom fluxes
                double F_l = 0.0, F_r = 0.0, F_b = 0.0, F_t = 0.0;

                // interior indicator
                double eta = surface(i * dx, j * dy);
                double eta_l = surface((i - 1) * dx, j * dy);
                double eta_r = surface((i + 1) * dx, j * dy);
                double eta_b = surface(i * dx, (j - 1) * dy);
                double eta_t = surface(i * dx, (j + 1) * dy);

                // modify eps and F if discontinuity is detected
                if (eta * eta_l < 0.0) {
                    double eta_p = (eta > 0.0) ? eta : eta_l;
                    double eta_m = (eta <= 0.0) ? eta : eta_l;
                    double eps_p = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
                    double eps_m = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
                    eps_l = eps_p * eps_m / (eps_p * eta_m + eps_m * eta_p);

                    double a_gamma = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));

                    double theta = abs(eta_l) / (abs(eta) + abs(eta_l));
                    auto [nx, ny] = normal(i, j, dx, dy);
                    auto [nx_l, ny_l] = normal(i - 1, j, dx, dy);
                    double b_gamma =
                        (b(i, j) * nx * abs(eta_l) + b(i - 1, j) * nx_l * abs(eta)) / (abs(eta) + abs(eta_l));
                    if (eta <= 0.0)
                        F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
                    else
                        F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
                }
                if (eta * eta_r < 0.0) {
                    double theta = 0.0;
                }
                if (eta * eta_b < 0.0) {
                    double theta = 0.0;
                }
                if (eta * eta_t < 0.0) {
                    double theta = 0.0;
                }

                // update potential field
                double denom = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy);
                double average = (eps_l * phi(i - 1, j) + eps_r * phi(i + 1, j)) / (dx * dx) +
                                 (eps_b * phi(i, j - 1) + eps_t * phi(i, j + 1)) / (dy * dy);
                double Fx = F_l + F_r;
                double Fy = F_b + F_t;

                phi(i, j) = (average + rho(i, j) + Fx + Fy) / denom;
                // phi(i, j) = 0.25 * (phi(i - 1, j) + phi(i + 1, j) + phi(i, j - 1) + phi(i, j + 1) + rho(i, j));
            });
        // update black grid point using already updated red grid points
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({1, 1}, {nx - 1, ny - 1}), KOKKOS_LAMBDA(int i, int j) {
                if ((i + j) % 2 != 1)
                    return;
                // eps at left, right, top, bottom fluxes
                double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
                double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
                double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
                double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

                // jump condition term at left, right, top, bottom fluxes
                double F_l = 0.0, F_r = 0.0, F_b = 0.0, F_t = 0.0;

                // interior indicator
                double eta = surface(i * dx, j * dy);
                double eta_l = surface((i - 1) * dx, j * dy);
                double eta_r = surface((i + 1) * dx, j * dy);
                double eta_b = surface(i * dx, (j - 1) * dy);
                double eta_t = surface(i * dx, (j + 1) * dy);

                // modify eps and F if discontinuity is detected
                if (eta * eta_l < 0.0) {
                    eps_l = 0.0;
                }
                if (eta * eta_r < 0.0) {
                    eps_r = 0.0;
                }
                if (eta * eta_b < 0.0) {
                    eps_b = 0.0;
                }
                if (eta * eta_t < 0.0) {
                    eps_t = 0.0;
                }

                // update potential field
                double denom = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy);
                double average = (eps_l * phi(i - 1, j) + eps_r * phi(i + 1, j)) / (dx * dx) +
                                 (eps_b * phi(i, j - 1) + eps_t * phi(i, j + 1)) / (dy * dy);
                double Fx = F_l + F_r;
                double Fy = F_b + F_t;

                phi(i, j) = (average + rho(i, j) + Fx + Fy) / denom;
                // phi(i, j) = 0.25 * (phi(i - 1, j) + phi(i + 1, j) + phi(i, j - 1) + phi(i, j + 1) + rho(i, j));
            });
        writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0], cell.location[1]); });
        writer.write(iter);
    }

    return 0;
}
