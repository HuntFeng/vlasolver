#include <Kokkos_Core.hpp>
#include <gridformat/gridformat.hpp>
#include <impl/Kokkos_Profiling.hpp>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    const int nx = 128, ny = 128;
    GridFormat::ImageGrid<2, double> grid{
        {1.0, 1.0}, // domain size
        {nx, ny}    // number of cells (pixels) in each direction
    };
    auto [dx, dy] = grid.spacing();
    GridFormat::VTKHDFTimeSeriesWriter writer{grid, "data"};

    // potential field
    Kokkos::View<double**> phi("phi", nx, ny);
    Kokkos::View<double**> rho("rho", nx, ny);
    Kokkos::View<double**> eps("eps", nx, ny);

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
                if ((i + j) % 2 == 0)
                    phi(i, j) = 0.25 * (phi(i - 1, j) + phi(i + 1, j) + phi(i, j - 1) + phi(i, j + 1) + rho(i, j));
            });
        // update black grid point using already updated red grid points
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({1, 1}, {nx - 1, ny - 1}), KOKKOS_LAMBDA(int i, int j) {
                if ((i + j) % 2 == 1)
                    phi(i, j) = 0.25 * (phi(i - 1, j) + phi(i + 1, j) + phi(i, j - 1) + phi(i, j + 1) + rho(i, j));
            });
        writer.set_cell_field("phi", [&](const auto cell) { return phi(cell.location[0], cell.location[1]); });
        writer.write(iter);
    }

    return 0;
}
