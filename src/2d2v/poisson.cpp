#include "poisson.hpp"
#include "world.hpp"
#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

PoissonSolver::PoissonSolver(World& world, double tol)
    : world(world),
      tol(tol) {
    int nx  = world.grid.ncells[0];
    int ny  = world.grid.ncells[1];
    phi_old = Kokkos::View<double**>("phi_old", nx, ny);
    omega   = 2 / (1 + Kokkos::sin(Kokkos::numbers::pi / (ny + 1)));
}

void PoissonSolver::apply_potential_boundary_conditions() {
    using Kokkos::abs;
    auto& grid              = world.grid;
    auto& phi               = world.phi;
    int nx                  = grid.ncells[0];
    int ny                  = grid.ncells[1];
    auto [dx, dy, dvx, dvy] = grid.spacing;
    int ngc                 = grid.ngc;
    double phi_w            = -20.0; // normalized potential at the wall of the charged cylinder
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});
            double eta          = world.surface(x, y);

            if (eta <= 0.0) {
                phi(i, j) = phi_w; // inside the immersed object, set potential to a constant value
            }
            if (i == ngc - 1) {
                phi(i, j) = 0.0; // left boundary, dirichlet
            }
            if (i == nx - ngc) {
                phi(i, j) = phi(nx - ngc - 2, j); // right boundary, neumann
            }
            if (j == ngc - 1) {
                phi(i, j) = phi(i, ngc + 1); // bottom boundary, neumann
            }
            if (j == ny - ngc) {
                phi(i, j) = phi(i, ny - ngc - 2); // top boundary, neumann
            }
        });
}

void PoissonSolver::red_black_update(int is_update_red) {
    using Kokkos::abs;
    auto& grid = world.grid;
    auto& phi  = world.phi;
    auto& rho  = world.rho;
    auto& eps  = world.eps;
    auto& a    = world.a;
    auto& b    = world.b;

    int nx     = grid.ncells[0];
    int ny     = grid.ncells[1];
    double dx  = grid.spacing[0];
    double dy  = grid.spacing[1];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            if (i < grid.ngc || i >= nx - grid.ngc || j < grid.ngc || j >= ny - grid.ngc)
                return; // skip boundary cells
            if ((i + j) % 2 == is_update_red)
                return; // skip red grid points

            auto [x, y, vx, vy] = grid.center({i, j, 0, 0});

            // jump condition term at left, right, top, bottom fluxes
            double F_l = 0.0, F_r = 0.0, F_b = 0.0, F_t = 0.0;

            // eps at left, right, top, bottom fluxes
            double eps_l = 0.5 * (eps(i - 1, j) + eps(i, j));
            double eps_r = 0.5 * (eps(i + 1, j) + eps(i, j));
            double eps_b = 0.5 * (eps(i, j - 1) + eps(i, j));
            double eps_t = 0.5 * (eps(i, j + 1) + eps(i, j));

            // interior indicator
            double eta   = world.surface(x, y);
            double eta_l = world.surface(x - dx, y);
            double eta_r = world.surface(x + dx, y);
            double eta_b = world.surface(x, y - dy);
            double eta_t = world.surface(x, y + dy);

            // modify eps and F if discontinuity is detected
            if (eta * eta_l <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_l;
                double eta_m  = (eta <= 0.0) ? eta : eta_l;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i - 1, j);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i - 1, j);
                eps_l         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_l, n2_l] = world.normal(x - dx, y, dx, dy);
                double theta      = abs(eta_l) / (abs(eta) + abs(eta_l));
                double a_gamma    = (a(i, j) * abs(eta_l) + a(i - 1, j) * abs(eta)) / (abs(eta) + abs(eta_l));
                double b_gamma = (b(i, j) * n1 * abs(eta_l) + b(i - 1, j) * n1_l * abs(eta)) / (abs(eta) + abs(eta_l));
                if (eta <= 0.0)
                    F_l = eps_l * a_gamma / (dx * dx) - eps_l * b_gamma * theta / (eps_p * dx);
                else
                    F_l = -eps_l * a_gamma / (dx * dx) + eps_l * b_gamma * theta / (eps_m * dx);
            }
            if (eta * eta_r <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_r;
                double eta_m  = (eta <= 0.0) ? eta : eta_r;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i + 1, j);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i + 1, j);
                eps_r         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_r, n2_r] = world.normal(x + dx, y, dx, dy);
                double theta      = abs(eta_r) / (abs(eta) + abs(eta_r));
                double a_gamma    = (a(i, j) * abs(eta_r) + a(i + 1, j) * abs(eta)) / (abs(eta) + abs(eta_r));
                double b_gamma = (b(i, j) * n1 * abs(eta_r) + b(i + 1, j) * n1_r * abs(eta)) / (abs(eta) + abs(eta_r));
                if (eta <= 0.0)
                    F_r = eps_r * a_gamma / (dx * dx) + eps_r * b_gamma * theta / (eps_p * dx);
                else
                    F_r = -eps_r * a_gamma / (dx * dx) - eps_r * b_gamma * theta / (eps_m * dx);
            }
            if (eta * eta_b <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_b;
                double eta_m  = (eta <= 0.0) ? eta : eta_b;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i, j - 1);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i, j - 1);
                eps_b         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_b, n2_b] = world.normal(x, y - dy, dx, dy);
                double theta      = abs(eta_b) / (abs(eta) + abs(eta_b));
                double a_gamma    = (a(i, j) * abs(eta_b) + a(i, j - 1) * abs(eta)) / (abs(eta) + abs(eta_b));
                double b_gamma = (b(i, j) * n2 * abs(eta_b) + b(i, j - 1) * n2_b * abs(eta)) / (abs(eta) + abs(eta_b));
                if (eta <= 0.0)
                    F_b = eps_b * a_gamma / (dy * dy) - eps_b * b_gamma * theta / (eps_p * dy);
                else
                    F_b = -eps_b * a_gamma / (dy * dy) + eps_b * b_gamma * theta / (eps_m * dy);
            }
            if (eta * eta_t <= 0.0) {
                double eta_p  = (eta > 0.0) ? eta : eta_t;
                double eta_m  = (eta <= 0.0) ? eta : eta_t;
                double eps_p  = (eta > 0.0) ? eps(i, j) : eps(i, j + 1);
                double eps_m  = (eta <= 0.0) ? eps(i, j) : eps(i, j + 1);
                eps_t         = eps_p * eps_m * (abs(eta_m) + abs(eta_p)) / (eps_p * abs(eta_m) + eps_m * abs(eta_p));

                auto [n1, n2] = world.normal(x, y, dx, dy);
                auto [n1_t, n2_t] = world.normal(x, y + dy, dx, dy);
                double theta      = abs(eta_t) / (abs(eta) + abs(eta_t));
                double a_gamma    = (a(i, j) * abs(eta_t) + a(i, j + 1) * abs(eta)) / (abs(eta) + abs(eta_t));
                double b_gamma = (b(i, j) * n2 * abs(eta_t) + b(i, j + 1) * n2_t * abs(eta)) / (abs(eta) + abs(eta_t));
                if (eta <= 0.0)
                    F_t = eps_t * a_gamma / (dy * dy) + eps_t * b_gamma * theta / (eps_p * dy);
                else
                    F_t = -eps_t * a_gamma / (dy * dy) - eps_t * b_gamma * theta / (eps_m * dy);
            }

            // update potential field
            double denom   = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy);
            double average = (eps_l * phi(i - 1, j) + eps_r * phi(i + 1, j)) / (dx * dx) +
                             (eps_b * phi(i, j - 1) + eps_t * phi(i, j + 1)) / (dy * dy);
            double Fx = F_l + F_r;
            double Fy = F_b + F_t;

            // sor update
            phi(i, j) = (1 - omega) * phi(i, j) + omega * (average + rho(i, j) - Fx - Fy) / denom;
        });
}

double PoissonSolver::compute_error() {
    using Kokkos::abs;
    auto& phi = world.phi;
    int nx    = world.grid.ncells[0];
    int ny    = world.grid.ncells[1];
    double err;
    Kokkos::Max<double> max_reducer(err);
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double& err) {
            max_reducer.join(err, abs(phi(i, j) - phi_old(i, j)));
        },
        max_reducer);
    return err;
}

void PoissonSolver::solve() {
    if (debug) {
        using namespace HighFive;
        auto& grid = world.grid;
        int nx     = grid.ncells[0];
        int ny     = grid.ncells[1];
        File file("data/debug_potential.hdf", File::Overwrite);
        size_t stepnum                 = 1000; // number of steps
        std::vector<size_t> dims_field = {stepnum + 1, (size_t)(nx * ny)};
        DataSetCreateProps props_field;
        props_field.add(Chunking(std::vector<hsize_t>{1, (hsize_t)(nx * ny)}));
        DataSet dataset_phi = file.createDataSet<double>("VTKHDF/CellData/phi", DataSpace(dims_field), props_field);
        std::vector<size_t> offset_field = {0, 0};
        std::vector<size_t> count_field  = {1, (size_t)(nx * ny)};

        std::vector<std::vector<double>> phi_data(1, std::vector<double>(nx * ny));

        apply_potential_boundary_conditions();
        Kokkos::deep_copy(phi_old, world.phi);
        auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                phi_data[0][i * ny + j] = phi_host(i, j);
            }
        }
        dataset_phi.select(offset_field, count_field).write(phi_data);
        red_black_update(0);
        red_black_update(1);
        for (int iter = 1; iter < stepnum; ++iter) {
            apply_potential_boundary_conditions();
            Kokkos::deep_copy(phi_old, world.phi);
            auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    phi_data[0][i * ny + j] = phi_host(i, j);
                }
            }
            offset_field = {static_cast<size_t>(iter), 0};
            dataset_phi.select(offset_field, count_field).write(phi_data);
            red_black_update(0);
            red_black_update(1);
        }
        apply_potential_boundary_conditions();
        phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                phi_data[0][i * ny + j] = phi_host(i, j);
            }
        }
        offset_field = {static_cast<size_t>(stepnum), 0};
        dataset_phi.select(offset_field, count_field).write(phi_data);
        double err = compute_error();
        Kokkos::printf("(PoissonSolver) Iteration = %d, Error(L_inf) = %e\n", stepnum, err);
    } else {
        int iter = 0;
        apply_potential_boundary_conditions();

        Kokkos::deep_copy(phi_old, world.phi);
        red_black_update(0);
        red_black_update(1);
        apply_potential_boundary_conditions();
        double err = compute_error();
        iter++;
        while (err > tol && iter < max_iter) {
            Kokkos::deep_copy(phi_old, world.phi);
            red_black_update(0);
            red_black_update(1);
            apply_potential_boundary_conditions();
            err = compute_error();
            iter++;
            if (iter % 1000 == 0) {
                Kokkos::printf("(PoissonSolver) Iteration = %d, Error(L_inf) = %e\n", iter, err);
            }
        }
        Kokkos::printf("(PoissonSolver) Iteration = %d, Error(L_inf) = %e\n", iter, err);
    }
};
