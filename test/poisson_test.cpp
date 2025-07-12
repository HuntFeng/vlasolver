#include "../src/2d2v/poisson.hpp"
#include "../src/2d2v/world.hpp"
#include <cmath>
#include <gtest/gtest.h>

void fill(Kokkos::View<double**> view) {
    int nx = view.extent(0);
    int ny = view.extent(1);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_LAMBDA(const int i, const int j) {
            if (i < 3 || i >= nx - 3 || j < 3 || j >= ny - 3) {
                view(i, j) = 0.0;
            } else {
                view(i, j) = 1.0;
            }
        });
}

TEST(poisson_test, prolong) {
    int ngc                                 = 3;
    int n_interior                          = 4;
    int n                                   = n_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {n_interior, n_interior, n_interior, n_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine   = {(size_t)n, (size_t)n};
    Kokkos::Array<size_t, 2> n_coarse = {(n_fine[0] - 2 * ngc) / 2 + 2 * ngc, (n_fine[1] - 2 * ngc) / 2 + 2 * ngc};
    Kokkos::View<double**> ec("ec", n_coarse[0], n_coarse[1]);

    fill(ec);

    Kokkos::View<double**> ef = poisson_solver.prolong(ec, n_fine);
    auto ec_h                 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ec);
    auto ef_h                 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ef);

    for (int i = ngc; i < n - ngc; ++i) {
        for (int j = ngc; j < n - ngc; ++j) {
            if (i == n_fine[0] - ngc - 1 && j == n_fine[1] - ngc - 1)
                EXPECT_DOUBLE_EQ(ef_h(i, j), 0.25);
            else if (i == n_fine[0] - ngc - 1 || j == n_fine[1] - ngc - 1)
                EXPECT_DOUBLE_EQ(ef_h(i, j), 0.5);
            else
                EXPECT_DOUBLE_EQ(ef_h(i, j), 1.0);
        }
    }
}

TEST(poisson_test, restrict) {
    int ngc                                 = 3;
    int n_interior                          = 4;
    int n                                   = n_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {n_interior, n_interior, n_interior, n_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine   = {(size_t)n, (size_t)n};
    Kokkos::Array<size_t, 2> n_coarse = {(n_fine[0] - 2 * ngc) / 2 + 2 * ngc, (n_fine[1] - 2 * ngc) / 2 + 2 * ngc};
    Kokkos::View<double**> uf("uf", n_fine[0], n_fine[1]);

    fill(uf);

    Kokkos::View<double**> uc = poisson_solver.restrict(uf, n_coarse);
    auto uc_h                 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uc);
    auto uf_h                 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uf);

    for (int i = ngc; i < n_coarse[0] - ngc; ++i) {
        for (int j = ngc; j < n_coarse[1] - ngc; ++j) {
            EXPECT_DOUBLE_EQ(uc_h(i, j), 1.0);
        }
    }
}

TEST(poisson_test, apply_boundary_not_nan) {
    int ngc                                 = 3;
    int nx_interior                         = 128;
    int ny_interior                         = 64;
    int nx                                  = nx_interior + 2 * ngc;
    int ny                                  = ny_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx_interior, ny_interior, nx_interior, ny_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine = {(size_t)nx, (size_t)ny};
    fill(world.phi);

    poisson_solver.apply_boundary(world.phi);

    auto phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);

    for (int i = ngc; i < n_fine[0] - ngc; ++i) {
        for (int j = ngc; j < n_fine[1] - ngc; ++j) {
            ASSERT_FALSE(std::isnan(phi(i, j)));
        }
    }
}

TEST(poisson_test, gauss_seidel_not_nan) {
    int ngc                                 = 3;
    int nx_interior                         = 128;
    int ny_interior                         = 64;
    int nx                                  = nx_interior + 2 * ngc;
    int ny                                  = ny_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx_interior, ny_interior, nx_interior, ny_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine = {(size_t)nx, (size_t)ny};

    fill(world.phi);

    poisson_solver.gauss_seidel(world.phi, world.rho, world.eps, world.a, world.b, 30);

    auto phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);

    for (int i = ngc; i < n_fine[0] - ngc; ++i) {
        for (int j = ngc; j < n_fine[1] - ngc; ++j) {
            ASSERT_FALSE(std::isnan(phi(i, j)));
        }
    }
}

TEST(poisson_test, nonlinear_operator_not_nan) {
    int ngc                                 = 3;
    int nx_interior                         = 128;
    int ny_interior                         = 64;
    int nx                                  = nx_interior + 2 * ngc;
    int ny                                  = ny_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx_interior, ny_interior, nx_interior, ny_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine = {(size_t)nx, (size_t)ny};

    fill(world.phi);

    poisson_solver.nonlinear_operator(world.phi, world.eps);

    auto phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);

    for (int i = ngc; i < n_fine[0] - ngc; ++i) {
        for (int j = ngc; j < n_fine[1] - ngc; ++j) {
            ASSERT_FALSE(std::isnan(phi(i, j)));
        }
    }
}

TEST(poisson_test, v_cycle_not_nan) {
    int ngc                                 = 3;
    int nx_interior                         = 128;
    int ny_interior                         = 64;
    int nx                                  = nx_interior + 2 * ngc;
    int ny                                  = ny_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx_interior, ny_interior, nx_interior, ny_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine = {(size_t)nx, (size_t)ny};

    poisson_solver.apply_boundary(world.phi);
    poisson_solver.v_cycle(world.phi, world.rho, world.eps, world.a, world.b, 0);

    auto phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);

    for (int i = ngc; i < n_fine[0] - ngc; ++i) {
        for (int j = ngc; j < n_fine[1] - ngc; ++j) {
            ASSERT_FALSE(Kokkos::isnan(phi(i, j)));
        }
    }
}

TEST(poisson_test, multiple_v_cycle_not_nan) {
    int ngc                                 = 3;
    int nx_interior                         = 128;
    int ny_interior                         = 64;
    int nx                                  = nx_interior + 2 * ngc;
    int ny                                  = ny_interior + 2 * ngc;
    Kokkos::Array<double, DIM> origin       = {0, 0, -1, -1}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1, 1, 2, 2};   // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {nx_interior, ny_interior, nx_interior, ny_interior};
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid);
    PoissonSolver poisson_solver(world);
    Kokkos::Array<size_t, 2> n_fine = {(size_t)nx, (size_t)ny};

    auto& u                         = world.phi;
    auto& rho                       = world.rho;
    auto& eps                       = world.eps;
    auto& b                         = world.b;
    auto& a                         = world.a;
    poisson_solver.apply_boundary(u);
    for (int iter = 0; iter < 5; ++iter) {
        Kokkos::printf("Iteration %d\n", iter);
        poisson_solver.v_cycle(u, rho, eps, a, b, 0);
    }

    auto phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);

    for (int i = ngc; i < n_fine[0] - ngc; ++i) {
        for (int j = ngc; j < n_fine[1] - ngc; ++j) {
            ASSERT_FALSE(Kokkos::isnan(phi(i, j)));
        }
    }
}
