/* 1d1v vlasov simulation
 *
 * Normalization:
 * x_tilde = x / lamda_D
 * v_tilde = v / v_th
 * t_tilde = t * omega_pe
 * E_tilde = E / (m_e * v_th,e * omega_pe / e)
 * f_tilde = f / (n_0 / v_th,e)
 * The subscript tilde denotes normalized quantities.
 *
 * Normalized Vlasov-Poisson system (subscript n omitted):
 * fe_t + ve fe_x - E fe_v = 0 (electron)
 * fi_t + vi fi_x + E fi_v / mu = 0 (ion)
 * E_x = int (fi - fe) dv
 * where mu = m_i / m_e is the mass ratio of the ion to the electron.
 */
#include "grid.hpp"
#include "poisson.hpp"
#include "world.hpp"
#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

class Vlasolver {
  public:
    World& world;                  // reference to the world object
    PoissonSolver& poisson_solver; // reference to the Poisson solver
                                   //
    Vlasolver(World& world, PoissonSolver& poisson_solver)
        : world(world),
          poisson_solver(poisson_solver) {}

    void initialize_distribution() {
        using Kokkos::cos;
        using Kokkos::exp;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        // double v_d = 1.0;    // drift velocity
        // double Te = 0.5;     // electron temperature
        // double Ti = 0.01;    // ion temperature
        double alpha = 0.5; // modulation amplitude
        double k     = 0.5;

        // must assign grid and f here, otherwise, using world.grid.xxx in device region causes illegal memory access
        auto& grid    = world.grid;
        auto& f       = world.f;

        auto [nx, nv] = grid.extents();
        auto [dx, dv] = grid.spacing();
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, nv}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (j < ngc || j >= nv - ngc)
                    return;

                auto [x, v] = grid.center({i, j});
                // f(i, j, 0) = (exp(-pow(v - v_d, 2) / (2.0 * Te)) + exp(-pow(v + v_d, 2) / (2.0 * Te))) *
                //              (1 + alpha * cos(2.0 * pi * x)) / (2.0 * sqrt(2.0 * pi * Te));
                // f(i, j, 1) = exp(-pow(v, 2) / (2.0 * Ti)) / sqrt(2.0 * pi * Ti);

                f(i, j, 0) = exp(-pow(v, 2) / 2.0) * (1 + alpha * cos(k * x)) / sqrt(2.0 * pi);
                f(i, j, 1) = 1.0 / (nv * dv);
            });

        // open boundary condition in the v-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                if (j < ngc || j >= nv - ngc) {
                    f(i, j, s) = 0.0;
                }
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                if (i < ngc) {
                    f(i, j, s) = f(nx - 2 * ngc + i, j, s);
                } else if (i >= nx - ngc) {
                    f(i, j, s) = f(i - nx + 2 * ngc, j, s);
                }
            });
    }

    void compute_charge_density() const {
        auto& rho  = world.rho;
        auto& q    = world.q;
        auto& f    = world.f;
        auto& grid = world.grid;

        Kokkos::deep_copy(rho, 0.0);
        auto [dx, dv] = grid.spacing();
        auto [nx, nv] = grid.extents();
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                if (i < ngc || i >= nx - ngc)
                    return;
                Kokkos::atomic_add(&rho(i), q[s] * f(i, j, s) * dv);
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(const int i) {
                if (i < ngc) {
                    rho(i) = rho(nx - 2 * ngc + i);
                } else if (i >= nx - ngc) {
                    rho(i) = rho(i - nx + 2 * ngc);
                }
            });
    }

    void compute_potential_field() const {
        // laplacian phi = -rho
        auto& phi  = world.phi;
        auto& rho  = world.rho;
        auto& grid = world.grid;

        Kokkos::deep_copy(phi, 0.0);
        int ngc       = grid.ngc;
        auto [nx, nv] = grid.extents();
        double omega  = 2 / (1 + Kokkos::sin(Kokkos::numbers::pi / ((nx - 2 * ngc) + 1)));
        auto [dx, dv] = grid.spacing();
        double denom  = 2.0 / (dx * dx);

        for (int iter = 0; iter < 500; ++iter) {
            Kokkos::parallel_for(
                nx, KOKKOS_CLASS_LAMBDA(const int i) {
                    if (i < ngc || i >= nx - ngc)
                        return;
                    if (i % 2 != 0)
                        return; // skip red grid points

                    double average = (phi(i - 1) + phi(i + 1)) / (dx * dx);
                    Kokkos::atomic_add(&phi(i), omega * (average + rho(i) - denom * phi(i)) / denom);
                });

            Kokkos::parallel_for(
                nx, KOKKOS_CLASS_LAMBDA(const int i) {
                    if (i < ngc || i >= nx - ngc)
                        return;
                    if (i % 2 != 1)
                        return; // skip black grid points

                    double average = (phi(i - 1) + phi(i + 1)) / (dx * dx);
                    Kokkos::atomic_add(&phi(i), omega * (average + rho(i) - denom * phi(i)) / denom);
                });

            // periodic boundary condition in the x-direction
            Kokkos::parallel_for(
                nx, KOKKOS_CLASS_LAMBDA(const int i) {
                    if (i < ngc) {
                        phi(i) = phi(nx - 2 * ngc + i);
                    } else if (i >= nx - ngc) {
                        phi(i) = phi(i - nx + 2 * ngc);
                    }
                });
        }
    }

    void compute_electric_field() const {
        auto& E    = world.E;
        auto& phi  = world.phi;
        auto& grid = world.grid;

        Kokkos::deep_copy(E, 0.0);
        double dx = grid.spacing()[0];
        int nx    = grid.extents()[0];
        int ngc   = grid.ngc;
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(const int i) {
                if (i < ngc || i >= nx - ngc)
                    return;

                E(i) = -(phi(i + 1) - phi(i - 1)) / (2.0 * dx);
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(const int i) {
                if (i < ngc) {
                    E(i) = E(nx - 2 * ngc + i);
                } else if (i >= nx - ngc) {
                    E(i) = E(i - nx + 2 * ngc);
                }
            });
    }

    KOKKOS_FUNCTION
    double compute_flux(int i, int j, int s, int axis, double nu, const Kokkos::View<double***>& f) const {
        // third order upwind-biased interpolation
        using Kokkos::max;
        using Kokkos::min;
        auto f_val = KOKKOS_CLASS_LAMBDA(int offset)->double {
            if (axis == 0) {
                return f(i + offset, j, s);
            } else {
                return f(i, j + offset, s);
            }
        };

        double fm2    = f_val(-2);
        double fm1    = f_val(-1);
        double f0     = f_val(0);
        double fp1    = f_val(1);
        double fp2    = f_val(2);

        double f_max1 = max(max(fm1, f0), min(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        double f_max2 = max(max(fp1, f0), min(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        double f_min1 = min(min(fm1, f0), max(2.0 * fm1 - fm2, 2.0 * f0 - fp1));
        double f_min2 = min(min(fp1, f0), max(2.0 * fp1 - fp2, 2.0 * f0 - fm1));
        double f_max  = max(f_max1, f_max2);
        double f_min  = max(0.0, min(f_min1, f_min2));

        double flux   = 0.0;
        if (nu < 0.0) {
            // upwind
            double Lp = (f0 >= fp1) ? min(2.0 * (f0 - f_min), f0 - fp1) : max(2.0 * (f0 - f_max), f0 - fp1);
            double Lm = (fp1 >= fp2) ? min(2.0 * (f_max - f0), fp1 - fp2) : max(2.0 * (f_min - f0), fp1 - fp2);
            flux      = nu * fp1 + nu * (1.0 + nu) * (2.0 + nu) * Lp / 6.0 + nu * (1.0 - nu) * (1.0 + nu) * Lm / 6.0;
        } else {
            // downwind
            double Lp = (fp1 >= f0) ? min(2.0 * (f0 - f_min), fp1 - f0) : max(2.0 * (f0 - f_max), fp1 - f0);
            double Lm = (f0 >= fm1) ? min(2.0 * (f_max - f0), f0 - fm1) : max(2.0 * (f_min - f0), f0 - fm1);
            flux      = nu * f0 + nu * (1.0 - nu) * (2.0 - nu) * Lp / 6.0 + nu * (1.0 - nu) * (1.0 + nu) * Lm / 6.0;
        }
        return flux;
    }

    void pfc_update_along_x(double dt) const {
        auto& f       = world.f;
        auto& grid    = world.grid;

        auto [dx, dv] = grid.spacing();
        auto [nx, nv] = grid.extents();
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                if (i < ngc || i >= nx - ngc)
                    return;

                auto [x, v]     = grid.center({i, j});
                double nu       = -v * dt / dx;

                double flux_out = compute_flux(i, j, s, 0, nu, f);
                double flux_in  = compute_flux(i - 1, j, s, 0, nu, f);
                f(i, j, s) += flux_in - flux_out;
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                if (i < ngc) {
                    f(i, j, s) = f(nx - 2 * ngc + i, j, s);
                } else if (i >= nx - ngc) {
                    f(i, j, s) = f(i - nx + 2 * ngc, j, s);
                }
            });
    }

    void pfc_update_along_v(double dt) const {
        using Kokkos::max;
        using Kokkos::min;
        auto& f       = world.f;
        auto& E       = world.E;
        auto& q       = world.q;
        auto& mu      = world.mu;
        auto& grid    = world.grid;

        auto [dx, dv] = grid.spacing();
        auto [nx, nv] = grid.extents();
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int s) {
                // third order upwind-biased interpolation
                double a  = q[s] / mu[s] * E(i); // acceleration
                double nu = -a * dt / dv;
                // open boundary condition in the v-direction
                if (j < ngc || j >= nv - ngc)
                    return;

                // open boundary condition in the v-direction
                double flux_in = 0.0, flux_out = 0.0;
                if (j == 2 && a < 0.0) {
                    flux_in  = 0.0;
                    flux_out = a * dt / dv * f(i, j, s);
                } else if (j == nv - ngc && a > 0.0) {
                    flux_in  = 0.0;
                    flux_out = a * dt / dv * f(i, j, s);
                } else {
                    flux_in  = compute_flux(i, j - 1, s, 1, nu, f);
                    flux_out = compute_flux(i, j, s, 1, nu, f);
                }
                f(i, j, s) += flux_in - flux_out;
            });
    }

    void advance(double dt) {
        // perform half time step shift along the x-axis
        pfc_update_along_x(dt / 2.0);
        // compute electric field
        compute_charge_density();
        compute_potential_field();
        // poisson_solver.solve();
        compute_electric_field();
        // perform shift along the v-axis
        pfc_update_along_v(dt);
        // perform second half time step shift along the x-axis
        pfc_update_along_x(dt / 2.0);
    }

    void solve() {
        using namespace HighFive;

        size_t stepnum = 500;       // number of steps
        double dt      = 1.0 / 8.0; // time step size
        auto [nx, nv]  = world.grid.extents();

        File file("data/strong_landau_new.hdf", File::Overwrite);
        std::vector<size_t> dims_dist  = {stepnum, (size_t)nx, (size_t)nv};
        std::vector<size_t> dims_field = {stepnum, (size_t)nx};
        DataSetCreateProps props_dist;
        props_dist.add(Chunking(
            std::vector<hsize_t>{1, (hsize_t)nx, (hsize_t)nv})); // enable chunking for efficient writing per step
        DataSetCreateProps props_field;
        props_field.add(
            Chunking(std::vector<hsize_t>{1, (hsize_t)nx})); // enable chunking for efficient writing per step
        DataSet dataset_fe  = file.createDataSet<double>("VTKHDF/CellData/fe", DataSpace(dims_dist), props_dist);
        DataSet dataset_fi  = file.createDataSet<double>("VTKHDF/CellData/fi", DataSpace(dims_dist), props_dist);
        DataSet dataset_rho = file.createDataSet<double>("VTKHDF/CellData/rho", DataSpace(dims_field), props_field);
        DataSet dataset_phi = file.createDataSet<double>("VTKHDF/CellData/phi", DataSpace(dims_field), props_field);
        DataSet dataset_E   = file.createDataSet<double>("VTKHDF/CellData/E", DataSpace(dims_field), props_field);

        initialize_distribution();
        compute_charge_density();
        compute_potential_field();
        // poisson_solver.solve();
        compute_electric_field();

        std::vector<size_t> offset_dist  = {0, 0, 0};
        std::vector<size_t> count_dist   = {1, (size_t)nx, (size_t)nv};
        std::vector<size_t> offset_field = {0, 0};
        std::vector<size_t> count_field  = {1, (size_t)nx};
        auto f_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        auto rho_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        auto phi_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        auto E_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        std::vector<std::vector<std::vector<double>>> fe(1,
                                                         std::vector<std::vector<double>>(nx, std::vector<double>(nv)));
        std::vector<std::vector<std::vector<double>>> fi(1,
                                                         std::vector<std::vector<double>>(nx, std::vector<double>(nv)));
        std::vector<std::vector<double>> rho_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> phi_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> E_data(1, std::vector<double>(nx));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nv; ++j) {
                fe[0][i][j]    = f_host(i, j, 0);
                fi[0][i][j]    = f_host(i, j, 1);
                rho_data[0][i] = rho_host(i);
                phi_data[0][i] = phi_host(i);
                E_data[0][i]   = E_host(i);
            }
        }
        dataset_fe.select(offset_dist, count_dist).write(fe);
        dataset_fi.select(offset_dist, count_dist).write(fi);
        dataset_rho.select(offset_field, count_field).write(rho_data);
        dataset_phi.select(offset_field, count_field).write(phi_data);
        dataset_E.select(offset_field, count_field).write(E_data);

        Kokkos::printf("Step %d completed.\n", 0);

        for (int step = 1; step < stepnum; ++step) {
            advance(dt);
            Kokkos::printf("Step %d completed.\n", step);
            f_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
            rho_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
            phi_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
            E_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
            offset_dist  = {static_cast<size_t>(step), 0, 0};
            offset_field = {static_cast<size_t>(step), 0};
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < nv; ++j) {
                    fe[0][i][j]    = f_host(i, j, 0);
                    fi[0][i][j]    = f_host(i, j, 1);
                    rho_data[0][i] = rho_host(i);
                    phi_data[0][i] = phi_host(i);
                    E_data[0][i]   = E_host(i);
                }
            }
            dataset_fe.select(offset_dist, count_dist).write(fe);
            dataset_fi.select(offset_dist, count_dist).write(fi);
            dataset_rho.select(offset_field, count_field).write(rho_data);
            dataset_phi.select(offset_field, count_field).write(phi_data);
            dataset_E.select(offset_field, count_field).write(E_data);
        }
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::Array<double, 2> origin        = {0.0, -6.0};                     // origin of the grid
    Kokkos::Array<double, 2> size          = {4 * Kokkos::numbers::pi, 12.0}; // size of the grid
    Kokkos::Array<int, 2> ncells_interior  = {32, 64};    // number of cells in the grid (excluding ghost cells)
    int ngc                                = 3;           // number of ghost cells on each side
    Kokkos::Array<double, 2> charge_number = {-1.0, 1.0}; // charge number of the particle
    Kokkos::Array<double, 2> mass_ratio    = {1.0, 1e6};  // mass ratio of the particle (relative to the electron mass)
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid, charge_number, mass_ratio);
    PoissonSolver poisson_solver(world);
    Vlasolver vlasolver(world, poisson_solver);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
    Kokkos::fence();
}
