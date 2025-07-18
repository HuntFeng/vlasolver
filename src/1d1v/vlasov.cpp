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
        // double alpha = 0.5; // modulation amplitude
        // double k     = 0.5;
        double Te = 1.0; // electron temperature, 1ev
        double Ti = 0.1; // ion temperature, 0.1ev

        // must assign grid and f here, otherwise, using world.grid.xxx in device region causes illegal memory access
        auto& grid    = world.grid;
        auto& f       = world.f;

        auto [nx, nv] = grid.ncells;
        auto [dx, dv] = grid.spacing;
        auto [Lx, Lv] = grid.size;
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, nv}), KOKKOS_CLASS_LAMBDA(const int i, const int iv) {
                if (iv < ngc || iv >= nv - ngc)
                    return;

                auto [x, v] = grid.center({i, iv});
                double eta  = world.surface(x);
                if (eta < 0.0) {
                    f(i, iv, 0) = 0.0;
                    f(i, iv, 1) = 0.0;
                } else {
                    // f(i, iv, 0) = exp(-pow(v, 2) / 2.0) * (1 + alpha * cos(k * x)) / sqrt(2.0 * pi);
                    // f(i, iv, 1) = 1.0 / Lv;
                    f(i, iv, 0) = exp(-pow(v, 2) / (2.0 * Te) + (x - Lx) / (20 * Te));
                    f(i, iv, 1) = exp(-pow(v, 2) / (2.0 * Ti)) * Te / Ti;
                }
            });

        // open boundary condition in the v-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int s) {
                if (iv < ngc || iv >= nv - ngc) {
                    f(i, iv, s) = 0.0;
                }
            });

        // inject boundary condition at the right boundary
        Kokkos::parallel_for(
            nv, KOKKOS_CLASS_LAMBDA(const int iv) {
                for (int i = 1; i <= ngc; ++i) {
                    auto [x, v] = grid.center({nx - i, iv});
                    if (v < 0.0) {
                        f(nx - i, iv, 0) = exp(-pow(v, 2) / Te) / pi;
                        f(nx - i, iv, 1) = exp(-pow(v, 2) / Ti) / pi;
                    }
                }
            });

        // periodic boundary condition in the x-direction
        // Kokkos::parallel_for(
        //     Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int
        //     s) {
        //         if (i < ngc) {
        //             f(i, iv, s) = f(nx - 2 * ngc + i, iv, s);
        //         } else if (i >= nx - ngc) {
        //             f(i, iv, s) = f(i - nx + 2 * ngc, iv, s);
        //         }
        //     });
    }

    void extrapolate_distribution_function() const {
        auto& grid    = world.grid;
        auto& f       = world.f;
        auto [nx, nv] = grid.ncells;
        int ngc       = grid.ngc;
        double dx     = grid.spacing[0];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int s) {
                if (i < ngc || i >= nx - ngc)
                    return;

                auto [x, v] = grid.center({i, iv});

                // always extrapolate the ghost cells of immersed boundary from the interior cells
                double eta = world.surface(x);
                if (eta < 0.0)
                    return;

                double eta_l = world.surface(x - dx);
                double eta_r = world.surface(x + dx);
                auto [n1]    = world.normal(x, dx);
                // extrapolate outflow (v.n < 0), 0 inflow(v.n >= 0)
                if (eta * eta_l <= 0.0) {
                    f(i - 1, iv, s) = (v * n1 < 0.0) ? 1.5 * f(i, iv, s) - 0.5 * f(i + 1, iv, s) : 0.0;
                }
                if (eta * eta_r <= 0.0) {
                    f(i + 1, iv, s) = (v * n1 < 0.0) ? 1.5 * f(i, iv, s) - 0.5 * f(i - 1, iv, s) : 0.0;
                }
            });
    }

    void compute_charge_density() const {
        auto& rho  = world.rho;
        auto& q    = world.q;
        auto& f    = world.f;
        auto& n    = world.n;
        auto& grid = world.grid;

        Kokkos::deep_copy(rho, 0.0);
        auto [dx, dv] = grid.spacing;
        auto [nx, nv] = grid.ncells;
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int s) {
                if (i < ngc || i >= nx - ngc)
                    return;
                Kokkos::atomic_add(&n(i, s), f(i, iv, s) * dv);
                Kokkos::atomic_add(&rho(i), q[s] * f(i, iv, s) * dv);
            });

        // periodic boundary condition in the x-direction
        // Kokkos::parallel_for(
        //     nx, KOKKOS_CLASS_LAMBDA(const int i) {
        //         if (i < ngc) {
        //             rho(i) = rho(nx - 2 * ngc + i);
        //         } else if (i >= nx - ngc) {
        //             rho(i) = rho(i - nx + 2 * ngc);
        //         }
        //     });
    }

    void compute_poisson_jump_conditions() const {
        auto& a   = world.a;
        auto& b   = world.b;
        auto& rho = world.rho;
        int nx    = world.grid.ncells[0];
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(const int i) {
                a(i) = 0.0;
                b(i) = -rho(i);
            });
    }

    void compute_electric_field() const {
        auto& E    = world.E;
        auto& phi  = world.phi;
        auto& grid = world.grid;

        Kokkos::deep_copy(E, 0.0);
        double dx = grid.spacing[0];
        int nx    = grid.ncells[0];
        int ngc   = grid.ngc;
        Kokkos::parallel_for(
            nx, KOKKOS_CLASS_LAMBDA(const int i) {
                if (i < ngc || i >= nx - ngc)
                    return;

                E(i) = -(phi(i + 1) - phi(i - 1)) / (2.0 * dx);
            });

        // periodic boundary condition in the x-direction
        // Kokkos::parallel_for(
        //     nx, KOKKOS_CLASS_LAMBDA(const int i) {
        //         if (i < ngc) {
        //             E(i) = E(nx - 2 * ngc + i);
        //         } else if (i >= nx - ngc) {
        //             E(i) = E(i - nx + 2 * ngc);
        //         }
        //     });
    }

    KOKKOS_FUNCTION
    double compute_flux(int i, int iv, int s, int axis, double nu, const Kokkos::View<double***>& f) const {
        // third order upwind-biased interpolation
        using Kokkos::max;
        using Kokkos::min;
        auto f_val = KOKKOS_CLASS_LAMBDA(int offset)->double {
            if (axis == 0) {
                return f(i + offset, iv, s);
            } else {
                return f(i, iv + offset, s);
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

        auto [dx, dv] = grid.spacing;
        auto [nx, nv] = grid.ncells;
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int s) {
                if (i < ngc || i >= nx - ngc)
                    return;

                auto [x, v] = grid.center({i, iv});
                double eta  = world.surface(x);
                if (eta < 0.0)
                    return; // skip interior of immersed object
                double nu       = -v * dt / dx;

                double flux_out = compute_flux(i, iv, s, 0, nu, f);
                double flux_in  = compute_flux(i - 1, iv, s, 0, nu, f);
                f(i, iv, s) += flux_in - flux_out;
            });

        // periodic boundary condition in the x-direction
        // Kokkos::parallel_for(
        //     Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int
        //     s) {
        //         if (i < ngc) {
        //             f(i, iv, s) = f(nx - 2 * ngc + i, iv, s);
        //         } else if (i >= nx - ngc) {
        //             f(i, iv, s) = f(i - nx + 2 * ngc, iv, s);
        //         }
        //     });
    }

    void pfc_update_along_v(double dt) const {
        using Kokkos::max;
        using Kokkos::min;
        auto& f       = world.f;
        auto& E       = world.E;
        auto& q       = world.q;
        auto& mu      = world.mu;
        auto& grid    = world.grid;

        auto [dx, dv] = grid.spacing;
        auto [nx, nv] = grid.ncells;
        int ngc       = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0}, {nx, nv, 2}), KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int s) {
                // third order upwind-biased interpolation
                double a  = q[s] / mu[s] * E(i); // acceleration
                double nu = -a * dt / dv;

                // open boundary condition in the v-direction
                if (i < ngc || i >= nx - ngc || iv < ngc || iv >= nv - ngc)
                    return;

                auto [x, v] = grid.center({i, iv});
                double eta  = world.surface(x);
                if (eta < 0.0)
                    return; // skip interior of immersed object

                // open boundary condition in the v-direction
                double flux_in = 0.0, flux_out = 0.0;
                if (iv == 2 && a < 0.0) {
                    flux_in  = 0.0;
                    flux_out = a * dt / dv * f(i, iv, s);
                } else if (iv == nv - ngc && a > 0.0) {
                    flux_in  = 0.0;
                    flux_out = a * dt / dv * f(i, iv, s);
                } else {
                    flux_in  = compute_flux(i, iv - 1, s, 1, nu, f);
                    flux_out = compute_flux(i, iv, s, 1, nu, f);
                }
                f(i, iv, s) += flux_in - flux_out;
            });
    }

    void advance(double dt) {
        // perform half time step shift along the x-axis
        pfc_update_along_x(dt / 2.0);
        // extrapolate the distribution function for the ghost cells
        extrapolate_distribution_function();
        // compute electric field
        compute_charge_density();
        // compute_potential_field();
        compute_poisson_jump_conditions();
        poisson_solver.solve();
        compute_electric_field();
        // perform shift along the v-axis
        pfc_update_along_v(dt);
        // perform second half time step shift along the x-axis
        pfc_update_along_x(dt / 2.0);
    }

    void solve() {
        using namespace HighFive;

        size_t stepnum = 100; // number of steps
        auto [nx, nv]  = world.grid.ncells;

        File file("data/plasma_sheath.hdf", File::Overwrite);
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
        DataSet dataset_ne  = file.createDataSet<double>("VTKHDF/CellData/ne", DataSpace(dims_field), props_field);
        DataSet dataset_ni  = file.createDataSet<double>("VTKHDF/CellData/ni", DataSpace(dims_field), props_field);
        DataSet dataset_rho = file.createDataSet<double>("VTKHDF/CellData/rho", DataSpace(dims_field), props_field);
        DataSet dataset_phi = file.createDataSet<double>("VTKHDF/CellData/phi", DataSpace(dims_field), props_field);
        DataSet dataset_E   = file.createDataSet<double>("VTKHDF/CellData/E", DataSpace(dims_field), props_field);

        initialize_distribution();
        extrapolate_distribution_function();
        compute_charge_density();
        // compute_potential_field();
        compute_poisson_jump_conditions();
        poisson_solver.solve();
        compute_electric_field();

        std::vector<size_t> offset_dist  = {0, 0, 0};
        std::vector<size_t> count_dist   = {1, (size_t)nx, (size_t)nv};
        std::vector<size_t> offset_field = {0, 0};
        std::vector<size_t> count_field  = {1, (size_t)nx};
        auto f_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        auto n_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
        auto rho_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        auto phi_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        auto E_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        std::vector<std::vector<std::vector<double>>> fe(1,
                                                         std::vector<std::vector<double>>(nx, std::vector<double>(nv)));
        std::vector<std::vector<std::vector<double>>> fi(1,
                                                         std::vector<std::vector<double>>(nx, std::vector<double>(nv)));
        std::vector<std::vector<double>> ne_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> ni_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> rho_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> phi_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> E_data(1, std::vector<double>(nx));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nv; ++j) {
                fe[0][i][j]    = f_host(i, j, 0);
                fi[0][i][j]    = f_host(i, j, 1);
                ne_data[0][i]  = n_host(i, 0);
                ni_data[0][i]  = n_host(i, 1);
                rho_data[0][i] = rho_host(i);
                phi_data[0][i] = phi_host(i);
                E_data[0][i]   = E_host(i);
            }
        }
        dataset_fe.select(offset_dist, count_dist).write(fe);
        dataset_fi.select(offset_dist, count_dist).write(fi);
        dataset_ne.select(offset_field, count_field).write(ne_data);
        dataset_ni.select(offset_field, count_field).write(ni_data);
        dataset_rho.select(offset_field, count_field).write(rho_data);
        dataset_phi.select(offset_field, count_field).write(phi_data);
        dataset_E.select(offset_field, count_field).write(E_data);

        Kokkos::printf("Step %d completed.\n", 0);

        for (int step = 1; step < stepnum; ++step) {
            advance(world.dt);
            Kokkos::printf("Step %d completed.\n", step);
            f_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
            n_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
            rho_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
            phi_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
            E_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
            offset_dist  = {static_cast<size_t>(step), 0, 0};
            offset_field = {static_cast<size_t>(step), 0};
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < nv; ++j) {
                    fe[0][i][j]    = f_host(i, j, 0);
                    fi[0][i][j]    = f_host(i, j, 1);
                    ne_data[0][i]  = n_host(i, 0);
                    ni_data[0][i]  = n_host(i, 1);
                    rho_data[0][i] = rho_host(i);
                    phi_data[0][i] = phi_host(i);
                    E_data[0][i]   = E_host(i);
                }
            }
            dataset_fe.select(offset_dist, count_dist).write(fe);
            dataset_fi.select(offset_dist, count_dist).write(fi);
            dataset_ne.select(offset_field, count_field).write(ne_data);
            dataset_ni.select(offset_field, count_field).write(ni_data);
            dataset_rho.select(offset_field, count_field).write(rho_data);
            dataset_phi.select(offset_field, count_field).write(phi_data);
            dataset_E.select(offset_field, count_field).write(E_data);
        }
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    // Kokkos::Array<double, 2> origin        = {0.0, -6.0};                     // origin of the grid
    // Kokkos::Array<double, 2> size          = {4 * Kokkos::numbers::pi, 12.0}; // size of the grid
    Kokkos::Array<double, 2> origin        = {0.0, -10.0}; // origin of the grid
    Kokkos::Array<double, 2> size          = {20.0, 20.0}; // size of the grid
    int ngc                                = 3;            // number of ghost cells on each side
    Kokkos::Array<int, 2> ncells_interior  = {32 - 2 * ngc,
                                              64 - 2 * ngc}; // number of cells in the grid (excluding ghost cells)
    Kokkos::Array<double, 2> charge_number = {-1.0, 1.0};    // charge number of the particle
    Kokkos::Array<double, 2> mass_ratio    = {1.0, 1836.0};  // mass ratio of the particle (relative to the electron)
    double dt                              = 1.0 / 8.0;      // time step size
    Grid grid(origin, size, ncells_interior, ngc);
    World world(grid, dt, charge_number, mass_ratio);
    PoissonSolver poisson_solver(world);
    Vlasolver vlasolver(world, poisson_solver);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
    Kokkos::fence();
}
