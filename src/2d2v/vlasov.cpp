/* 2d2v vlasov simulation
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
#include <impl/Kokkos_Profiling.hpp>

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

        // must assign grid and f here, otherwise, using world.grid.xxx in device region causes illegal memory access
        auto& grid              = world.grid;
        auto& f                 = world.f;

        auto [nx, ny, nvx, nvy] = grid.ncells;
        auto [dx, dy, dvx, dvy] = grid.spacing;
        auto [Lx, Ly, Lvx, Lvy] = grid.size;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, ny, nvx, nvy}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                if (iv < ngc || iv >= nvx - ngc || jv < ngc || jv >= nvy - ngc)
                    return;

                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                double eta          = world.surface(x, y);
                // example 2 plasma sheath from IFE-CSL
                if (eta < 0.0) {
                    f(i, j, iv, jv, 0) = 0.0;
                    f(i, j, iv, jv, 1) = 0.0;
                } else {
                    f(i, j, iv, jv, 0) = exp(-(pow(vx, 2) + pow(vy, 2)) / 2.0);
                    f(i, j, iv, jv, 1) = exp(-(pow(vx, 2) + pow(vy, 2)) / 2.0);
                }
            });

        // open boundary condition in the v-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
                if (iv < ngc || iv >= nvx - ngc || jv < ngc || jv >= nvy - ngc) {
                    f(i, j, iv, jv, s) = 0.0;
                }
            });

        // inject boundary at y=Ly
        // zero-inflow boundary at y=0 will be handled in the extrapolate_distribution_function()
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {nx, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int iv, const int jv, const int s) {
                auto [x, y, vx, vy]       = grid.center({i, ny - ngc, iv, jv});
                f(i, ny - ngc, iv, jv, s) = exp(-(pow(vx, 2) + pow(vy, 2))) / pi;
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int j, const int iv, const int jv, const int s) {
                for (int i = 0; i < ngc; ++i) {
                    f(i, j, iv, jv, s) = f(nx - 2 * ngc + i, j, iv, jv, s);
                }
                for (int i = nx - ngc; i < nx; ++i) {
                    f(i, j, iv, jv, s) = f(i - nx + 2 * ngc, j, iv, jv, s);
                }
            });
    }

    void extrapolate_distribution_function() const {
        auto& grid              = world.grid;
        auto& f                 = world.f;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        double dx               = grid.spacing[0];
        double dy               = grid.spacing[1];

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                    return;

                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});

                // always extrapolate the ghost cells of immersed boundary from the interior cells
                double eta = world.surface(x, y);
                if (eta < 0.0)
                    return;

                double eta_l  = world.surface(x - dx, y);
                double eta_r  = world.surface(x + dx, y);
                double eta_b  = world.surface(x, y - dy);
                double eta_t  = world.surface(x, y + dy);
                auto [n1, n2] = world.normal(x, y, dx, dy);
                // extrapolate outflow (v.n < 0), zero-inflow(v.n >= 0)
                if (eta * eta_l <= 0.0) {
                    f(i - 1, j, iv, jv, s) =
                        (vx * n1 < 0.0) ? 1.5 * f(i, j, iv, jv, s) - 0.5 * f(i + 1, j, iv, jv, s) : 0.0;
                }
                if (eta * eta_r <= 0.0) {
                    f(i + 1, j, iv, jv, s) =
                        (vx * n1 < 0.0) ? 1.5 * f(i, j, iv, jv, s) - 0.5 * f(i - 1, j, iv, jv, s) : 0.0;
                }

                if (eta * eta_b <= 0.0) {
                    f(i, j - 1, iv, jv, s) =
                        (vy * n2 < 0.0) ? 1.5 * f(i, j, iv, jv, s) - 0.5 * f(i, j + 1, iv, jv, s) : 0.0;
                }
                if (eta * eta_t <= 0.0) {
                    f(i, j + 1, iv, jv, s) =
                        (vy * n2 < 0.0) ? 1.5 * f(i, j, iv, jv, s) - 0.5 * f(i, j - 1, iv, jv, s) : 0.0;
                }
            });
    }

    void compute_charge_density() const {
        auto& rho  = world.rho;
        auto& q    = world.q;
        auto& f    = world.f;
        auto& grid = world.grid;

        Kokkos::deep_copy(rho, 0.0);
        auto [dx, dy, dvx, dvy] = grid.spacing;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                    return;
                Kokkos::atomic_add(&rho(i, j), q[s] * f(i, j, iv, jv, s) * dvx * dvy);
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            ny, KOKKOS_CLASS_LAMBDA(const int j) {
                for (int i = 0; i < ngc; ++i) {
                    rho(i, j) = rho(nx - 2 * ngc + i, j);
                }
                for (int i = nx - ngc; i < nx; ++i) {
                    rho(i, j) = rho(i - nx + 2 * ngc, j);
                }
            });
    }

    void compute_poisson_jump_conditions() const {
        auto& a   = world.a;
        auto& b   = world.b;
        auto& rho = world.rho;
        int nx    = world.grid.ncells[0];
        int ny    = world.grid.ncells[1];
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                a(i, j) = 0.0;
                b(i, j) = -rho(i, j);
            });
    }

    void compute_electric_field() const {
        auto& E    = world.E;
        auto& phi  = world.phi;
        auto& grid = world.grid;

        Kokkos::deep_copy(E, 0.0);
        double dx = grid.spacing[0];
        double dy = grid.spacing[1];
        int nx    = grid.ncells[0];
        int ny    = grid.ncells[1];
        int ngc   = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                    return;

                E(i, j, 0) = -(phi(i + 1, j) - phi(i - 1, j)) / (2.0 * dx);
                E(i, j, 1) = -(phi(i, j + 1) - phi(i, j - 1)) / (2.0 * dy);
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            ny, KOKKOS_CLASS_LAMBDA(const int j) {
                for (int i = 0; i < ngc; ++i) {
                    E(i, j, 0) = E(nx - 2 * ngc + i, j, 0);
                    E(i, j, 1) = E(nx - 2 * ngc + i, j, 1);
                }
                for (int i = nx - ngc; i < nx; ++i) {
                    E(i, j, 0) = E(i - nx + 2 * ngc, j, 0);
                    E(i, j, 1) = E(i - nx + 2 * ngc, j, 1);
                }
            });
    }

    KOKKOS_FUNCTION
    double
    compute_flux(int i, int j, int iv, int jv, int s, int axis, double nu, const Kokkos::View<double*****>& f) const {
        // third order upwind-biased interpolation
        using Kokkos::max;
        using Kokkos::min;
        auto f_val = KOKKOS_CLASS_LAMBDA(int offset)->double {
            if (axis == 0) {
                return f(i + offset, j, iv, jv, s);
            } else if (axis == 1) {
                return f(i, j + offset, iv, jv, s);
            } else if (axis == 2) {
                return f(i, j, iv + offset, jv, s);
            } else {
                return f(i, j, iv, jv + offset, s);
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

    void pfc_update_along_space(double dt) const {
        auto& f                 = world.f;
        auto& grid              = world.grid;

        auto [dx, dy, dvx, dvy] = grid.spacing;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
                if (i < ngc || i >= nx - ngc || j < ngc || j >= ny - ngc)
                    return;

                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                double eta          = world.surface(x, y);
                if (eta < 0.0)
                    return; // skip interior of immersed object
                double flux_in = 0.0, flux_out = 0.0;
                double nu_x = -vx * dt / dx;
                flux_out += compute_flux(i, j, iv, jv, s, 0, nu_x, f);
                flux_in += compute_flux(i - 1, j, iv, jv, s, 0, nu_x, f);
                double nu_y = -vy * dt / dy;
                flux_out += compute_flux(i, j, iv, jv, s, 1, nu_y, f);
                flux_in += compute_flux(i - 1, j, iv, jv, s, 1, nu_y, f);
                f(i, j, iv, jv, s) += flux_in - flux_out;
            });

        // periodic boundary condition in the x-direction
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0}, {ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int j, const int iv, const int jv, const int s) {
                for (int i = 0; i < ngc; ++i) {
                    f(i, j, iv, jv, s) = f(nx - 2 * ngc + i, j, iv, jv, s);
                }
                for (int i = nx - ngc; i < nx; ++i) {
                    f(i, j, iv, jv, s) = f(i - nx + 2 * ngc, j, iv, jv, s);
                }
            });
    }

    void pfc_update_along_velocity(double dt) const {
        using Kokkos::max;
        using Kokkos::min;
        auto& f                 = world.f;
        auto& E                 = world.E;
        auto& q                 = world.q;
        auto& mu                = world.mu;
        auto& grid              = world.grid;

        auto [dx, dy, dvx, dvy] = grid.spacing;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, 0, 0, 0}, {nx, ny, nvx, nvy, 2}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, const int s) {
                // third order upwind-biased interpolation
                double ax    = q[s] / mu[s] * E(i, j, 0); // acceleration
                double nu_vx = -ax * dt / dvx;
                double ay    = q[s] / mu[s] * E(i, j, 1); // acceleration
                double nu_vy = -ay * dt / dvy;
                // open boundary condition in the v-direction
                if (iv < ngc || iv >= nvx - ngc || jv < ngc || jv >= nvy - ngc)
                    return;

                auto [x, y, vx, vy] = grid.center({i, j, iv, jv});
                double eta          = world.surface(x, y);
                if (eta < 0.0)
                    return; // skip interior of immersed object

                // open boundary condition in the v-direction
                double flux_in = 0.0, flux_out = 0.0;
                if (iv == ngc - 1 && ax < 0.0) {
                    flux_in += 0.0;
                    flux_out += ax * dt / dvx * f(i, j, iv, jv, s);
                } else if (iv == nvx - ngc && ax > 0.0) {
                    flux_in += 0.0;
                    flux_out += ax * dt / dvx * f(i, j, iv, jv, s);
                }

                if (jv == ngc - 1 && ay < 0.0) {
                    flux_in += 0.0;
                    flux_out += ay * dt / dvy * f(i, j, iv, jv, s);
                } else if (jv == nvy - ngc && ay > 0.0) {
                    flux_in += 0.0;
                    flux_out += ay * dt / dvy * f(i, j, iv, jv, s);
                }

                if (iv >= ngc && iv < nvx - ngc && jv >= ngc && jv < nvy - ngc) {
                    flux_in += compute_flux(i, j, iv - 1, jv, s, 2, nu_vx, f);
                    flux_in += compute_flux(i, j, iv, jv - 1, s, 3, nu_vy, f);
                    flux_out += compute_flux(i, j, iv, jv, s, 2, nu_vx, f);
                    flux_out += compute_flux(i, j, iv, jv, s, 3, nu_vy, f);
                }
                f(i, j, iv, jv, s) += flux_in - flux_out;
            });
    }

    void advance(double dt) {
        // perform half time step shift along the x-axis
        pfc_update_along_space(dt / 2.0);
        // extrapolate the distribution function for the ghost cells
        extrapolate_distribution_function();
        // compute electric field
        compute_charge_density();
        // compute_potential_field();
        compute_poisson_jump_conditions();
        poisson_solver.solve();
        compute_electric_field();
        // perform shift along the v-axis
        pfc_update_along_velocity(dt);
        // perform second half time step shift along the x-axis
        pfc_update_along_space(dt / 2.0);
    }

    void solve() {
        using namespace HighFive;

        size_t stepnum          = 10;        // number of steps
        double dt               = 1.0 / 8.0; // time step size
        auto [nx, ny, nvx, nvy] = world.grid.ncells;

        File file("data/plasma_sheath_2d.hdf", File::Overwrite);
        // std::vector<size_t> dims_dist  = {stepnum, (size_t)nx, (size_t)ny, (size_t)nvx, (size_t)nvy};
        std::vector<size_t> dims_dist  = {stepnum, (size_t)(nx * ny * nvx * nvy)};
        std::vector<size_t> dims_field = {stepnum, (size_t)(nx * ny)};
        // enable chunking for efficient writing distribution per step
        DataSetCreateProps props_dist;
        // props_dist.add(Chunking(std::vector<hsize_t>{1, (hsize_t)nx, (hsize_t)ny, (hsize_t)nvx, (hsize_t)nvy}));
        props_dist.add(Chunking(std::vector<hsize_t>{1, (hsize_t)(nx * ny * nvx * nvy)}));
        // enable chunking for efficient writing fields per step
        DataSetCreateProps props_field;
        // props_field.add(Chunking(std::vector<hsize_t>{1, (hsize_t)nx, (hsize_t)ny}));
        props_dist.add(Chunking(std::vector<hsize_t>{1, (hsize_t)(nx * ny)}));
        DataSet dataset_fe  = file.createDataSet<double>("VTKHDF/CellData/fe", DataSpace(dims_dist), props_dist);
        DataSet dataset_fi  = file.createDataSet<double>("VTKHDF/CellData/fi", DataSpace(dims_dist), props_dist);
        DataSet dataset_rho = file.createDataSet<double>("VTKHDF/CellData/rho", DataSpace(dims_field), props_field);
        DataSet dataset_phi = file.createDataSet<double>("VTKHDF/CellData/phi", DataSpace(dims_field), props_field);
        DataSet dataset_Ex  = file.createDataSet<double>("VTKHDF/CellData/Ex", DataSpace(dims_field), props_field);
        DataSet dataset_Ey  = file.createDataSet<double>("VTKHDF/CellData/Ey", DataSpace(dims_field), props_field);

        initialize_distribution();
        extrapolate_distribution_function();
        compute_charge_density();
        // compute_potential_field();
        compute_poisson_jump_conditions();
        poisson_solver.solve();
        compute_electric_field();

        std::vector<size_t> offset_dist  = {0, 0};
        std::vector<size_t> count_dist   = {1, (size_t)(nx * ny * nvx * nvy)};
        std::vector<size_t> offset_field = {0, 0};
        std::vector<size_t> count_field  = {1, (size_t)(nx * ny)};
        auto f_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        auto rho_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        auto phi_host                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        auto E_host                      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        // std::vector<std::vector<std::vector<double>>> fe(1,
        //                                                  std::vector<std::vector<double>>(nx,
        //                                                  std::vector<double>(nv)));
        // std::vector<std::vector<std::vector<double>>> fi(1,
        //                                                  std::vector<std::vector<double>>(nx,
        //                                                  std::vector<double>(nv)));
        // std::vector<std::vector<double>> rho_data(1, std::vector<double>(nx));
        // std::vector<std::vector<double>> phi_data(1, std::vector<double>(nx));
        // std::vector<std::vector<double>> E_data(1, std::vector<double>(nx));
        std::vector<std::vector<double>> fe(1, std::vector<double>(nx * ny * nvx * nvy));
        std::vector<std::vector<double>> fi(1, std::vector<double>(nx * ny * nvx * nvy));
        std::vector<std::vector<double>> rho_data(1, std::vector<double>(nx * ny));
        std::vector<std::vector<double>> phi_data(1, std::vector<double>(nx * ny));
        std::vector<std::vector<double>> Ex_data(1, std::vector<double>(nx * ny));
        std::vector<std::vector<double>> Ey_data(1, std::vector<double>(nx * ny));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int iv = 0; iv < nvx; ++iv) {
                    for (int jv = 0; jv < nvy; ++jv) {
                        fe[0][i * ny + j * nvx + iv * nvy + jv] = f_host(i, j, iv, jv, 0);
                        fi[0][i * ny + j * nvx + iv * nvy + jv] = f_host(i, j, iv, jv, 1);
                        rho_data[0][i * ny + j]                 = rho_host(i, j);
                        phi_data[0][i * ny + j]                 = phi_host(i, j);
                        Ex_data[0][i * ny + j]                  = E_host(i, j, 0);
                        Ey_data[0][i * ny + j]                  = E_host(i, j, 1);
                    }
                }
            }
        }
        dataset_fe.select(offset_dist, count_dist).write(fe);
        dataset_fi.select(offset_dist, count_dist).write(fi);
        dataset_rho.select(offset_field, count_field).write(rho_data);
        dataset_phi.select(offset_field, count_field).write(phi_data);
        dataset_Ex.select(offset_field, count_field).write(Ex_data);
        dataset_Ey.select(offset_field, count_field).write(Ey_data);

        Kokkos::printf("Step %d completed.\n", 0);

        for (int step = 1; step < stepnum; ++step) {
            advance(dt);
            Kokkos::printf("Step %d completed.\n", step);
            f_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
            rho_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
            phi_host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
            E_host       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
            offset_dist  = {static_cast<size_t>(step), 0};
            offset_field = {static_cast<size_t>(step), 0};
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int iv = 0; iv < nvx; ++iv) {
                        for (int jv = 0; jv < nvy; ++jv) {
                            fe[0][i * ny + j * nvx + iv * nvy + jv] = f_host(i, j, iv, jv, 0);
                            fi[0][i * ny + j * nvx + iv * nvy + jv] = f_host(i, j, iv, jv, 1);
                            rho_data[0][i * ny + j]                 = rho_host(i, j);
                            phi_data[0][i * ny + j]                 = phi_host(i, j);
                            Ex_data[0][i * ny + j]                  = E_host(i, j, 0);
                            Ey_data[0][i * ny + j]                  = E_host(i, j, 1);
                        }
                    }
                }
            }
            dataset_fe.select(offset_dist, count_dist).write(fe);
            dataset_fi.select(offset_dist, count_dist).write(fi);
            dataset_rho.select(offset_field, count_field).write(rho_data);
            dataset_phi.select(offset_field, count_field).write(phi_data);
            dataset_Ex.select(offset_field, count_field).write(Ex_data);
            dataset_Ey.select(offset_field, count_field).write(Ey_data);
        }
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::Array<double, DIM> origin       = {0.0, 0.0, -6.0, -6.0}; // origin of the grid
    Kokkos::Array<double, DIM> size         = {1.0, 1.0, 12.0, 12.0}; // size of the grid
    Kokkos::Array<int, DIM> ncells_interior = {10, 122, 26, 122}; // number of cells in the grid (excluding ghost cells)
    int ngc                                 = 3;                  // number of ghost cells on each side
    Kokkos::Array<double, 2> charge_number  = {-1.0, 1.0};        // charge number of the particle
    Kokkos::Array<double, 2> mass_ratio     = {1.0, 1e6}; // mass ratio of the particle (relative to the electron mass)
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
