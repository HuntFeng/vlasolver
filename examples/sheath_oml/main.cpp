#include "full/vlasov.hpp"
#include "full/world.hpp"
#include "full/writer.hpp"
#include "grid.hpp"
#include "poisson_1st_order.hpp"
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct ImmersedWorld : World<ImmersedWorld> {
    static constexpr double Y_WALL = 2.0;

    // all quantities are normalized by electron parameters
    double phi_w  = -Kokkos::log(Kokkos::sqrt(m[1] / (2 * Kokkos::numbers::pi * m[0]))); // wall potential (estimate)
    double v_th_e = Kokkos::sqrt(T[0] / m[0]);                                           // electron thermal velocity
    double v_th_i = Kokkos::sqrt(T[1] / m[1]);                                           // ion thermal velocity
    double u0     = Kokkos::sqrt(T[0] / m[1]);                                           // Bohm velocity

    // accumulated surface charge on the immersed interface (the OML state variable)
    double sigma_w_host = 0.0;

    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {
        construct_surface();      // fill eta
        construct_permittivity(); // fill eps
        construct_normal_field(); // base method, reads eta
    }

    // Fill the surface field eta(i,j) = S(x,y) over the full domain (including ghost cells).
    void construct_surface() {
        auto& grid              = this->grid;
        auto& eta               = this->eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                eta(i, j)   = y - Y_WALL;
            });
    }

    // Conductor-like interior so the immersed body is (nearly) equipotential.
    void construct_permittivity() {
        auto& grid              = this->grid;
        auto& eta               = this->eta;
        auto& eps               = this->eps;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) { eps(i, j) = (eta(i, j) <= 0.0) ? 4.0 : 1.0; });
    }

    void initialize_distribution() {
        using Kokkos::exp;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y] = grid.center(i, j);
                phi(i, j)   = (eta(i, j) >= 0.0) ? phi_w * exp(-(y - Y_WALL) / 2.5) : phi_w;
            });

        Kokkos::deep_copy(f, 0.0);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // electron
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    if (eta(i, j) >= 0.0) {
                        double v_ce = sqrt(2 * (phi(i, j) - phi_w) / m[0]);
                        f(i, j, iv, jv, 0) =
                            (vy <= v_ce) ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                               (2.0 * pi * pow(v_th_e, 2))
                                         : 0.0;
                    }
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    if (eta(i, j) >= 0.0) {
                        double v_ci        = -sqrt(2 * Kokkos::abs(phi(i, j)) / m[1]); // ion cutoff velocity
                        f(i, j, iv, jv, 1) = (vy <= v_ci)
                                                 ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
                                                       (2.0 * pow(v_th_i, 2))) /
                                                       (2.0 * pi * pow(v_th_i, 2))
                                                 : 0.0;
                    }
                };
            });

        // periodic boundary conditions for left and right boundaries
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(0, ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL));
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    };

    void particle_boundary_conditions() {
        using Kokkos::exp;
        using Kokkos::pow;
        using Kokkos::sqrt;
        using Kokkos::numbers::pi;
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0, ngc, ngc}, {nx, ny, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv) {
                // electron
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 0);
                    double n1           = normal(i, j, 0);
                    double n2           = normal(i, j, 1);
                    double v_ce         = sqrt(2 * (0.0 - phi_w) / m[0]); // electron cutoff velocity
                    if (j < ngc && vy > 0.0) {
                        f(i, j, iv, jv, 0) = 0.0; // bottom domain boundary, zero-inflow
                    } else if (eta(i, j) < 0.0 && vx * n1 + vy * n2 > 0.0) {
                        f(i, j, iv, jv, 0) = 0.0; // immersed wall absorbs, emits nothing back into the plasma
                    } else if (j >= ny - ngc) {
                        // top reservoir, with dynamic electron density adjustment
                        double ne = (n(i, ny - ngc - 1, 0) > 0.0) ? n(i, ny - ngc - 1, 1) / n(i, ny - ngc - 1, 0) : 1.0;
                        f(i, j, iv, jv, 0) =
                            (vy <= v_ce) ? exp(-(pow(vx, 2) + pow(vy, 2)) / (2.0 * pow(v_th_e, 2)) + phi(i, j)) /
                                               (2.0 * pi * pow(v_th_e, 2)) * ne
                                         : 0.0;
                    }
                };
                // ion
                {
                    auto [x, y, vx, vy] = grid.center(i, j, iv, jv, 1);
                    double n1           = normal(i, j, 0);
                    double n2           = normal(i, j, 1);
                    double v_ci         = 0.0; // ion cutoff velocity, since phi(top) = 0
                    if (j < ngc && vy > 0.0) {
                        f(i, j, iv, jv, 1) = 0.0; // bottom domain boundary, zero-inflow
                    } else if (eta(i, j) < 0.0 && vx * n1 + vy * n2 > 0.0) {
                        f(i, j, iv, jv, 1) = 0.0; // immersed wall absorbs, emits nothing back into the plasma
                    } else if (j >= ny - ngc) {
                        f(i, j, iv, jv, 1) = (vy <= v_ci)
                                                 ? exp(-(pow(vx, 2) + pow(sqrt(pow(vy, 2) - pow(v_ci, 2)) - u0, 2)) /
                                                       (2.0 * pow(v_th_i, 2))) /
                                                       (2.0 * pi * pow(v_th_i, 2))
                                                 : 0.0;
                    }
                };
            });

        // periodic boundary conditions for left and right boundaries
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(0, ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(nx - 2 * ngc, nx - ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                            Kokkos::ALL));
        Kokkos::deep_copy(
            Kokkos::subview(f, Kokkos::make_pair(nx - ngc, nx), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(f, Kokkos::make_pair(ngc, 2 * ngc), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    };

    void potential_boundary_conditions() {
        auto& grid              = this->grid;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        int ngc                 = grid.ngc;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (i < ngc || i >= nx - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Periodic);
                else if (j >= ny - ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Dirichlet, 0.0);
                else if (j < ngc)
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::Neumann, 0.0); // floating wall: field set by sigma_w
                else
                    poisson_bc_map(i, j) = PoissonBCPair(PoissonBCType::None);
            });

        auto [_dx, _dy, dvx_e, dvy_e]   = grid.spacing(0);
        auto [_dx, _dy, dvx_i, dvy_i] = grid.spacing(1);
        double flux                   = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({ngc, ngc, ngc, ngc}, {nx - ngc, ny - ngc, nvx - ngc, nvy - ngc}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int iv, const int jv, double& acc) {
                // keep only the first fluid row above the interface
                if (!(eta(i, j) > 0.0 && eta(i, j - 1) <= 0.0))
                    return;
                {
                    auto [_xc, _yc, _vx, vy_e] = grid.center(i, j, iv, jv, 0);
                    if (vy_e < 0.0)
                        acc += q[0] * (-vy_e) * f(i, j, iv, jv, 0) * dvx_e * dvy_e;
                }
                {
                    auto [_xc, _yc, _vx, vy_i] = grid.center(i, j, iv, jv, 1);
                    if (vy_i < 0.0)
                        acc += q[1] * (-vy_i) * f(i, j, iv, jv, 1) * dvx_i * dvy_i;
                }
            },
            flux);
        double flux_net = flux / grid.ncells_interior[0];
        sigma_w_host += flux_net * dt;
    }

    // Fill the Poisson jump condition fields. Being a host method filling fields, it can
    // freely read the accumulated surface charge (a host scalar) updated each step, so the
    // time-dependent normal-derivative jump is trivial to express.
    void poisson_jump_conditions() {
        auto& grid              = this->grid;
        auto& jump_a            = this->jump_a;
        auto& jump_b            = this->jump_b;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        double sigma            = sigma_w_host;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                jump_a(i, j) = 0.0;    // no jump in the potential across the interface
                jump_b(i, j) = -sigma; // jump in the normal derivative from accumulated surface charge
            });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_file_path>\n";
        return 1;
    }
    std::string input_file_path = argv[1];
    INIReader reader(input_file_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load input file, please specify path\n";
        return 1;
    }
    double x_min_e            = reader.GetReal("grid_electron", "x_min", 0.0);
    double y_min_e            = reader.GetReal("grid_electron", "y_min", 0.0);
    double vx_min_e           = reader.GetReal("grid_electron", "vx_min", 0.0);
    double vy_min_e           = reader.GetReal("grid_electron", "vy_min", 0.0);
    double Lx_e               = reader.GetReal("grid_electron", "Lx", 1.0);
    double Ly_e               = reader.GetReal("grid_electron", "Ly", 1.0);
    double Lvx_e              = reader.GetReal("grid_electron", "Lvx", 1.0);
    double Lvy_e              = reader.GetReal("grid_electron", "Lvy", 1.0);

    double x_min_i            = reader.GetReal("grid_ion", "x_min", 0.0);
    double y_min_i            = reader.GetReal("grid_ion", "y_min", 0.0);
    double vx_min_i           = reader.GetReal("grid_ion", "vx_min", 0.0);
    double vy_min_i           = reader.GetReal("grid_ion", "vy_min", 0.0);
    double Lx_i               = reader.GetReal("grid_ion", "Lx", 1.0);
    double Ly_i               = reader.GetReal("grid_ion", "Ly", 1.0);
    double Lvx_i              = reader.GetReal("grid_ion", "Lvx", 1.0);
    double Lvy_i              = reader.GetReal("grid_ion", "Lvy", 1.0);

    int nx_intr               = reader.GetInteger("grid", "nx", 128);
    int ny_intr               = reader.GetInteger("grid", "ny", 128);
    int nvx_intr              = reader.GetInteger("grid", "nvx", 128);
    int nvy_intr              = reader.GetInteger("grid", "nvy", 128);
    int ngc                   = reader.GetInteger("grid", "ngc", 3);
    double dt                 = reader.GetReal("world", "dt", 1e-3);
    double total_time         = reader.GetReal("world", "total_time", 1.0);
    int total_steps           = reader.GetInteger("world", "total_steps", 1000);
    int diag_steps            = reader.GetInteger("world", "diag_steps", 10);
    std::string output_folder = reader.Get("output", "folder", "data/plasma_past_charged_cylinder");
    std::string output_prefix = reader.Get("output", "prefix", "output");

    Kokkos::printf("Input parameters:\n");
    Kokkos::printf("Phase space (x,y,vx,vy):\nElectron: [%f, %f, %f, %f]x[%f, %f, %f, %f]\nIon: [%f, %f, %f, %f]x[%f, "
                   "%f, %f, %f]\n",
                   x_min_e, y_min_e, vx_min_e, vy_min_e, x_min_e + Lx_e, y_min_e + Ly_e, vx_min_e + Lvx_e,
                   vy_min_e + Lvy_e, x_min_i, y_min_i, vx_min_i, vy_min_i, x_min_i + Lx_i, y_min_i + Ly_i,
                   vx_min_i + Lvx_i, vy_min_i + Lvy_i);
    Kokkos::printf("Grid size, interior (nx,ny,nvx,nvy): [%d, %d, %d, %d]\n", nx_intr, ny_intr, nvx_intr, nvy_intr);
    Kokkos::printf("Simulation control: dt: %f, total_time: %f, total_steps: %d, diag_steps: %d\n", dt, total_time,
                   total_steps, diag_steps);

    double Te     = 1.0;                   // electron temperature
    double Ti     = 0.1;                   // ion temperature normalized to Te
    double me     = 1.0;                   // electron mass
    double mi     = 2 * 1836.0;            // ion mass, normalized to me
    double v_th_e = Kokkos::sqrt(Te / me); // electron thermal velocity
    double v_th_i = Kokkos::sqrt(Ti / mi); // ion thermal velocity, normalized to v_th_e
    double u0     = Kokkos::sqrt(Te / mi); // Bohm velocity, normalized to v_th_e

    Grid grid({nx_intr, ny_intr, nvx_intr, nvy_intr}, ngc);
    grid.set_grid({x_min_e, y_min_e, vx_min_e, vy_min_e}, {Lx_e, Ly_e, Lvx_e, Lvy_e}, 0); // electrons
    grid.set_grid({x_min_i, y_min_i, vx_min_i * v_th_i, vy_min_i * v_th_i},
                  {Lx_i, Ly_i, Lvx_i * v_th_i, Lvy_i * v_th_i}, 1); // ions

    ImmersedWorld world(grid);
    world.dt          = dt;                                  // time step size
    world.total_time  = total_time;                          // total simulation time
    world.total_steps = total_steps;                         // number of total_steps
    world.diag_steps  = diag_steps;                          // number of steps between diagnostics
    world.m           = Kokkos::Array<double, 2>{me, mi};    // relative mass of electrons and ions
    world.q           = Kokkos::Array<double, 2>{-1.0, 1.0}; // charge number of electrons and ions
    world.T           = Kokkos::Array<double, 2>{Te, Ti};    // relative temperature of electrons and ions
    world.v_th_e      = v_th_e;
    world.v_th_i      = v_th_i;
    world.u0          = u0;

    PoissonSolver1stOrder poisson_solver(world, 1e-6, 1e5, 1.0); // set lower relaxation for convergence
    Writer writer(world, output_folder, output_prefix, {"ni", "ne", "phi"});
    Vlasolver vlasolver(world, poisson_solver, writer);

    Kokkos::Timer timer;
    double start_time = timer.seconds();
    vlasolver.solve();
    double end_time = timer.seconds();
    Kokkos::printf("Total time taken: %f seconds\n", end_time - start_time);
}
