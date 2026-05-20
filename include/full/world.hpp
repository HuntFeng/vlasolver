#pragma once
#include "../grid.hpp"
#include "../poisson.hpp"
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
template <typename WorldType>
struct World {
    Grid& grid;
    Kokkos::View<double*****> f; // distribution function f(x,y,vx,vy) of ion and electron
    // Kokkos::View<double****> flux; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_l; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_r; // storing fluxes to update distribution function
    // Kokkos::View<double****> flux_1st; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_l; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_r; // storing first order fluxes to update distribution function
    // Kokkos::View<double****> ep; // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_l;  // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_r;  // storing first order fluxes to update distribution function
    Kokkos::View<double***> n;      // number density of species
    Kokkos::View<double**> rho;     // ion charge density
    Kokkos::View<double**> phi;     // potential field (assuming Boltzmann distribution for electron)
    Kokkos::View<double***> E;      // Ex(x,y), Ey(x,y)
    Kokkos::View<double***> normal; // n1(x,y), n2(x,y) unit normal vector
    Kokkos::View<PoissonBCPair**, Kokkos::HostSpace> poisson_bc_map;
    Kokkos::Array<double, 2> m = {1.0, 1836.0};     // relative mass of electrons and ions
    Kokkos::Array<double, 2> q = {-1.0, 1.0};       // charge number of electrons and ions
    Kokkos::Array<double, 2> T = {1.0, 1.0 / 10.0}; // relative temperature of electrons and ions

    // simulation time control
    double dt           = 0.0; // time step size
    double total_time   = 1.0; // total simulation time
    size_t total_steps  = 1;   // number of total steps
    size_t diag_steps   = 1;   // number of steps between diagnostics
    size_t current_step = 0;   // current step in the simulation

    World(Grid& grid)
        : grid(grid) {
        // Initialize the views with appropriate dimensions
        auto [nx, ny, nvx, nvy] = grid.ncells;
        f                       = Kokkos::View<double*****>("f", nx, ny, nvx, nvy, 2);
        flux_l                  = Kokkos::View<double****>("flux_l", nx, ny, nvx, nvy);
        flux_r                  = Kokkos::View<double****>("flux_r", nx, ny, nvx, nvy);
        flux_1st_l              = Kokkos::View<double****>("flux_1st_l", nx, ny, nvx, nvy);
        flux_1st_r              = Kokkos::View<double****>("flux_1st_r", nx, ny, nvx, nvy);
        ep_l                    = Kokkos::View<double****>("ep_l", nx, ny, nvx, nvy);
        ep_r                    = Kokkos::View<double****>("ep_r", nx, ny, nvx, nvy);
        n                       = Kokkos::View<double***>("n", nx, ny, 2);
        rho                     = Kokkos::View<double**>("rho", nx, ny);
        phi                     = Kokkos::View<double**>("phi", nx, ny);
        E                       = Kokkos::View<double***>("E", nx, ny, 2);
        normal                  = Kokkos::View<double***>("norm_vec", nx, ny, 2);
        poisson_bc_map          = Kokkos::View<PoissonBCPair**, Kokkos::HostSpace>("poisson_bc_map", nx, ny);
        Kokkos::deep_copy(f, 0.0);
        Kokkos::deep_copy(flux_l, 0.0);
        Kokkos::deep_copy(flux_r, 0.0);
        Kokkos::deep_copy(flux_1st_l, 0.0);
        Kokkos::deep_copy(flux_1st_r, 0.0);
        Kokkos::deep_copy(ep_l, 0.0);
        Kokkos::deep_copy(ep_r, 0.0);
        Kokkos::deep_copy(n, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(E, 0.0);
        Kokkos::deep_copy(normal, 0.0);

        construct_normal_field();
    }

    KOKKOS_INLINE_FUNCTION
    bool isclose(double val1, double val2, double rtol = 1e-12, double atol = 1e-12) const {
        return Kokkos::abs(val1 - val2) <= atol + rtol * Kokkos::abs(val2);
    }

    void construct_normal_field() {
        auto& grid              = this->grid;
        auto& normal            = this->normal;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        auto [dx, dy, dvx, dvy] = grid.spacing(0);
        int ngc                 = grid.ngc;
        // pre-compute normal field
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                auto [x, y]   = grid.center(i, j);

                double dx_eta = (-surface(x + 2 * dx, y) + 8 * surface(x + dx, y) - 8 * surface(x - dx, y) +
                                 surface(x - 2 * dx, y)) /
                                (12 * dx);
                double dy_eta = (-surface(x, y + 2 * dy) + 8 * surface(x, y + dy) - 8 * surface(x, y - dy) +
                                 surface(x, y - 2 * dy)) /
                                (12 * dy);
                double norm = sqrt(pow(dx_eta, 2) + pow(dy_eta, 2));

                // normal field
                if (isclose(norm, 0.0)) {
                    normal(i, j, 0) = 0.0;
                    normal(i, j, 1) = 0.0;
                } else {
                    normal(i, j, 0) = dx_eta / norm;
                    normal(i, j, 1) = dy_eta / norm;
                }
            });
    }

    /**
     * Expression of the immersed boundary.
     * S(x) = 0 is the surface of the immersed boundary.
     * S(x) < 0 is the exterior of the computational domain (interior of the immersed object).
     * S(x) > 0 is the interior of the computational domain (exterior of the immersed object).
     *
     * @param x The coordinate at which to evaluate the surface function.
     * @return The value of the surface function at x.
     */
    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return static_cast<const WorldType*>(this)->surface(x, y); }

    /**
     * Permittivity as function of spatial coordinate
     *
     * @param x The x coordinate at which to evaluate the permittivity.
     * @param y The y coordinate at which to evaluate the permittivity.
     * @return The permittivity at (x,y).
     */
    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return static_cast<const WorldType*>(this)->permittivity(x, y); }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const {
        return static_cast<const WorldType*>(this)->poisson_jump_condition_a(x, y);
    };

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_b(double x, double y) const {
        return static_cast<const WorldType*>(this)->poisson_jump_condition_b(x, y);
    }

    /**
     * Initialize the particle distribution function.
     * This function will be called by Vlasov solver
     */
    void initialize_distribution() { static_cast<WorldType*>(this)->initialize_distribution(); };

    /**
     * Boundary conditions for the particle distribution function.
     * This function will be called by Vlasov solver
     */
    void particle_boundary_conditions() { static_cast<WorldType*>(this)->particle_boundary_conditions(); };

    /**
     * Compute the Poisson jump conditions.
     * This function will be called before Poisson solver
     */
    // void poisson_jump_conditions() { static_cast<WorldType*>(this)->poisson_jump_conditions(); };

    /**
     * Apply boundary conditions to the potential field.
     * This function will be called by Poisson solver
     */
    void potential_boundary_conditions() { static_cast<WorldType*>(this)->potential_boundary_conditions(); };

    /**
     * Apply boundary conditions to the potential field.
     * This function will be called by Poisson solver, used in multigrid method
     *
     * @param u: Potential field.
     */
    void potential_boundary_conditions(Kokkos::View<double**>& u) {
        static_cast<WorldType*>(this)->potential_boundary_conditions(u);
    };
};
