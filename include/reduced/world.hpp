#pragma once
#include "../grid.hpp"
#include "../poisson.hpp"
#include <KokkosCore_Config_SetupBackend.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
template <typename WorldType>
struct World {
    Grid& grid;
    Kokkos::View<double****> f;        // distribution function f(x,y,vx,vy) of ion
    Kokkos::View<double****> flux;     // storing fluxes at cell interfaces
    Kokkos::View<double****> flux_1st; // storing first order fluxes at cell interfaces
    Kokkos::View<double****> ep_l;     // storing flux limiters at cell interfaces
    Kokkos::View<double****> ep_r;     // storing flux limiters at cell interfaces
    Kokkos::View<double**> n;          // number density of ion
    Kokkos::View<double**> rho;        // ion charge density
    Kokkos::View<double**> phi;        // potential field (assuming Boltzmann distribution for electron)
    Kokkos::View<double***> E;         // Ex(x,y), E_y(x,y)
    Kokkos::View<PoissonBCPair**, Kokkos::HostSpace> poisson_bc_map;
    Kokkos::View<double***> norm_vec; // normal vector

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
        f                       = Kokkos::View<double****>("f", nx, ny, nvx, nvy);
        flux                    = Kokkos::View<double****>("flux", nx, ny, nvx, nvy);
        flux_1st                = Kokkos::View<double****>("flux_1st", nx, ny, nvx, nvy);
        ep_l                    = Kokkos::View<double****>("ep_l", nx, ny, nvx, nvy);
        ep_r                    = Kokkos::View<double****>("ep_r", nx, ny, nvx, nvy);
        n                       = Kokkos::View<double**>("n", nx, ny);
        rho                     = Kokkos::View<double**>("rho", nx, ny);
        phi                     = Kokkos::View<double**>("phi", nx, ny);
        E                       = Kokkos::View<double***>("E", nx, ny, 2);
        poisson_bc_map          = Kokkos::View<PoissonBCPair**, Kokkos::HostSpace>("poisson_bc_map", nx, ny);
        norm_vec                  = Kokkos::View<double***>("norm_vec", nx, ny, 2);
        Kokkos::deep_copy(flux, 0.0);
        Kokkos::deep_copy(flux_1st, 0.0);
        Kokkos::deep_copy(ep_l, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(E, 0.0);
        Kokkos::deep_copy(norm_vec, 0.0);
    }
    /**
     * Expression of the immersed boundary.
     * S(x) = 0 is the surface of the immersed boundary.
     * S(x) < 0 is the exterior of the computational domain (interior of the immersed object).
     * S(x) > 0 is the interior of the computational domain (exterior of the immersed object).
     *
     * @param x The x coordinate at which to evaluate the surface function.
     * @param y The y coordinate at which to evaluate the surface function.
     * @return The value of the surface function at (x,y).
     */
    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return static_cast<WorldType*>(this)->surface(x, y); }

    /**
     * Unit normal vector at the surface.
     * The normal vector is pointing inward, i.e. into the computational domain.
     *
     * @param x The x coordinate at which to evaluate the normal vector.
     * @param y The y coordinate at which to evaluate the normal vector.
     * @param dx Spacing in the x direction.
     * @param dy Spacing in the y direction.
     * @return The unit normal vector at (x,y).
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> normal(double x, double y, double dx, double dy) const {
        return static_cast<WorldType*>(this)->normal(x, y, dx, dy);
    }

    /**
     * Permittivity as function of spatial coordinate
     *
     * @param x The x coordinate at which to evaluate the permittivity.
     * @param y The y coordinate at which to evaluate the permittivity.
     * @return The permittivity at (x,y).
     */
    KOKKOS_INLINE_FUNCTION
    double permittivity(double x, double y) const { return static_cast<WorldType*>(this)->permittivity(x, y); }

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_a(double x, double y) const {
        return static_cast<WorldType*>(this)->poisson_jump_condition_a(x, y);
    };

    KOKKOS_INLINE_FUNCTION
    double poisson_jump_condition_b(double x, double y) const {
        return static_cast<WorldType*>(this)->poisson_jump_condition_b(x, y);
    };

    /**
     * Boundary conditions for the particle distribution function.
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
    void poisson_jump_conditions() { static_cast<WorldType*>(this)->poisson_jump_conditions(); };

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
