#pragma once
#include "grid.hpp"
#include <KokkosCore_Config_SetupBackend.hpp>
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
template <typename WorldType>
struct World {
    Grid& grid;
    Kokkos::View<double****> f; // distribution function f(x,y,vx,vy) of ion
    // Kokkos::View<double****> flux; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_l; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_r; // storing fluxes to update distribution function
    // Kokkos::View<double****> flux_1st; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_l; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_r; // storing first order fluxes to update distribution function
    // Kokkos::View<double****> ep; // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_l; // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_r; // storing first order fluxes to update distribution function
    Kokkos::View<double**> n;      // number density of ion
    Kokkos::View<double**> rho;    // ion charge density
    Kokkos::View<double**> phi;    // potential field (assuming Boltzmann distribution for electron)
    Kokkos::View<double***> E;     // Ex(x,y), E_y(x,y)
    Kokkos::View<double**> eps;    // permittivity field
    Kokkos::View<double**> a;      // jump condition for poisson
    Kokkos::View<double**> b;      // jump condition for poisson

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
        // flux                    = Kokkos::View<double****>("flux", nx, ny, nvx, nvy);
        // flux_1st                = Kokkos::View<double****>("flux_1st", nx, ny, nvx, nvy);
        // ep                      = Kokkos::View<double****>("ep", nx, ny, nvx, nvy);
        flux_l     = Kokkos::View<double****>("flux_l", nx, ny, nvx, nvy);
        flux_r     = Kokkos::View<double****>("flux_r", nx, ny, nvx, nvy);
        flux_1st_l = Kokkos::View<double****>("flux_1st_l", nx, ny, nvx, nvy);
        flux_1st_r = Kokkos::View<double****>("flux_1st_r", nx, ny, nvx, nvy);
        ep_l       = Kokkos::View<double****>("ep_l", nx, ny, nvx, nvy);
        ep_r       = Kokkos::View<double****>("ep_r", nx, ny, nvx, nvy);
        n          = Kokkos::View<double**>("n", nx, ny);
        rho        = Kokkos::View<double**>("rho", nx, ny);
        phi        = Kokkos::View<double**>("phi", nx, ny);
        E          = Kokkos::View<double***>("E", nx, ny, 2);
        eps        = Kokkos::View<double**>("eps", nx, ny);
        a          = Kokkos::View<double**>("a", nx, ny); // jump condition for poisson
        b          = Kokkos::View<double**>("b", nx, ny); // jump condition for poisson
        Kokkos::deep_copy(f, 0.0);
        // Kokkos::deep_copy(flux, 0.0);
        Kokkos::deep_copy(flux_l, 0.0);
        Kokkos::deep_copy(flux_r, 0.0);
        Kokkos::deep_copy(flux_1st_l, 0.0);
        Kokkos::deep_copy(flux_1st_r, 0.0);
        Kokkos::deep_copy(ep_l, 0.0);
        Kokkos::deep_copy(ep_r, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(E, 0.0);
        Kokkos::deep_copy(eps, 1.0);
        Kokkos::deep_copy(a, 0.0);
        Kokkos::deep_copy(b, 0.0);
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
    double surface(double x, double y) const {
        // example 4 plasma sheath from IFE-CSL
        // return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
        // debug use, a square immersed object near left boundary
        // return Kokkos::max(Kokkos::abs(x - 0.15) - 0.04, Kokkos::abs(y - 0.1) - 0.1);
        // return x + 1;
        // return 0.0;
        return static_cast<WorldType*>(this)->surface(x, y);
    }

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
        // using Kokkos::abs;
        // using Kokkos::pow;
        // using Kokkos::sqrt;
        // double norm = sqrt(pow(x - 0.375, 2) + pow(y, 2));
        // return {(x - 0.375) / norm, y / norm};
        return static_cast<WorldType*>(this)->normal(x, y, dx, dy);
    }
};
