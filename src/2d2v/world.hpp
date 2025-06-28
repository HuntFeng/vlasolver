#pragma once
#include "grid.hpp"
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
struct World {
    Grid& grid;
    Kokkos::Array<double, 2> q;     // charge number of the particle
    Kokkos::Array<double, 2> mu;    // mass ratio of the particle (relative to the electron mass)
    Kokkos::View<double*****> f;    // distribution function f(x,y,vx,vy,s) where s is the species index
    Kokkos::View<double*****> flux; // storing fluxes to update distribution function
    Kokkos::View<double***> n;      // number density
    Kokkos::View<double**> rho;
    Kokkos::View<double**> phi;
    Kokkos::View<double***> E; // Ex(x,y), E_y(x,y)
    Kokkos::View<double**> eps;
    Kokkos::View<double**> a; // jump condition for poisson
    Kokkos::View<double**> b; // jump condition for poisson

    // simulation time control
    double dt           = 0.0; // time step size
    double total_time   = 1.0; // total simulation time
    size_t total_steps  = 1;   // number of total steps
    size_t diag_steps   = 1;   // number of steps between diagnostics
    size_t current_step = 0;   // current step in the simulation

    World(Grid& grid, Kokkos::Array<double, 2> q = {-1.0, 1.0}, Kokkos::Array<double, 2> mu = {1.0, 1e6});

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
        return Kokkos::pow(x - 0.375, 2) + Kokkos::pow(y, 2) - Kokkos::pow(0.125, 2);
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
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;
        double eta           = surface(x, y);
        double eta_l         = surface(x - dx, y);
        double eta_r         = surface(x + dx, y);
        double eta_b         = surface(x, y - dy);
        double eta_t         = surface(x, y + dy);
        double eta_grad_x    = (eta_r - eta_l) / (2 * dx);
        double eta_grad_y    = (eta_t - eta_b) / (2 * dy);
        double eta_grad_norm = sqrt(pow(eta_grad_x, 2) + pow(eta_grad_y, 2));
        // to avoid confusion, recommand to use n1 and n2 normal vector instead of nx and ny
        double n1 = eta_grad_x / eta_grad_norm;
        double n2 = eta_grad_y / eta_grad_norm;
        return {n1, n2};
    }
};
