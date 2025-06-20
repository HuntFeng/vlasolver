#pragma once
#include "grid.hpp"
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
struct World {
    Grid& grid;
    Kokkos::Array<double, 2> q;  // charge number of the particle
    Kokkos::Array<double, 2> mu; // mass ratio of the particle (relative to the electron mass)
    Kokkos::View<double***> f;
    Kokkos::View<double*> rho;
    Kokkos::View<double*> phi;
    Kokkos::View<double*> E;
    Kokkos::View<double*> eps;
    Kokkos::View<double*> a; // jump condition for poisson
    Kokkos::View<double*> b; // jump condition for poisson

    World(Grid& grid, Kokkos::Array<double, 2> q = {-1.0, 1.0}, Kokkos::Array<double, 2> mu = {1.0, 1e6});

    /**
     * Expression of the immersed boundary. S(x) = 0 is the surface of the immersed boundary.
     * @param x The coordinate at which to evaluate the surface function.
     * @return The value of the surface function at x.
     */
    KOKKOS_INLINE_FUNCTION
    double surface(double x) const {
        using Kokkos::abs;
        // example 1
        return x - 0.5;
        // example 2
        // return abs(x - 0.45) - 0.15;
    }

    /**
     * Normal vector at the surface.
     * @param x The coordinate at which to evaluate the normal vector.
     * @return The normal vector at x.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 1> normal(double x, double dx) const {
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;
        double eta           = surface(x);
        double eta_l         = surface(x - dx);
        double eta_r         = surface(x + dx);
        double eta_grad_x    = (eta_r - eta_l) / (2 * dx);
        double eta_grad_norm = abs(eta_grad_x);
        double nx            = eta_grad_x / eta_grad_norm;
        return {nx};
    }
};
