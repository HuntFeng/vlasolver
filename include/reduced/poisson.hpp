#pragma once
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
// template <typename World, typename PoissonSolverType>
// struct PoissonSolver {
//     World& world;
//
//     PoissonSolver(World& world)
//         : world(world) {}
template <typename PoissonSolverType>
class PoissonSolver {
  public:
    PoissonSolver() {}
    /**
     * Solve for the potential field phi using the Poisson equation.
     */
    void solve() const { return static_cast<PoissonSolverType*>(this)->solve(); }

    /**
     * Compute the electric field E = -grad(phi)
     */
    void compute_electric_field() const { return static_cast<PoissonSolverType*>(this)->compute_electric_field(); }
};
