#pragma once
#include <Kokkos_Core.hpp>

enum PoissonBCType {
    Dirichlet,
    Neumann,
    Periodic,
    None,
};

struct PoissonBCPair {
    PoissonBCType type;
    double val;

    KOKKOS_INLINE_FUNCTION
    PoissonBCPair()
        : type(PoissonBCType::None),
          val(0.0) {}

    KOKKOS_INLINE_FUNCTION
    PoissonBCPair(PoissonBCType t)
        : type(t),
          val(0.0) {}

    KOKKOS_INLINE_FUNCTION
    PoissonBCPair(PoissonBCType t, double v)
        : type(t),
          val(v) {}
};

template <typename PoissonSolverType>
class PoissonSolver {
  public:
    PoissonSolver() {}

    void solve() const { return static_cast<PoissonSolverType*>(this)->solve(); }

    void compute_electric_field() const {
        return static_cast<PoissonSolverType*>(this)->compute_electric_field();
    }
};
