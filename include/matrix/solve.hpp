#pragma once
#include <Kokkos_Core.hpp>

/**
 * Helper function to solve a small (at most 4x4) linear system using Gaussian elimination
 *
 * @param A Coefficient matrix (modified in place)
 * @param b Right-hand side vector (modified in place to contain the solution)
 */
template <size_t N>
KOKKOS_INLINE_FUNCTION void solve_linear_system(Kokkos::Array<Kokkos::Array<double, N>, N>& A,
                                                Kokkos::Array<double, N>& b) {
    // Forward elimination
    for (int k = 0; k < N - 1; ++k) {
        for (int i = k + 1; i < N; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < N; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < N; ++j) {
            sum -= A[i][j] * b[j];
        }
        b[i] = sum / A[i][i];
    }
}
