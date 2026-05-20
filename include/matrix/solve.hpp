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
    // Forward elimination with partial pivoting
    for (int k = 0; k < N - 1; ++k) {
        // Find pivot row
        int pivot = k;
        double max_val = Kokkos::abs(A[k][k]);
        for (int i = k + 1; i < N; ++i) {
            double val = Kokkos::abs(A[i][k]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }
        // Swap rows if needed
        if (pivot != k) {
            for (int j = k; j < N; ++j) {
                double tmp = A[k][j];
                A[k][j] = A[pivot][j];
                A[pivot][j] = tmp;
            }
            double tmp = b[k];
            b[k] = b[pivot];
            b[pivot] = tmp;
        }
        // Eliminate rows below
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
