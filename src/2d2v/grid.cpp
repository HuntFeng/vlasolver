#include "grid.hpp"
#include <Kokkos_Array.hpp>

Grid::Grid(const Kokkos::Array<double, DIM> origin,
           const Kokkos::Array<double, DIM> size,
           const Kokkos::Array<int, DIM> ncells_interior,
           int ngc)
    : origin(origin),
      size(size),
      ngc(ngc) {
    for (int d = 0; d < DIM; ++d) {
        // Add ghost cells to the specified cell counts
        ncells[d]  = ncells_interior[d] + 2 * ngc;
        spacing[d] = size[d] / ncells_interior[d];
    }
}
