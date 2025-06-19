#include "grid.hpp"
#include <Kokkos_Array.hpp>

Grid::Grid(const Kokkos::Array<double, 2> origin,
           const Kokkos::Array<double, 2> size,
           const Kokkos::Array<int, 2> ncells_interior,
           int ngc)
    : origin(origin),
      size(size),
      ngc(ngc) {
    // Add ghost cells to the specified cell counts
    ncells[0]  = ncells_interior[0] + 2 * ngc;
    ncells[1]  = ncells_interior[1] + 2 * ngc;

    spacing[0] = size[0] / ncells_interior[0]; // spacing in the x-direction
    spacing[1] = size[1] / ncells_interior[1]; // spacing in the v-direction
}
