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

    velocity_ranges[0] = origin[2];
    velocity_ranges[1] = origin[3];
    velocity_ranges[2] = origin[2] + size[2];
    velocity_ranges[3] = origin[3] + size[3];

    space_ranges[0]    = origin[0];
    space_ranges[1]    = origin[1];
    space_ranges[2]    = origin[0] + size[0];
    space_ranges[3]    = origin[1] + size[1];
}
