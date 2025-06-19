#include "grid.hpp"
#include <Kokkos_Array.hpp>

Grid::Grid(const Kokkos::Array<double, 2>& origin_in,
           const Kokkos::Array<double, 2>& size_in,
           const Kokkos::Array<int, 2>& ncells_in_without_ghosts,
           int ngc_in)
    : origin(origin_in),
      size(size_in),
      ngc(ngc_in) {
    // Add ghost cells to the specified cell counts
    ncells[0] = ncells_in_without_ghosts[0] + 2 * ngc;
    ncells[1] = ncells_in_without_ghosts[1] + 2 * ngc;
}

KOKKOS_FUNCTION
Kokkos::Array<int, 2> Grid::extents() const { return {ncells[0], ncells[1]}; }

KOKKOS_FUNCTION
Kokkos::Array<double, 2> Grid::spacing() const {
    return {
        size[0] / (ncells[0] - 2 * ngc), // spacing in the x-direction
        size[1] / (ncells[1] - 2 * ngc)  // spacing in the v-direction
    };
}

KOKKOS_FUNCTION
Kokkos::Array<double, 2> Grid::center(const Kokkos::Array<int, 2>& coord_idx) const {
    auto [dx, dv] = spacing();
    return {
        origin[0] + (coord_idx[0] - ngc) * dx + dx / 2.0, // center in the x-direction
        origin[1] + (coord_idx[1] - ngc) * dv + dv / 2.0  // center in the v-direction
    };
}
