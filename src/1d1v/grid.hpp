#pragma once
#include <Kokkos_Core.hpp>

class Grid {
  private:
    Kokkos::Array<double, 2> origin; // origin of the grid
    Kokkos::Array<double, 2> size;   // size of the grid
    Kokkos::Array<int, 2> ncells;    // number of all cells in the grid

  public:
    int ngc; // number of ghost cells on each side
    /**
     * Constructor to initialize Grid with custom parameters.
     *
     * @param origin_in Origin coordinates of the grid
     * @param size_in Size of the grid
     * @param ncells_in Number of cells in the grid (excluding ghost cells)
     * @param (optional) ngc_in Number of ghost cells on each side, defaults to
     * 3
     */
    Grid(const Kokkos::Array<double, 2> origin,
         const Kokkos::Array<double, 2> size,
         const Kokkos::Array<int, 2> ncells_without_ghosts,
         int ngc = 3);

    /**
     * Returns the spacing of the grid in each direction.
     * Spacing is calculated as the grid size / number of cells (excluding ghost
     *cells) in that direction.
     *
     * @return A Kokkos::Array containing the spacing in the x and v directions.
     **/
    // KOKKOS_FUNCTION
    // Kokkos::Array<double, 2> spacing() const;

    /**
     * Returns the number of cells in the grid (including ghost cells).
     *
     * @return A Kokkos::Array containing the number of cells in the x and v
     *directions.
     **/
    // KOKKOS_FUNCTION
    // Kokkos::Array<int, 2> extents() const;

    /**
     * Calculate the center of the cell given its coordinate indexes of cells
     *(including ghost cells).
     *
     * @param coord_idx Coordinate indexes of the cell, starts from 0 to ncells
     *- 1.
     * @return A Kokkos::Array containing the center coordinates in the x and v
     *directions.
     **/
    // KOKKOS_FUNCTION
    // Kokkos::Array<double, 2> center(const Kokkos::Array<int, 2> coord_idx) const;

    KOKKOS_FUNCTION
    Kokkos::Array<int, 2> extents() const { return {ncells[0], ncells[1]}; }

    KOKKOS_FUNCTION
    Kokkos::Array<double, 2> spacing() const {
        return {
            size[0] / (ncells[0] - 2 * ngc), // spacing in the x-direction
            size[1] / (ncells[1] - 2 * ngc)  // spacing in the v-direction
        };
    }

    KOKKOS_FUNCTION
    Kokkos::Array<double, 2> center(const Kokkos::Array<int, 2> coord_idx) const {
        auto [dx, dv] = spacing();
        return {
            origin[0] + (coord_idx[0] - ngc) * dx + dx / 2.0, // center in the x-direction
            origin[1] + (coord_idx[1] - ngc) * dv + dv / 2.0  // center in the v-direction
        };
    };
};
