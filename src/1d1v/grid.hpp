#pragma once
#include <Kokkos_Core.hpp>

/**
 * Grid struct contains the parameters of the grid used in the Vlasov solver.
 */
struct Grid {
    int ngc;                          // number of ghost cells on each side
    Kokkos::Array<double, 2> origin;  // origin of the grid
    Kokkos::Array<double, 2> size;    // size of the grid
    Kokkos::Array<int, 2> ncells;     // number of all cells in the grid
    Kokkos::Array<double, 2> spacing; // spacing in the x and v directions

    /**
     * Constructor to initialize Grid with custom parameters.
     *
     * @param origin Origin coordinates of the grid
     * @param size Size of the grid
     * @param ncells_interior Number of cells in the grid (excluding ghost cells)
     * @param (optional) ngc_in Number of ghost cells on each side, defaults to
     * 3
     */
    Grid(const Kokkos::Array<double, 2> origin,
         const Kokkos::Array<double, 2> size,
         const Kokkos::Array<int, 2> ncells_interior,
         int ngc = 3);

    /**
     * Calculate the center of the cell given its coordinate indexes of cells
     *(including ghost cells).
     *
     * @param coord_idx Coordinate indexes of the cell, starts from 0 to ncells
     *- 1.
     * @return A Kokkos::Array containing the center coordinates in the x and v
     *directions.
     **/
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> center(const Kokkos::Array<int, 2> coord_idx) const {
        auto [dx, dv] = spacing;
        return {
            origin[0] + (coord_idx[0] - ngc) * dx + dx / 2.0, // center in the x-direction
            origin[1] + (coord_idx[1] - ngc) * dv + dv / 2.0  // center in the v-direction
        };
    };
};
