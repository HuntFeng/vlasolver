#pragma once
#include <Kokkos_Core.hpp>

const int DIM = 4; // Number of dimensions in the Vlasov solver
/**
 * Grid struct contains the parameters of the grid used in the Vlasov solver.
 */
struct Grid {
    int ngc;                                    // number of ghost cells on each side
    Kokkos::Array<double, DIM> origin;          // origin of the grid
    Kokkos::Array<double, DIM> size;            // size of the grid
    Kokkos::Array<int, DIM> ncells;             // number of all cells in the grid
    Kokkos::Array<double, DIM> spacing;         // spacing in the x and v directions
    Kokkos::Array<double, DIM> velocity_ranges; // velocity ranges in the x and y directions
    Kokkos::Array<double, DIM> space_ranges;    // space ranges in the x and y directions

    /**
     * Constructor to initialize Grid with custom parameters.
     *
     * @param origin Origin coordinates of the grid
     * @param size Size of the grid
     * @param ncells_interior Number of cells in the grid (excluding ghost cells)
     * @param (optional) ngc_in Number of ghost cells on each side, defaults to
     * 3
     */
    Grid(const Kokkos::Array<double, DIM> origin,
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

    /**
     * Calculate the center of the cell given its coordinate indexes of cells (including ghost cells).
     *
     * @param coord_idx Coordinate indexes of the cell, starts from 0 to ncells - 1.
     * @return A Kokkos::Array containing the center coordinates in the x and v directions.
     **/
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, DIM> center(const Kokkos::Array<int, DIM> coord_idx) const {
        auto [dx, dy, dvx, dvy] = spacing;
        return {
            origin[0] + (coord_idx[0] - ngc) * dx + dx / 2.0,   // center in the x-direction
            origin[1] + (coord_idx[1] - ngc) * dy + dy / 2.0,   // center in the y-direction
            origin[2] + (coord_idx[2] - ngc) * dvx + dvx / 2.0, // center in the vx-direction
            origin[3] + (coord_idx[3] - ngc) * dvy + dvy / 2.0  // center in the vy-direction
        };
    };
};
