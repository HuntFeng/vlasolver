#pragma once
#include <Kokkos_Core.hpp>

const int DIM = 4; // Number of dimensions in the Vlasov solver
/**
 * Grid struct contains the parameters of the grid used in the Vlasov solver.
 * The grid is defined in a 4D phase space (x, y, vx, vy) for two species (electrons and ions).
 * Both species share the same spatial grid but have different velocity grids.
 * Both species have the same number of (ghost) cells.
 */
struct Grid {
    int ngc;                                              // number of ghost cells on each side
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> origin;  // origin of the grid
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> size;    // size of the grid
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> spacing; // spacing in the x and v directions
    Kokkos::Array<int, DIM> ncells;                       // number of all cells in the grid
    Kokkos::Array<int, DIM> ncells_interior;              // number of interior cells in the grid

    /**
     * Constructor to initialize Grid with custom parameters.
     *
     * @param origin Origin coordinates of the grid
     * @param size Size of the grid
     * @param ncells_interior Number of cells in the grid (excluding ghost cells)
     * @param (optional) ngc_in Number of ghost cells on each side, defaults to
     * 3
     */
    Grid(const Kokkos::Array<int, DIM> ncells_interior, int ngc)
        : ncells_interior(ncells_interior),
          ngc(ngc) {
        for (int d = 0; d < DIM; ++d)
            ncells[d] = ncells_interior[d] + 2 * ngc;
    }

    /**
     * Set the grid parameters for a specific species.
     *
     * @param origin_in Origin coordinates of the grid
     * @param size_in Size of the grid
     * @param sp Species index (0 or 1)
     */
    void set_grid(const Kokkos::Array<double, DIM> origin_in, const Kokkos::Array<double, DIM> size_in, const int sp) {
        origin[sp] = origin_in;
        size[sp]   = size_in;
        for (int d = 0; d < DIM; ++d) {
            spacing[sp][d] = size[sp][d] / ncells_interior[d];
        }
    }

    /**
     * Calculate the center of the cell given its coordinate indexes of cells (including ghost cells).
     *
     * @param coord_idx Coordinate indexes of the cell, starts from 0 to ncells - 1.
     * @return A Kokkos::Array containing the center coordinates in the x and v directions.
     * @param sp Species index (0 or 1)
     **/
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, DIM> center(const Kokkos::Array<int, DIM> coord_idx, const int sp) const {
        auto [dx, dy, dvx, dvy] = spacing[sp];
        return {
            origin[sp][0] + (coord_idx[0] - ngc) * dx + dx / 2.0,   // center in the x-direction
            origin[sp][1] + (coord_idx[1] - ngc) * dy + dy / 2.0,   // center in the y-direction
            origin[sp][2] + (coord_idx[2] - ngc) * dvx + dvx / 2.0, // center in the vx-direction
            origin[sp][3] + (coord_idx[3] - ngc) * dvy + dvy / 2.0  // center in the vy-direction
        };
    };
};
