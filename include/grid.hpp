#pragma once
#include <Kokkos_Core.hpp>

const int DIM = 4; // Number of dimensions in the Vlasov solver

/**
 * Grid struct contains the parameters of the grid used in the Vlasov solver.
 * The grid is defined in a 4D phase space (x, y, vx, vy) for up to two species.
 * Both species share the same spatial grid but may have different velocity grids.
 */
struct Grid {
    int ngc;                                              // number of ghost cells on each side
    Kokkos::Array<int, DIM> ncells;                       // number of all cells in each dimension
    Kokkos::Array<int, DIM> ncells_interior;              // number of interior cells in each dimension

    // Per-species grid parameters (sp=0 for ion, sp=1 for electron)
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> origin;  // origin per species
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> size;    // size per species
    Kokkos::Array<Kokkos::Array<double, DIM>, 2> sp;      // spacing per species (dx,dy,dvx,dvy)

    // Convenience ranges for species 0 (spatial + velocity min/max)
    Kokkos::Array<double, DIM> velocity_ranges;
    Kokkos::Array<double, DIM> space_ranges;

    /**
     * Constructor.
     * @param ncells_interior Number of interior cells (excluding ghost cells)
     * @param ngc Number of ghost cells on each side
     */
    Grid(const Kokkos::Array<int, DIM> ncells_interior, int ngc)
        : ncells_interior(ncells_interior),
          ngc(ngc) {
        for (int d = 0; d < DIM; ++d)
            ncells[d] = ncells_interior[d] + 2 * ngc;
    }

    /**
     * Set the grid parameters for a specific species.
     * @param origin_in Origin coordinates
     * @param size_in Size of the grid
     * @param sp_in Species index (0 or 1)
     */
    void set_grid(const Kokkos::Array<double, DIM> origin_in,
                  const Kokkos::Array<double, DIM> size_in,
                  const int sp_in) {
        origin[sp_in] = origin_in;
        size[sp_in]   = size_in;
        for (int d = 0; d < DIM; ++d)
            sp[sp_in][d] = size[sp_in][d] / ncells_interior[d];

        if (sp_in == 0) {
            velocity_ranges[0] = origin[0][2];
            velocity_ranges[1] = origin[0][3];
            velocity_ranges[2] = origin[0][2] + size[0][2];
            velocity_ranges[3] = origin[0][3] + size[0][3];

            space_ranges[0]    = origin[0][0];
            space_ranges[1]    = origin[0][1];
            space_ranges[2]    = origin[0][0] + size[0][0];
            space_ranges[3]    = origin[0][1] + size[0][1];
        }
    }

    // ---- Overloaded spacing() methods ----

    /// Phase-space spacing for species sp: returns {dx, dy, dvx, dvy}
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, DIM> spacing(int sp_in) const { return sp[sp_in]; }

    /// Spatial spacing (species 0): returns {dx, dy}
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> spacing(int, int) const { return {sp[0][0], sp[0][1]}; }

    /// Phase-space spacing (species 0): returns {dx, dy, dvx, dvy}
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, DIM> spacing(int, int, int, int) const { return sp[0]; }

    // ---- Overloaded center() methods ----

    /// Spatial center of cell (i, j) (species 0): returns {x, y}
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, 2> center(int i, int j) const {
        auto dx = sp[0][0], dy = sp[0][1];
        return {
            origin[0][0] + (i - ngc) * dx + dx / 2.0,
            origin[0][1] + (j - ngc) * dy + dy / 2.0,
        };
    }

    /// Phase-space center of cell (i, j, iv, jv) for species sp: returns {x, y, vx, vy}
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, DIM> center(int i, int j, int iv, int jv, int sp_in = 0) const {
        auto p = sp[sp_in];
        return {
            origin[sp_in][0] + (i - ngc) * p[0] + p[0] / 2.0,
            origin[sp_in][1] + (j - ngc) * p[1] + p[1] / 2.0,
            origin[sp_in][2] + (iv - ngc) * p[2] + p[2] / 2.0,
            origin[sp_in][3] + (jv - ngc) * p[3] + p[3] / 2.0,
        };
    }

    /**
     * Logical coordinate of the cell given its phase-space position.
     * @param pos Position {x, y, vx, vy}
     * @return Logical cell indices {i, j, iv, jv}
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<int, DIM> coord(const Kokkos::Array<double, DIM> pos) const {
        auto p = sp[0];
        return {
            static_cast<int>((pos[0] - origin[0][0] - p[0] / 2.0) / p[0]) + ngc,
            static_cast<int>((pos[1] - origin[0][1] - p[1] / 2.0) / p[1]) + ngc,
            static_cast<int>((pos[2] - origin[0][2] - p[2] / 2.0) / p[2]) + ngc,
            static_cast<int>((pos[3] - origin[0][3] - p[3] / 2.0) / p[3]) + ngc,
        };
    }
};
