#include <Kokkos_Core.hpp>

class Grid {
  public:
    Kokkos::Array<double, 2> origin;       // origin of the grid
    Kokkos::Array<double, 2> size;         // size of the grid
    Kokkos::Array<int, 2> ncells_interior; // number of cells in the grid (excluding ghost cells)
    int ngc;                               // number of ghost cells on each side

    Grid(const Kokkos::Array<double, 2>& origin,
         const Kokkos::Array<double, 2>& size,
         const Kokkos::Array<int, 2>& ncells_interior,
         int ngc = 3)
        : origin(origin),
          size(size),
          ncells_interior(ncells_interior),
          ngc(ngc) {
        Kokkos::printf("Kokkos::printf Grid initialized with origin = (%f, %f), size = (%f, %f), "
                       "ncells_interior = (%d, %d), ngc = %d\n",
                       origin[0], origin[1], size[0], size[1], ncells_interior[0], ncells_interior[1], ngc);
    }

    KOKKOS_FUNCTION
    Kokkos::Array<int, 2> extents() const { return {ncells_interior[0] + 2 * ngc, ncells_interior[1] + 2 * ngc}; }

    KOKKOS_FUNCTION
    Kokkos::Array<double, 2> spacing() const {
        return {size[0] / (ncells_interior[0] + 2 * ngc), size[1] / (ncells_interior[1] + 2 * ngc)};
    }

    KOKKOS_FUNCTION
    Kokkos::Array<double, 2> center(Kokkos::Array<int, 2> coord) const {
        auto [dx, dv] = spacing();
        return {origin[0] + (coord[0] + 0.5) * dx, origin[1] + (coord[1] + 0.5) * dv};
    }
};

class World {
  public:
    Grid& grid;                // reference to the grid object
    Kokkos::View<double***> f; // distribution function

    World(Grid& grid)
        : grid(grid) { // 2 species
        Kokkos::printf("Kokkos::printf World initialized with grid.extents = (%d, %d)\n", grid.extents()[0],
                       grid.extents()[1]);
        auto [nx, nv] = grid.extents();
        f             = Kokkos::View<double***>("f", nx, nv, 2); // 2 species
    }
};

class Vlasolver {
  public:
    World& world; // reference to the world object

    Vlasolver(World& world)
        : world(world) {
        auto [nx, nv] = world.grid.extents();
        Kokkos::printf("Kokkos::printf Vlasolver initialized with world.grid.extents = (%d, %d)\n", nx, nv);
    }

    void initialize_distribution() {
        auto grid     = world.grid; // reference to the grid object
        auto [nx, nv] = grid.extents();
        int ngc       = grid.ngc;
        Kokkos::printf("Kokkos::printf (nx,nv) = (%d,%d)\n", nx, nv);

        auto [x, v] = grid.center({0, 0});
        Kokkos::printf("(x_%d,v_%d) = (%f,%f)\n", 0, 0, x, v);

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0, 0}, {nx, nv}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                if (j < ngc || j >= nv - ngc)
                    return;
                // auto [x, v] = world.grid.center({i, j});
                auto [x, v] = grid.center({i, j});
                // FIXME: illegal memory access??????
                Kokkos::printf("(x_%d,v_%d) = (%f,%f)\n", i, j, x, v);
            });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    Grid grid({0, 0}, {1, 1}, {32, 64});
    // Kokkos::printf("Kokkos::printf grid.ncells = (%d, %d)\n", grid.ncells[0], grid.ncells[1]);
    // auto [nx, nv] = grid.extents();
    // Kokkos::printf("Kokkos::printf calling grid.extents = (%d, %d)\n", nx, nv);
    World world(grid);
    // Kokkos::printf("Kokkos::printf world.grid.ncells = (%d, %d)\n", world.grid.ncells[0], world.grid.ncells[1]);
    Vlasolver vlasolver(world);
    vlasolver.initialize_distribution();
    Kokkos::fence();
}
