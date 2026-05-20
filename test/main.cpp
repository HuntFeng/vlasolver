#include "../include/full/world.hpp"
#include <Kokkos_Core.hpp>
#include <bit>
#include <gtest/gtest.h>

enum Flags : std::size_t {
    A = 1 << 0,
    B = 1 << 1,
    C = 1 << 2,
};

struct ImmersedWorld : World<ImmersedWorld> {
    double dy = 0.0;
    ImmersedWorld(Grid& grid)
        : World<ImmersedWorld>(grid) {
        // load_initial_potential();
        dy = grid.spacing(0, 0)[1];
    }

    KOKKOS_INLINE_FUNCTION
    double surface(double x, double y) const { return 2.0; }
};

template <typename World>
class TestClass {
  private:
    World& world;
    Kokkos::View<double**>& phi;
    double dx;
    double dy;

  public:
    TestClass(World& world)
        : world(world),
          phi(world.phi) {
        dx = world.grid.spacing(0, 0)[0];
        dy = world.grid.spacing(0, 0)[0];
    }

    KOKKOS_INLINE_FUNCTION
    double compute_theta(int i) const { return world.surface(0, 0) * phi(i, i) / dy; }

    void run() {
        // int nx = world.grid.ncells[0];
        // int ny = world.grid.ncells[1];
        // Kokkos::parallel_for(
        //     Kokkos::MDRangePolicy({0, 0}, {nx, ny}), KOKKOS_CLASS_LAMBDA(int i, int j) { phi(i, j) = i + j; });

        //     Kokkos::parallel_for(
        //         Kokkos::RangePolicy<>(0, 5), KOKKOS_CLASS_LAMBDA(int i) {
        //             double theta = compute_theta(i);
        //             Kokkos::printf("Computed theta in TestClass: %f\n", theta);
        //         });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::size_t mask = 0;
    mask |= Flags::B;
    mask |= Flags::C;
    // int count = __builtin_popcount(mask);
    int count = std::popcount(mask);
    Kokkos::printf("Count of set bits: %d\n", count);

    switch (mask) {
    case 0:
        Kokkos::printf("No flags set.\n");
        break;
    case A:
        Kokkos::printf("Only flag A is set.\n");
        break;
    case B:
        Kokkos::printf("Only flag B is set.\n");
        break;
    case C:
        Kokkos::printf("Only flag C is set.\n");
        break;
    case A | B:
        Kokkos::printf("Flags A and B are set.\n");
        break;
    case A | C:
        Kokkos::printf("Flags A and C are set.\n");
        break;
    case B | C:
        Kokkos::printf("Flags B and C are set.\n");
        break;
    }

    Grid grid({128, 128, 10, 10}, 1);
    grid.set_grid({0, 0, 0, 0}, {1, 1, 1, 1}, 0); // electrons
    grid.set_grid({0, 0, 0, 0}, {1, 1, 1, 1}, 1); // ions

    ImmersedWorld world(grid);
    Kokkos::printf("surface: %f\n", world.surface(0, 0));

    TestClass<ImmersedWorld> test_class(world);
    test_class.run();

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
