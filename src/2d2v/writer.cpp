#include "writer.hpp"
#include <Kokkos_Core.hpp>
#include <filesystem>
#include <sstream>
#include <string>

Writer::Writer(World& world,
               const std::string& folder,
               const std::string& prefix,
               const std::set<std::string> diag_fields)
    : world(world),
      diag_fields(diag_fields),
      folder(folder),
      prefix(prefix) {

    // Clear the folder if it exists and create a new one
    if (std::filesystem::exists(folder))
        std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);

    using namespace HighFive;
    auto [nx, ny, nvx, nvy] = world.grid.ncells;

    // host arrays for temporary holding the data
    fe  = std::vector<double>(nx * ny * nvx * nvy);
    fi  = std::vector<double>(nx * ny * nvx * nvy);
    ne  = std::vector<double>(nx * ny);
    ni  = std::vector<double>(nx * ny);
    rho = std::vector<double>(nx * ny);
    phi = std::vector<double>(nx * ny);
    Ex  = std::vector<double>(nx * ny);
    Ey  = std::vector<double>(nx * ny);
}

void Writer::write(double time) {
    Kokkos::printf("(Writer) Writing step %zu at time %f\n", world.current_step, time);

    std::ostringstream oss;
    size_t length = std::to_string(world.total_steps).length();
    oss << std::setw(length) << std::setfill('0') << world.current_step;
    HighFive::File file(folder + "/" + prefix + "_" + oss.str() + ".h5", HighFive::File::Overwrite);
    auto [nx, ny, nvx, nvy] = world.grid.ncells;

    if (diag_fields.contains("fe")) {
        auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int iv = 0; iv < nvx; ++iv) {
                    for (int jv = 0; jv < nvy; ++jv) {
                        fe[i * ny * nvx * nvy + j * nvx * nvy + iv * nvy + jv] = f_host(i, j, iv, jv, 0);
                    }
                }
            }
        }
        file.createDataSet("VTKHDF/CellData/fe", fe);
    }

    if (diag_fields.contains("fi")) {
        auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int iv = 0; iv < nvx; ++iv) {
                    for (int jv = 0; jv < nvy; ++jv) {
                        fi[i * ny * nvx * nvy + j * nvx * nvy + iv * nvy + jv] = f_host(i, j, iv, jv, 1);
                    }
                }
            }
        }
        file.createDataSet("VTKHDF/CellData/fi", fi);
    }

    if (diag_fields.contains("ne")) {
        auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                ne[i * ny + j] = n_host(i, j, 0);
            }
        }
        file.createDataSet("VTKHDF/CellData/ne", ne);
    }

    if (diag_fields.contains("ni")) {
        auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                ni[i * ny + j] = n_host(i, j, 1);
            }
        }
        file.createDataSet("VTKHDF/CellData/ni", ni);
    }

    if (diag_fields.contains("rho")) {
        auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                rho[i * ny + j] = rho_host(i, j);
            }
        }
        file.createDataSet("VTKHDF/CellData/rho", rho);
    }

    if (diag_fields.contains("phi")) {
        auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                phi[i * ny + j] = phi_host(i, j);
            }
        }
        file.createDataSet("VTKHDF/CellData/phi", phi);
    }

    if (diag_fields.contains("Ex")) {
        auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Ex[i * ny + j] = E_host(i, j, 0);
            }
        }
        file.createDataSet("VTKHDF/CellData/Ex", Ex);
    }

    if (diag_fields.contains("Ey")) {
        auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Ey[i * ny + j] = E_host(i, j, 1);
            }
        }
        file.createDataSet("VTKHDF/CellData/Ey", Ey);
    }
}
