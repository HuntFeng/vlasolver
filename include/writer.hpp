#pragma once
#include <Kokkos_Core.hpp>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <set>
#include <string>
#include <vector>

template <typename World>
class Writer {
  private:
    static constexpr int N = World::nspecies;

    World& world;
    std::set<std::string> diag_fields;
    std::string folder;
    std::string prefix;

    // host arrays to temporarily store the data for writing
    std::vector<double> f_buf; // per-species distribution buffer (reused)
    std::vector<double> n_buf; // per-species number density buffer (reused)
    std::vector<double> phi;
    std::vector<double> Ex;
    std::vector<double> Ey;

  public:
    Writer(World& world, const std::string& folder, const std::string& prefix, const std::set<std::string> diag_fields)
        : world(world),
          diag_fields(diag_fields),
          folder(folder),
          prefix(prefix) {

        // Clear the folder if it exists and create a new one
        if (std::filesystem::exists(folder))
            std::filesystem::remove_all(folder);
        std::filesystem::create_directories(folder);

        auto [nx, ny, nvx, nvy] = world.grid.ncells;

        // host arrays for temporary holding the data
        f_buf = std::vector<double>(nx * ny * nvx * nvy);
        n_buf = std::vector<double>(nx * ny);
        phi   = std::vector<double>(nx * ny);
        Ex    = std::vector<double>(nx * ny);
        Ey    = std::vector<double>(nx * ny);
    }

    void write(double time) {
        Kokkos::printf("(Writer) Writing step %zu at time %f\n", world.current_step, time);

        std::ostringstream oss;
        size_t length = std::to_string(world.total_steps).length();
        oss << std::setw(length) << std::setfill('0') << world.current_step;
        HighFive::File file(folder + "/" + prefix + "_" + oss.str() + ".h5", HighFive::File::Overwrite);
        auto [nx, ny, nvx, nvy] = world.grid.ncells;

        // Per-species distribution ("f"+name) and number density ("n"+name) fields.
        // e.g. species_names = {"e","i"} -> fe/fi, ne/ni; {"i"} -> fi, ni.
        bool need_f = false, need_n = false;
        for (int sp = 0; sp < N; ++sp) {
            need_f |= diag_fields.contains("f" + world.species_names[sp]);
            need_n |= diag_fields.contains("n" + world.species_names[sp]);
        }

        if (need_f) {
            auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
            for (int sp = 0; sp < N; ++sp) {
                std::string key = "f" + world.species_names[sp];
                if (!diag_fields.contains(key))
                    continue;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int iv = 0; iv < nvx; ++iv)
                            for (int jv = 0; jv < nvy; ++jv)
                                f_buf[i * ny * nvx * nvy + j * nvx * nvy + iv * nvy + jv] = f_host(i, j, iv, jv, sp);
                file.createDataSet("VTKHDF/CellData/" + key, f_buf);
            }
        }

        if (need_n) {
            auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
            for (int sp = 0; sp < N; ++sp) {
                std::string key = "n" + world.species_names[sp];
                if (!diag_fields.contains(key))
                    continue;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        n_buf[i * ny + j] = n_host(i, j, sp);
                file.createDataSet("VTKHDF/CellData/" + key, n_buf);
            }
        }

        if (diag_fields.contains("phi")) {
            auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    phi[i * ny + j] = phi_host(i, j);
            file.createDataSet("VTKHDF/CellData/phi", phi);
        }

        if (diag_fields.contains("Ex")) {
            auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    Ex[i * ny + j] = E_host(i, j, 0);
            file.createDataSet("VTKHDF/CellData/Ex", Ex);
        }

        if (diag_fields.contains("Ey")) {
            auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    Ey[i * ny + j] = E_host(i, j, 1);
            file.createDataSet("VTKHDF/CellData/Ey", Ey);
        }
    }
};
