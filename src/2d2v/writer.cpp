#include "writer.hpp"
#include <Kokkos_Core.hpp>

Writer::Writer(World& world, const std::string& filename, const std::set<std::string> diag_fields)
    : world(world),
      file(filename, HighFive::File::Overwrite),
      diag_fields(diag_fields) {
    using namespace HighFive;
    auto [nx, ny, nvx, nvy] = world.grid.ncells;

    // Create datasets
    std::vector<size_t> dims_dist  = {world.total_steps / world.diag_steps, (size_t)(nx * ny * nvx * nvy)};
    std::vector<size_t> dims_field = {world.total_steps / world.diag_steps, (size_t)(nx * ny)};
    // enable chunking for efficient writing distribution per step
    DataSetCreateProps props_dist;
    props_dist.add(Chunking(std::vector<hsize_t>{1, (hsize_t)(nx * ny * nvx * nvy)}));
    // enable chunking for efficient writing fields per step
    DataSetCreateProps props_field;
    props_field.add(Chunking(std::vector<hsize_t>{1, (hsize_t)(nx * ny)}));
    if (diag_fields.contains("fe"))
        dataset_fe.emplace(file.createDataSet<double>("VTKHDF/CellData/fe", DataSpace(dims_dist), props_dist));
    if (diag_fields.contains("fi"))
        dataset_fi.emplace(file.createDataSet<double>("VTKHDF/CellData/fi", DataSpace(dims_dist), props_dist));
    if (diag_fields.contains("ne"))
        dataset_ne.emplace(file.createDataSet<double>("VTKHDF/CellData/ne", DataSpace(dims_field), props_field));
    if (diag_fields.contains("ni"))
        dataset_ni.emplace(file.createDataSet<double>("VTKHDF/CellData/ni", DataSpace(dims_field), props_field));
    if (diag_fields.contains("rho"))
        dataset_rho.emplace(file.createDataSet<double>("VTKHDF/CellData/rho", DataSpace(dims_field), props_field));
    if (diag_fields.contains("phi"))
        dataset_phi.emplace(file.createDataSet<double>("VTKHDF/CellData/phi", DataSpace(dims_field), props_field));
    if (diag_fields.contains("Ex"))
        dataset_Ex.emplace(file.createDataSet<double>("VTKHDF/CellData/Ex", DataSpace(dims_field), props_field));
    if (diag_fields.contains("Ey"))
        dataset_Ey.emplace(file.createDataSet<double>("VTKHDF/CellData/Ey", DataSpace(dims_field), props_field));
    count_dist   = {1, (size_t)(nx * ny * nvx * nvy)};
    count_field  = {1, (size_t)(nx * ny)};
    offset_dist  = {0, 0};
    offset_field = {0, 0};

    // host arrays for temporary holding the data
    fe  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny * nvx * nvy));
    fi  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny * nvx * nvy));
    ne  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
    ni  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
    rho = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
    phi = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
    Ex  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
    Ey  = std::vector<std::vector<double>>(1, std::vector<double>(nx * ny));
}

void Writer::write(double time) {
    Kokkos::printf("(Writer) Writing step %zu at time %f\n", offset_dist[0], time);

    auto [nx, ny, nvx, nvy] = world.grid.ncells;

    if (diag_fields.contains("fe")) {
        auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int iv = 0; iv < nvx; ++iv) {
                    for (int jv = 0; jv < nvy; ++jv) {
                        fe[0][i * ny * nvx * nvy + j * nvx * nvy + iv * nvy + jv] = f_host(i, j, iv, jv, 0);
                    }
                }
            }
        }
        dataset_fe->select(offset_dist, count_dist).write(fe);
    }

    if (diag_fields.contains("fi")) {
        auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.f);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int iv = 0; iv < nvx; ++iv) {
                    for (int jv = 0; jv < nvy; ++jv) {
                        fi[0][i * ny * nvx * nvy + j * nvx * nvy + iv * nvy + jv] = f_host(i, j, iv, jv, 1);
                    }
                }
            }
        }
        dataset_fi->select(offset_dist, count_dist).write(fi);
    }

    if (diag_fields.contains("ne")) {
        auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                ne[0][i * ny + j] = n_host(i, j, 0);
            }
        }
        dataset_ne->select(offset_field, count_field).write(ne);
    }

    if (diag_fields.contains("ni")) {
        auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.n);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                ni[0][i * ny + j] = n_host(i, j, 1);
            }
        }
        dataset_ni->select(offset_field, count_field).write(ni);
    }

    if (diag_fields.contains("rho")) {
        auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.rho);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                rho[0][i * ny + j] = rho_host(i, j);
            }
        }
        dataset_rho->select(offset_field, count_field).write(rho);
    }

    if (diag_fields.contains("phi")) {
        auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.phi);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                phi[0][i * ny + j] = phi_host(i, j);
            }
        }
        dataset_phi->select(offset_field, count_field).write(phi);
    }

    if (diag_fields.contains("Ex")) {
        auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Ex[0][i * ny + j] = E_host(i, j, 0);
            }
        }
        dataset_Ex->select(offset_field, count_field).write(Ex);
    }

    if (diag_fields.contains("Ey")) {
        auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), world.E);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Ey[0][i * ny + j] = E_host(i, j, 1);
            }
        }
        dataset_Ey->select(offset_field, count_field).write(Ey);
    }

    offset_dist[0]++;
    offset_field[0]++;
}
