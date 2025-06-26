#pragma once
#include "world.hpp"
#include <highfive/highfive.hpp>
#include <optional>
#include <set>
#include <string>
#include <vector>

class Writer {
  private:
    HighFive::File file;
    World& world;
    std::set<std::string> diag_fields;

    std::optional<HighFive::DataSet> dataset_fe;
    std::optional<HighFive::DataSet> dataset_fi;
    std::optional<HighFive::DataSet> dataset_ne;
    std::optional<HighFive::DataSet> dataset_ni;
    std::optional<HighFive::DataSet> dataset_rho;
    std::optional<HighFive::DataSet> dataset_phi;
    std::optional<HighFive::DataSet> dataset_Ex;
    std::optional<HighFive::DataSet> dataset_Ey;

    std::vector<size_t> offset_dist;
    std::vector<size_t> count_dist;
    std::vector<size_t> offset_field;
    std::vector<size_t> count_field;

    // host arrays to temporary store the data for writing
    std::vector<std::vector<double>> fe;
    std::vector<std::vector<double>> fi;
    std::vector<std::vector<double>> ne;
    std::vector<std::vector<double>> ni;
    std::vector<std::vector<double>> rho;
    std::vector<std::vector<double>> phi;
    std::vector<std::vector<double>> Ex;
    std::vector<std::vector<double>> Ey;

  public:
    Writer(World& world, const std::string& filename, const std::set<std::string> diag_fields);

    void write(double time);
};
