#pragma once
#include "world.hpp"
#include <highfive/highfive.hpp>
#include <set>
#include <string>
#include <vector>

class Writer {
  private:
    World& world;
    std::set<std::string> diag_fields;
    std::string folder;
    std::string prefix;

    // host arrays to temporary store the data for writing
    std::vector<double> fe;
    std::vector<double> fi;
    std::vector<double> ne;
    std::vector<double> ni;
    std::vector<double> rho;
    std::vector<double> phi;
    std::vector<double> Ex;
    std::vector<double> Ey;

  public:
    Writer(World& world, const std::string& folder, const std::string& prefix, const std::set<std::string> diag_fields);

    void write(double time);
};
