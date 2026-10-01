// Volume-element crystal generator (port of aggregation/generator.py).
#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "common.hpp"
#include "crystal.hpp"
#include "rng.hpp"
#include "rotator.hpp"

namespace snowagg {

// Samples a crystal on the same Cartesian lattice as the Python
// MonodisperseGenerator (np.mgrid with bounds rounded up to a multiple of
// grid_res) and returns the points in the same (C) order.
class MonodisperseGenerator {
public:
    std::shared_ptr<const Crystal> crystal;
    std::shared_ptr<const Rotator> rot;
    double grid_res;
    double max_r;
    int64_t n;  // lattice points per dimension
    Points lattice;  // unrotated points inside the crystal

    MonodisperseGenerator(std::shared_ptr<const Crystal> crystal_,
                          std::shared_ptr<const Rotator> rot_, double grid_res_)
        : crystal(std::move(crystal_)), rot(std::move(rot_)), grid_res(grid_res_) {
        max_r = crystal->max_radius();
        // round up to nearest multiple of grid_res
        max_r += grid_res - pymod(max_r, grid_res);
        const double start = -max_r;
        const double stop = max_r + grid_res * 0.001;
        n = static_cast<int64_t>(std::ceil((stop - start) / (grid_res * 1.0)));
        build_lattice(start);
    }

    double coord(int64_t i) const { return static_cast<double>(i) * grid_res + (-max_r); }

    // Volume-element realization, rotated with the generator's rotator.
    Points generate(Rng& rng) const {
        Points X = lattice;
        if (rot) rot->rotate(X, rng);
        return X;
    }

private:
    void build_lattice(double start) {
        std::vector<double> c(n);
        for (int64_t i = 0; i < n; ++i) c[i] = static_cast<double>(i) * grid_res + start;
        std::vector<Points> slabs(n);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t k = 0; k < n; ++k)
                    if (crystal->is_inside(c[i], c[j], c[k])) slabs[i].push_back({c[i], c[j], c[k]});
        }
        size_t total = 0;
        for (const auto& s : slabs) total += s.size();
        lattice.reserve(total);
        for (const auto& s : slabs) lattice.insert(lattice.end(), s.begin(), s.end());
    }
};

}  // namespace snowagg
