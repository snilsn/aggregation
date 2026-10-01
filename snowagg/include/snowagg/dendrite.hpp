// Reiter (2005) hexagonal-grid snow crystal growth (port of
// aggregation/dendrite.py).
//
// The Python version uses scipy.ndimage.convolve, which flips the kernels;
// the neighbourhoods and the summation order below reproduce that exactly
// (weights in the order of the flipped kernel, zero weights skipped, zero
// padding at the borders).
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "common.hpp"

namespace snowagg {

struct DendriteResult {
    int64_t size = 0;           // grid is size x size
    std::vector<double> grid;   // row-major
    int64_t iterations = 0;
    double margin = 0.0;
};

namespace dendrite_detail {
struct Tap {
    int di, dj;
    double w;
};
// Taps of ndimage.convolve(input, kernel): correlation with the flipped
// kernel, in C order of the flipped kernel, skipping |w| <= DBL_EPSILON.
inline std::vector<Tap> convolve_taps(const double k[3][3]) {
    std::vector<Tap> taps;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            const double w = k[2 - a][2 - b];
            if (std::fabs(w) > DBL_EPSILON) taps.push_back({a - 1, b - 1, w});
        }
    return taps;
}
}  // namespace dendrite_detail

inline DendriteResult generate_dendrite(double alpha, double beta, double gamma, int64_t grid_size = 1000,
                                        int64_t num_iter = 10000) {
    using dendrite_detail::Tap;
    const int64_t n = grid_size;
    DendriteResult res;
    res.size = n;

    // x = tile(linspace(-n/2, n/2, n), (n,1)); y = x.T
    std::vector<double> lin(n);
    {
        const double start = -n / 2.0, stop = n / 2.0;
        const double step = (stop - start) / static_cast<double>(n - 1);
        for (int64_t k = 0; k < n; ++k) lin[k] = static_cast<double>(k) * step + start;
        if (n > 1) lin[n - 1] = stop;
    }
    const double rlim = pypow(n / 2.0, 2);
    std::vector<uint8_t> boundary(n * n);
    std::vector<double> dist(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            const double r2 = sq(lin[j]) + sq(lin[i]);
            boundary[i * n + j] = r2 >= rlim;
            dist[i * n + j] = std::sqrt(r2);
        }

    std::vector<double> grid(n * n, beta);
    grid[(n / 2) * n + (n / 2)] = 1.0;

    const double nb_even[3][3] = {{1, 1, 0}, {1, 1, 1}, {1, 1, 0}};
    const double nb_odd[3][3] = {{0, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    const double a12 = alpha / 12.0, c = 1.0 - alpha / 2.0;
    const double avg_even[3][3] = {{a12, a12, 0.0}, {a12, c, a12}, {a12, a12, 0.0}};
    const double avg_odd[3][3] = {{0.0, a12, a12}, {a12, c, a12}, {0.0, a12, a12}};
    const auto t_nb_even = dendrite_detail::convolve_taps(nb_even);
    const auto t_nb_odd = dendrite_detail::convolve_taps(nb_odd);
    const auto t_avg_even = dendrite_detail::convolve_taps(avg_even);
    const auto t_avg_odd = dendrite_detail::convolve_taps(avg_odd);

    std::vector<uint8_t> ice(n * n), receptive(n * n);
    std::vector<double> nonrecp(n * n), nonrecp_conv(n * n);

    int64_t it = 0;
    double margin = 999.0;
    while ((it < num_iter) && (margin > 0.5)) {
        for (int64_t k = 0; k < n * n; ++k) ice[k] = grid[k] >= 1.0;

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const auto& taps = (i % 2 == 0) ? t_nb_even : t_nb_odd;
            for (int64_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (const Tap& t : taps) {
                    const int64_t ii = i + t.di, jj = j + t.dj;
                    if (ii < 0 || ii >= n || jj < 0 || jj >= n) continue;
                    s += t.w * ice[ii * n + jj];
                }
                receptive[i * n + j] = s != 0.0;
            }
        }
        for (int64_t k = 0; k < n * n; ++k) nonrecp[k] = receptive[k] ? 0.0 : grid[k];

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const auto& taps = (i % 2 == 0) ? t_avg_even : t_avg_odd;
            for (int64_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (const Tap& t : taps) {
                    const int64_t ii = i + t.di, jj = j + t.dj;
                    if (ii < 0 || ii >= n || jj < 0 || jj >= n) continue;
                    s += t.w * nonrecp[ii * n + jj];
                }
                nonrecp_conv[i * n + j] = s;
            }
        }

        double max_dist = -INFINITY;
        for (int64_t k = 0; k < n * n; ++k) {
            if (!receptive[k]) grid[k] = 0.0;
            if (grid[k] != 0.0) grid[k] += gamma;
            grid[k] += nonrecp_conv[k];
            if (boundary[k]) grid[k] = beta;
            if (ice[k] && dist[k] > max_dist) max_dist = dist[k];
        }
        margin = std::fabs(n / 2.0 - max_dist);
        ++it;
    }
    res.grid = std::move(grid);
    res.iterations = it;
    res.margin = margin;
    return res;
}

}  // namespace snowagg
