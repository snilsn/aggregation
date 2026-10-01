// Deposition growth / sublimation by random walkers (port of
// aggregation/deposition.py).
#pragma once

#include <cmath>
#include <functional>

#include "aggregate.hpp"
#include "index.hpp"
#include "mcs.hpp"
#include "rng.hpp"

namespace snowagg {

// Adds (ice_vol > 0) or removes (ice_vol < 0) about |ice_vol| of ice.
// acos_fn: optional replacement for std::acos (the Python reference uses
// np.arccos, whose SIMD implementation can differ from libm in the last bit;
// tests pass np.arccos here to compare bit for bit).
inline void grow_ice(Aggregate& agg, Rng& rng, double ice_vol, double outer_rad_norm = 2.,
                     double move_norm = 1., const std::function<double(double)>& acos_fn = nullptr) {
    const double g = agg.grid_res;
    const double g2 = pypow(g, 2);
    Index3D elem_index(g);
    elem_index.insert(agg.X);

    const double p_vol = pypow(g, 3);
    const int64_t n_needed = static_cast<int64_t>(npround(std::fabs(ice_vol) / p_vol));
    if (n_needed == 0) return;
    const double sig = move_norm * g;

    auto covering_sphere = [&]() -> std::pair<Vec3, double> {
        if (agg.X.size() > 1) return minimum_covering_sphere(agg.X);
        return {agg.X.at(0), g / 2.};
    };
    auto [c, rad] = covering_sphere();

    auto nearest_sqr = [&](const Vec3& p, Vec3* nearest) {
        double best = INFINITY;
        bool any = false;
        elem_index.for_each_near(p, g, [&](const Vec3& q) {
            const double d = sq(q[0] - p[0]) + sq(q[1] - p[1]) + sq(q[2] - p[2]);
            if (!any || d < best) {
                best = d;
                if (nearest) *nearest = q;
            }
            any = true;
        });
        return any ? best : NAN;
    };

    auto attach = [&](Vec3 p0, Vec3 p1, int n_iters = 8) {
        Vec3 p{};
        for (int i = 0; i < n_iters; ++i) {
            for (int d = 0; d < 3; ++d) p[d] = 0.5 * (p0[d] + p1[d]);
            double dist_sqr_nearest = nearest_sqr(p, nullptr);
            if (std::isnan(dist_sqr_nearest))
                dist_sqr_nearest = 999.;
            else
                dist_sqr_nearest /= g2;
            if (dist_sqr_nearest - 1 < 0.01) break;
            if (dist_sqr_nearest > 1)
                p0 = p;
            else
                p1 = p;
        }
        agg.add_elements(Points{p}, DEPOSITION_IDENT, false);
        return p;
    };

    int64_t n_added = 0;
    while (n_added < n_needed) {
        // random starting point on a sphere outside the particle
        const double phi = 2 * PI * rng.rand();
        const double u_theta = 1 - 2 * rng.rand();
        const double theta = acos_fn ? acos_fn(u_theta) : std::acos(u_theta);
        const double r = rad * outer_rad_norm;
        Vec3 p = {c[0] + r * std::sin(theta) * std::cos(phi), c[1] + r * std::sin(theta) * std::sin(phi),
                  c[2] + r * std::cos(theta)};
        while (true) {
            const Vec3 p_old = p;
            const double n0 = rng.gauss(), n1 = rng.gauss(), n2 = rng.gauss();
            p[0] += sig * n0;
            p[1] += sig * n1;
            p[2] += sig * n2;
            const double r_sqr = sq(p[0] - c[0]) + sq(p[1] - c[1]) + sq(p[2] - c[2]);
            if (r_sqr > pypow(rad * outer_rad_norm, 2)) break;
            if (r_sqr < pypow(rad, 2)) {
                Vec3 nearest{};
                const double dmin = nearest_sqr(p, &nearest);
                if (!std::isnan(dmin) && dmin < g2) {
                    if (ice_vol < 0) {
                        p = nearest;
                        agg.remove_elements(Points{p}, 0.001, false);
                        elem_index.remove(p);
                    } else {
                        p = attach(p_old, p);
                    }
                    if (sq(p[0] - c[0]) + sq(p[1] - c[1]) + sq(p[2] - c[2]) > pypow(rad, 2)) {
                        auto cs = covering_sphere();
                        c = cs.first;
                        rad = cs.second;
                    }
                    ++n_added;
                    break;
                }
            }
        }
    }
    agg.update_coordinates();
}

}  // namespace snowagg
