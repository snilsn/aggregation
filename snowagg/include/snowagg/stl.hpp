// ASCII STL export of a gridded snowflake (port of aggregation/stl.py).
// Produces byte-identical files to the Python version.
#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "aggregate.hpp"
#include "common.hpp"

namespace snowagg {

// X: integer grid coordinates, e.g. from Aggregate::grid().
inline void snowflake_grid_to_stl(const std::vector<IVec3>& X_in, const std::string& stl_fn) {
    if (X_in.empty()) throw std::invalid_argument("snowflake_grid_to_stl: no points");
    // shift coordinates to non-negative (by the global minimum, like X - X.min())
    int64_t gmin = X_in[0][0];
    for (const auto& p : X_in) gmin = std::min({gmin, p[0], p[1], p[2]});
    std::vector<IVec3> X(X_in.size());
    for (size_t i = 0; i < X.size(); ++i) X[i] = {X_in[i][0] - gmin, X_in[i][1] - gmin, X_in[i][2] - gmin};
    std::unordered_set<IVec3, IVec3Hash> points(X.begin(), X.end());
    std::sort(X.begin(), X.end());
    X.erase(std::unique(X.begin(), X.end()), X.end());

    // squares (4 corners) and normals in the loop order of the Python code
    std::vector<std::array<IVec3, 4>> squares;
    std::vector<IVec3> normals;
    for (const auto& p : X) {
        for (int axis = 0; axis < 3; ++axis)
            for (int direction : {-1, 1}) {
                IVec3 nb = p;
                nb[axis] += direction;
                if (points.count(nb)) continue;
                const int64_t inc = (direction == 1) ? 1 : 0;
                std::array<IVec3, 4> sq4;
                int k = 0;
                for (int64_t dx : (axis == 0 ? std::vector<int64_t>{inc} : std::vector<int64_t>{0, 1}))
                    for (int64_t dy : (axis == 1 ? std::vector<int64_t>{inc} : std::vector<int64_t>{0, 1}))
                        for (int64_t dz : (axis == 2 ? std::vector<int64_t>{inc} : std::vector<int64_t>{0, 1}))
                            sq4[k++] = {p[0] + dx, p[1] + dy, p[2] + dz};
                squares.push_back(sq4);
                IVec3 n = {0, 0, 0};
                n[axis] = direction;
                normals.push_back(n);
            }
    }

    FILE* f = std::fopen(stl_fn.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + stl_fn);
    std::fputs("solid snowflake\n", f);
    auto write_triangle = [&](IVec3 t0, IVec3 t1, IVec3 t2, const IVec3& n) {
        const IVec3 v = {t0[0] - t1[0], t0[1] - t1[1], t0[2] - t1[2]};
        const IVec3 w = {t2[0] - t1[0], t2[1] - t1[1], t2[2] - t1[2]};
        const IVec3 cr = {v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0]};
        if (cr[0] * n[0] + cr[1] * n[1] + cr[2] * n[2] > 0) std::swap(t0, t2);
        std::fprintf(f, "facet normal %lld %lld %lld\n    outer loop\n", (long long)n[0], (long long)n[1],
                     (long long)n[2]);
        for (const auto& t : {t0, t1, t2})
            std::fprintf(f, "        vertex %lld %lld %lld\n", (long long)t[0], (long long)t[1], (long long)t[2]);
        std::fputs("    endloop\nendfacet\n", f);
    };
    for (size_t s = 0; s < squares.size(); ++s)
        write_triangle(squares[s][0], squares[s][1], squares[s][2], normals[s]);
    for (size_t s = 0; s < squares.size(); ++s)
        write_triangle(squares[s][1], squares[s][2], squares[s][3], normals[s]);
    std::fputs("endsolid snowflake\n", f);
    std::fclose(f);
}

}  // namespace snowagg
