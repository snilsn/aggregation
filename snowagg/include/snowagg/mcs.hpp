// Minimum covering sphere (port of aggregation/mcs.py, Hopp & Reeve 1996).
#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "common.hpp"

namespace snowagg {

namespace mcs_detail {

inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// Solve A x = b by LU with partial pivoting (like LAPACK dgesv). Returns
// false for an exactly singular matrix (numpy raises LinAlgError).
template <int n>
bool solve(std::array<std::array<double, n>, n> A, std::array<double, n> b, std::array<double, n>& x) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(A[i][k]) > std::fabs(A[piv][k])) piv = i;
        if (A[piv][k] == 0.0) return false;
        std::swap(A[k], A[piv]);
        std::swap(b[k], b[piv]);
        const double inv = 1.0 / A[k][k];
        for (int i = k + 1; i < n; ++i) {
            const double l = A[i][k] * inv;
            for (int j = k + 1; j < n; ++j) A[i][j] -= l * A[k][j];
            b[i] -= l * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return true;
}

// Center of the minimum sphere through the candidate points; may drop
// candidates (mcs.mcsc).
inline Vec3 mcsc(const Points& pts, std::vector<int64_t>& cand) {
    const size_t count = cand.size();
    Vec3 center = {0.0, 0.0, 0.0};
    if (count == 1) {
        center = pts[cand[0]];
    } else if (count == 2) {
        const Vec3& q0 = pts[cand[0]];
        const Vec3& q1 = pts[cand[1]];
        for (int d = 0; d < 3; ++d) center[d] = (q0[d] + q1[d]) / 2.0;
    } else if (count == 3) {
        const Vec3 &q0 = pts[cand[0]], &q1 = pts[cand[1]], &q2 = pts[cand[2]];
        const Vec3 d0 = sub(q0, q2), d1 = sub(q1, q2);
        const double a00 = dot(d0, d0), a01 = dot(d0, d1), a10 = dot(d1, d0), a11 = dot(d1, d1);
        // solve(A.T, b)
        std::array<std::array<double, 2>, 2> At = {{{a00, a10}, {a01, a11}}};
        std::array<double, 2> b = {a00 / 2.0, a11 / 2.0}, l{};
        if (solve<2>(At, b, l)) {
            const double l2 = 1.0 - (l[0] + l[1]);
            const double ll[3] = {l[0], l[1], l2};
            int drop = -1;
            double minimum = 0;
            for (int i = 0; i < 3; ++i)
                if (ll[i] < minimum) {
                    drop = i;
                    minimum = ll[i];
                }
            if (drop >= 0) {
                cand.erase(cand.begin() + drop);
                center = mcsc(pts, cand);
            } else {
                for (int d = 0; d < 3; ++d) center[d] = l[0] * q0[d] + l[1] * q1[d] + l2 * q2[d];
            }
        } else {
            cand.pop_back();
            center = mcsc(pts, cand);
        }
    } else if (count == 4) {
        const Vec3 &q0 = pts[cand[0]], &q1 = pts[cand[1]], &q2 = pts[cand[2]], &q3 = pts[cand[3]];
        const Vec3 d0 = sub(q0, q3), d1 = sub(q1, q3), d2 = sub(q2, q3);
        const double a00 = dot(d0, d0), a01 = dot(d0, d1), a02 = dot(d0, d2);
        const double a10 = dot(d1, d0), a11 = dot(d1, d1), a12 = dot(d1, d2);
        const double a20 = dot(d2, d0), a21 = dot(d2, d1), a22 = dot(d2, d2);
        std::array<std::array<double, 3>, 3> At = {{{a00, a10, a20}, {a01, a11, a21}, {a02, a12, a22}}};
        std::array<double, 3> b = {a00 / 2.0, a11 / 2.0, a22 / 2.0}, l{};
        if (solve<3>(At, b, l)) {
            const double l3 = 1.0 - (l[0] + l[1] + l[2]);
            const double ll[4] = {l[0], l[1], l[2], l3};
            int drop = -1;
            double minimum = 0;
            for (int i = 0; i < 4; ++i)
                if (ll[i] < minimum) {
                    drop = i;
                    minimum = ll[i];
                }
            if (drop >= 0) {
                cand.erase(cand.begin() + drop);
                center = mcsc(pts, cand);
            } else {
                for (int d = 0; d < 3; ++d)
                    center[d] = l[0] * q0[d] + l[1] * q1[d] + l[2] * q2[d] + l3 * q3[d];
            }
        } else {
            cand.pop_back();
            center = mcsc(pts, cand);
        }
    }
    return center;
}

// One step of the covering-sphere reduction (mcs.find_next_candidate).
// Returns true when finished; updates center and candidates.
inline bool find_next_candidate(const Points& pts, Vec3& center, std::vector<int64_t>& cand,
                                std::vector<uint8_t>& is_cand, std::vector<double>& p) {
    const Vec3 t = mcsc(pts, cand);
    if (cand.size() == 4) {
        center = t;
        return true;
    }
    const size_t n = pts.size();
    std::fill(is_cand.begin(), is_cand.end(), 0);
    for (auto c : cand) is_cand[c] = 1;
    const Vec3 p0 = pts[cand[0]];
    const Vec3 tc = sub(t, center);
    for (size_t i = 0; i < n; ++i) {
        p[i] = 1.0;
        if (is_cand[i]) continue;
        const Vec3& q = pts[i];
        const double d = -((q[0] - p0[0]) * tc[0] + (q[1] - p0[1]) * tc[1] + (q[2] - p0[2]) * tc[2]);
        if (d > 0) {
            double s = 0;
            for (int k = 0; k < 3; ++k) {
                const double term = ((q[k] + p0[k]) / 2.0 - center[k]) * ((q[k] - p0[k]) / d);
                s = (k == 0) ? term : s + term;
            }
            p[i] = -s;
        }
    }
    double minimum = INFINITY;
    for (size_t i = 0; i < n; ++i)
        if (p[i] > 0 && p[i] < minimum) minimum = p[i];
    int64_t min_index = -1;
    for (size_t i = 0; i < n; ++i)
        if (p[i] == minimum) min_index = static_cast<int64_t>(i);
    if (minimum == 1) {
        center = t;
        return true;
    }
    Vec3 new_center;
    for (int k = 0; k < 3; ++k) new_center[k] = center[k] + p[min_index] * (t[k] - center[k]);
    cand.insert(cand.begin(), min_index);
    const bool same = new_center == center;
    center = new_center;
    return same;
}

}  // namespace mcs_detail

// Returns (center, radius) of the minimum covering sphere of the points.
inline std::pair<Vec3, double> minimum_covering_sphere(const Points& pts) {
    if (pts.empty()) throw std::invalid_argument("minimum_covering_sphere: no points");
    const Vec3 point_0 = pts[0];
    Vec3 center = point_0;
    int64_t point_1_index = 0;
    double max_d = -INFINITY;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double d = std::sqrt(sq(pts[i][0] - point_0[0]) + sq(pts[i][1] - point_0[1]) +
                                   sq(pts[i][2] - point_0[2]));
        if (d > max_d) {
            max_d = d;
            point_1_index = static_cast<int64_t>(i);
        }
    }
    std::vector<int64_t> cand = {point_1_index};
    std::vector<uint8_t> is_cand(pts.size());
    std::vector<double> p(pts.size());
    bool finished = false;
    while (!finished) finished = mcs_detail::find_next_candidate(pts, center, cand, is_cand, p);
    const Vec3 diff = mcs_detail::sub(pts[cand[0]], center);
    return {center, std::sqrt(mcs_detail::dot(diff, diff))};
}

}  // namespace snowagg
