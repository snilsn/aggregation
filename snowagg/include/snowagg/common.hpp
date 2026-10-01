// Common types and numeric helpers.
//
// Many helpers here exist only to reproduce the exact floating point
// behaviour of the Python/numpy reference implementation (package
// `aggregation`), so that the C++ port can be tested bit for bit against it.
// Build with -ffp-contract=off: fused multiply-adds would change results.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace snowagg {

using Vec3 = std::array<double, 3>;
using IVec3 = std::array<int64_t, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using Points = std::vector<Vec3>;

constexpr double PI = 3.141592653589793;  // == numpy.pi

// Python `a ** b` on Python floats and numpy float64 scalars calls libm
// pow(). pow(x, 2.0) is NOT always equal to x*x (differs in ~0.1% of cases),
// so scalar `**` in the reference must be reproduced with a real pow() call.
// The volatile hides the exponent from the compiler, which would otherwise
// replace pow(x, 2.0) by x*x.
inline double pypow(double a, double b) {
    volatile double exponent = b;
    return std::pow(a, exponent);
}

// numpy array `x**2` is computed as x*x (np.square fast path).
inline double sq(double a) { return a * a; }

// Python / numpy float modulo: result has the sign of the divisor.
inline double pymod(double a, double b) {
    double m = std::fmod(a, b);
    if (m != 0.0) {
        if ((b < 0) != (m < 0)) m += b;
    } else {
        m = std::copysign(0.0, b);
    }
    return m;
}

// numpy.round / Python round(float): round half to even.
inline double npround(double a) { return std::nearbyint(a); }

// Python int(float) / astype(int): truncation toward zero.
inline int64_t pyint(double a) { return static_cast<int64_t>(a); }

struct Extent {
    // [[xmin, xmax], [ymin, ymax], [zmin, zmax]] like Aggregate.extent
    std::array<std::array<double, 2>, 3> e{};
    double& operator()(int dim, int side) { return e[dim][side]; }
    double operator()(int dim, int side) const { return e[dim][side]; }
};

inline Extent extent_of(const Points& X) {
    Extent ext;
    if (X.empty()) return ext;
    const int64_t n = static_cast<int64_t>(X.size());
    double lo[3] = {X[0][0], X[0][1], X[0][2]}, hi[3] = {X[0][0], X[0][1], X[0][2]};
#pragma omp parallel for schedule(static) reduction(min : lo[:3]) reduction(max : hi[:3]) if (n > 50000)
    for (int64_t i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], X[i][d]);
            hi[d] = std::max(hi[d], X[i][d]);
        }
    for (int d = 0; d < 3; ++d) {
        ext(d, 0) = lo[d];
        ext(d, 1) = hi[d];
    }
    return ext;
}

// numpy's pairwise summation (loops_utils.h: DOUBLE_pairwise_sum) of
// column d of X, rows [i0, i0+n).
inline double np_pairwise_sum(const Points& X, int d, size_t i0, size_t n) {
    if (n < 8) {
        double res = -0.0;
        for (size_t i = 0; i < n; ++i) res += X[i0 + i][d];
        return res;
    } else if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = X[i0 + j][d];
        size_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += X[i0 + i + j][d];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += X[i0 + i][d];
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return np_pairwise_sum(X, d, i0, n2) + np_pairwise_sum(X, d, i0 + n2, n - n2);
    }
}

// Column mean X.mean(0) of an (N,3) array, reproducing numpy's summation
// order, which depends on the memory layout of the array:
//  * C order: rows are accumulated sequentially;
//  * Fortran order: each column is summed pairwise in chunks of 8192
//    elements (numpy's buffer size) and the chunk sums are accumulated.
inline Vec3 mean_rows(const Points& X, bool fortran_order) {
    const size_t N = X.size();
    Vec3 s;
    if (fortran_order && N > 1) {
        constexpr size_t BUF = 8192;
        const int64_t nb = static_cast<int64_t>((N + BUF - 1) / BUF);
        std::vector<Vec3> part(nb);
#pragma omp parallel for schedule(static) if (nb > 8)
        for (int64_t b = 0; b < nb; ++b)
            for (int d = 0; d < 3; ++d)
                part[b][d] = np_pairwise_sum(X, d, b * BUF, std::min(BUF, N - b * BUF));
        s = part[0];
        for (int64_t b = 1; b < nb; ++b)
            for (int d = 0; d < 3; ++d) s[d] += part[b][d];
    } else {
        s = X.at(0);
        for (size_t i = 1; i < N; ++i) {
            s[0] += X[i][0];
            s[1] += X[i][1];
            s[2] += X[i][2];
        }
    }
    const double n = static_cast<double>(N);
    return {s[0] / n, s[1] / n, s[2] / n};
}

}  // namespace snowagg
