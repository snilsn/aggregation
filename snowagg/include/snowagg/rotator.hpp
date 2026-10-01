// Random rotations (port of aggregation/rotator.py).
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include "common.hpp"
#include "rng.hpp"

namespace snowagg {

inline Mat3 matmul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    return C;
}

inline Mat3 transpose(const Mat3& A) {
    Mat3 T{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[i][j] = A[j][i];
    return T;
}

// Rotator.rotation_matrix: Ra(alpha) . Rb(beta) . Rp(psi)
inline Mat3 rotation_matrix(double alpha, double beta, double psi) {
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const double cp = std::cos(psi), sp = std::sin(psi);
    const Mat3 Rb = {{{1.0, 0.0, 0.0}, {0.0, cb, -sb}, {0.0, sb, cb}}};
    const Mat3 Ra = {{{ca, -sa, 0.0}, {sa, ca, 0.0}, {0.0, 0.0, 1.0}}};
    const Mat3 Rp = {{{cp, -sp, 0.0}, {sp, cp, 0.0}, {0.0, 0.0, 1.0}}};
    return matmul(matmul(Ra, Rb), Rp);
}

// Minimum size for running loops over points in parallel.
constexpr int64_t PAR_MIN = 50000;

// x' = R x for every point
inline void apply_rotation(const Mat3& R, Points& X) {
    const int64_t n = static_cast<int64_t>(X.size());
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; ++i) {
        auto& p = X[i];
        const double x = p[0], y = p[1], z = p[2];
        p[0] = R[0][0] * x + R[0][1] * y + R[0][2] * z;
        p[1] = R[1][0] * x + R[1][1] * y + R[1][2] * z;
        p[2] = R[2][0] * x + R[2][1] * y + R[2][2] * z;
    }
}

// X' = X . M for (N,3) row points (np.dot(X, M))
inline void right_multiply(Points& X, const Mat3& M) {
    const int64_t n = static_cast<int64_t>(X.size());
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; ++i) {
        auto& p = X[i];
        const double x = p[0], y = p[1], z = p[2];
        p[0] = x * M[0][0] + y * M[1][0] + z * M[2][0];
        p[1] = x * M[0][1] + y * M[1][1] + z * M[2][1];
        p[2] = x * M[0][2] + y * M[1][2] + z * M[2][2];
    }
}

class Rotator {
public:
    virtual ~Rotator() = default;
    // Draw the random linear map, consuming random numbers exactly like the
    // Python rotator does.
    virtual Mat3 draw(Rng& rng) const {
        (void)rng;
        return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    }
    // True for the identity (no random numbers, no arithmetic).
    virtual bool identity() const { return true; }
    // Rotate the points around the origin.
    void rotate(Points& X, Rng& rng) const {
        if (!identity()) apply_rotation(draw(rng), X);
    }
};

// Identity rotation. (The Python base class Rotator.rotate has a broken
// signature; the scripts used an external NullRotator for this.)
class NullRotator : public Rotator {};

class UniformRotator : public Rotator {
public:
    bool identity() const override { return false; }
    Mat3 draw(Rng& rng) const override {
        const double alpha = rng.rand() * 2 * PI;
        const double beta = std::acos(1.0 - 2 * rng.rand());
        const double psi = rng.rand() * 2 * PI;
        return rotation_matrix(alpha, beta, psi);
    }
};

class HorizontalRotator : public Rotator {
public:
    bool identity() const override { return false; }
    Mat3 draw(Rng& rng) const override {
        const double alpha = rng.rand() * 2 * PI;
        return rotation_matrix(alpha, 0.0, 0.0);
    }
};

// Inverse-CDF sampling from a tabulated PDF (Python SamplePDF: trapezoidal
// cumulative integral + scipy interp1d, which delegates to np.interp).
class SamplePDF {
public:
    std::vector<double> x, Y;

    // From a precomputed table (x: sample values, Y: normalized CDF).
    SamplePDF(std::vector<double> x_, std::vector<double> Y_) : x(std::move(x_)), Y(std::move(Y_)) {
        if (x.size() != Y.size() || x.size() < 2) throw std::invalid_argument("SamplePDF: bad table");
    }

    SamplePDF(const std::function<double(double)>& pdf, double a, double b, int num_points = 1024) {
        // np.linspace(a, b, num_points)
        x.resize(num_points);
        const double step = (b - a) / (num_points - 1);
        for (int i = 0; i < num_points; ++i) x[i] = i * step + a;
        x[num_points - 1] = b;
        std::vector<double> y(num_points);
        for (int i = 0; i < num_points; ++i) y[i] = pdf(x[i]);
        // np.hstack((0, cumtrapz(y, x)))
        Y.assign(num_points, 0.0);
        double cum = 0.0;
        for (int i = 1; i < num_points; ++i) {
            const double d = x[i] - x[i - 1];
            const double v = d * (y[i] + y[i - 1]) / 2.0;
            cum = (i == 1) ? v : cum + v;
            Y[i] = cum;
        }
        const double last = Y.back();
        for (auto& v : Y) v /= last;
    }

    // np.interp(u, Y, x)
    double operator()(double u) const {
        const size_t n = Y.size();
        if (u < Y[0]) return x[0];
        if (u > Y[n - 1]) return x[n - 1];
        // largest j with Y[j] <= u
        size_t j = std::upper_bound(Y.begin(), Y.end(), u) - Y.begin() - 1;
        if (j == n - 1) return x[j];
        if (Y[j] == u) return x[j];
        const double slope = (x[j + 1] - x[j]) / (Y[j + 1] - Y[j]);
        double res = slope * (u - Y[j]) + x[j];
        if (std::isnan(res)) {
            res = slope * (u - Y[j + 1]) + x[j + 1];
            if (std::isnan(res) && x[j] == x[j + 1]) res = x[j];
        }
        return res;
    }

    double sample(Rng& rng) const { return (*this)(rng.rand()); }
};

// Uniform azimuth, Gaussian-weighted canting angle.
class PartialAligningRotator : public Rotator {
public:
    double exp_sig;
    bool random_flip;
    SamplePDF beta_sample;

    explicit PartialAligningRotator(double exp_sig_deg = 40, bool random_flip_ = false)
        : exp_sig(exp_sig_deg * PI / 180),
          random_flip(random_flip_),
          beta_sample([s = exp_sig_deg * PI / 180](double x) {
              return std::sin(x) * std::exp(-0.5 * sq(x / s));
          }, 0, PI) {}

    // Construct from a table computed elsewhere (e.g. by numpy).
    PartialAligningRotator(double exp_sig_deg, bool random_flip_, SamplePDF table)
        : exp_sig(exp_sig_deg * PI / 180), random_flip(random_flip_), beta_sample(std::move(table)) {}

    bool identity() const override { return false; }
    Mat3 draw(Rng& rng) const override {
        const double alpha = rng.rand() * 2 * PI;
        const double beta = beta_sample.sample(rng);
        Mat3 R = rotation_matrix(alpha, beta, 0.0);
        if (random_flip && (rng.rand() > 0.5)) {
            // X[2] = -X[2]; X[1] = -X[1] after rotating. Negating the matrix
            // rows gives bit-identical results.
            for (int j = 0; j < 3; ++j) {
                R[1][j] = -R[1][j];
                R[2][j] = -R[2][j];
            }
        }
        return R;
    }
};

}  // namespace snowagg
