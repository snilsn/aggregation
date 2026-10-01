// Ice crystal geometries (port of aggregation/crystal.py).
//
// Dimension relations are written in exactly the same operation order as the
// Python reference; scalar `**` becomes pypow() (libm pow), see common.hpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"

namespace snowagg {

class Crystal {
public:
    virtual ~Crystal() = default;
    // Radius corresponding to the maximum/major diameter.
    virtual double max_radius() const = 0;
    virtual bool is_inside(double x, double y, double z) const = 0;
    virtual std::string name() const = 0;
};

// Port of scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c) with the
// scipy default tolerances, so that root-derived dimensions match exactly.
inline double brentq(const std::function<double(double)>& f, double xa, double xb,
                     double xtol = 2e-12, double rtol = 8.881784197001252e-16,
                     int iter = 100) {
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    double delta, stry, dpre, dblk;
    fpre = f(xpre);
    fcur = f(xcur);
    if (fpre == 0) return xpre;
    if (fcur == 0) return xcur;
    if (std::signbit(fpre) == std::signbit(fcur))
        throw std::runtime_error("brentq: f(a) and f(b) must have different signs");
    for (int i = 0; i < iter; i++) {
        if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        delta = (xtol + rtol * std::fabs(xcur)) / 2;
        sbis = (xblk - xcur) / 2;
        if (fcur == 0 || std::fabs(sbis) < delta) return xcur;
        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta) {
            xcur += scur;
        } else {
            xcur += (sbis > 0 ? delta : -delta);
        }
        fcur = f(xcur);
    }
    throw std::runtime_error("brentq: failed to converge");
}

// ---- dimension relations -------------------------------------------------
namespace dims {

// Plate, from Hong (2007)
inline double plate_L_from_a(double a) {
    if (a <= 2e-6) {
        return 2 * a;
    } else if (a < 5e-6) {
        return (2 + (2.4883 * pypow(a * 1e6, 0.474) - 2.0) / 4.0 * ((a * 1e6) - 1.0)) * 1e-6;
    } else {
        return (2.4883 * pypow(a * 1e6, 0.474)) * 1e-6;
    }
}

inline double column_a_from_L(double L) {
    if (L < 100e-6) return 0.35 * L;
    return 3.48 * pypow(L * 1e6, 0.5) * 1e-6;
}

inline double column_L_from_a(double a) {
    if (a < 35e-6) return a / 0.35;
    return pypow(a * 1e6 / 3.48, 2) * 1e-6;
}

// Needle, from Pruppacher and Klett
inline double needle_L_from_a(double a) {
    return pypow(1.0 / 3.527e-2 * 2 * a * (1e2), (1.0 / 0.437)) * 1e-2;
}

inline double needle_a_from_L(double L) {
    return 3.527e-2 * pypow(L * 1e2, 0.437) / 2.0 * 1e-2;
}

// Dendrite bounding prism, from Pruppacher and Klett
inline double dendrite_L_from_a(double a) {
    return (9.022e-3 * pypow(a * 2.0 * 1e2, 0.377)) * 1e-2;
}

}  // namespace dims

// ---- hexagonal prisms -----------------------------------------------------

// Hexagonal prism with side a and height L (Python: Plate and subclasses).
class HexPrism : public Crystal {
public:
    double D, a, L, V, A, r;
    std::array<std::array<double, 2>, 6> centers;

    double max_radius() const override { return std::max(a, L / 2.0); }

    bool is_inside(double x, double y, double z) const override {
        if (!(std::fabs(z) <= L / 2.0)) return false;
        for (int i = 0; i < 6; ++i) {
            const double cx = centers[i][0];
            const double cy = centers[i][1];
            const double dx = x - cx;
            const double dy = y - cy;
            if (!(dx * cx + dy * cy <= 0)) return false;
        }
        return true;
    }

protected:
    // Mirrors Plate.__init__(D): a = D/2, L = L_from_a(a), ...
    HexPrism(double D_plate, double (*L_from_a)(double)) : D(D_plate), a(D_plate / 2.0) {
        L = L_from_a(a);
        V = 3.0 * std::sqrt(3.0) / 2.0 * pypow(a, 2) * L;
        A = (3.0 * std::sqrt(3.0) * pypow(a, 2) + 6 * a * L) / 4.0;
        r = std::sqrt(3.0 / 4.0) * a;
        centers = {{{0.0, r},
                    {0.75 * a, 0.5 * r},
                    {0.75 * a, -0.5 * r},
                    {0.0, -r},
                    {-0.75 * a, -0.5 * r},
                    {-0.75 * a, 0.5 * r}}};
    }
};

class Plate : public HexPrism {
public:
    explicit Plate(double D) : HexPrism(D, dims::plate_L_from_a) {}
    std::string name() const override { return "plate"; }
};

// D is the column length.
class Column : public HexPrism {
public:
    explicit Column(double D) : HexPrism(dims::column_a_from_L(D) * 2.0, dims::column_L_from_a) {}
    std::string name() const override { return "column"; }
};

// D is the needle length.
class Needle : public HexPrism {
public:
    explicit Needle(double D) : HexPrism(dims::needle_a_from_L(D) * 2.0, dims::needle_L_from_a) {}
    std::string name() const override { return "needle"; }
};

// Dendrite: 2D hexagonal-grid dendrite (Reiter) extruded to the height of
// the bounding hexagonal prism.
class Dendrite : public HexPrism {
public:
    int64_t grid_size;  // NOTE: the constructor argument (default 400), not the grid shape
    int64_t rows, cols;
    double grid_D;
    std::vector<uint8_t> ice;

    // hex_grid: row-major (rows x cols) array as produced by generate_dendrite
    Dendrite(double D, const std::vector<double>& hex_grid, int64_t rows_, int64_t cols_,
             int64_t grid_size_ = 400)
        : HexPrism(D, dims::dendrite_L_from_a), grid_size(grid_size_), rows(rows_), cols(cols_) {
        if (static_cast<int64_t>(hex_grid.size()) != rows * cols)
            throw std::invalid_argument("Dendrite: hex_grid size mismatch");
        if (grid_size > rows || grid_size > cols)
            throw std::invalid_argument("Dendrite: grid_size larger than hex_grid");
        ice.resize(hex_grid.size());
        for (size_t k = 0; k < hex_grid.size(); ++k) ice[k] = hex_grid[k] >= 1.0;
        // mg = hex_grid.max(1); r = arange(len(mg))[mg > 1.0]
        int64_t first = -1, last = -1;
        for (int64_t i = 0; i < rows; ++i) {
            double mg = hex_grid[i * cols];
            for (int64_t j = 1; j < cols; ++j) mg = std::max(mg, hex_grid[i * cols + j]);
            if (mg > 1.0) {
                if (first < 0) first = i;
                last = i;
            }
        }
        if (first < 0) throw std::invalid_argument("Dendrite: hex_grid contains no ice");
        const double D_width = static_cast<double>(last - first) + 1.0;
        grid_D = D_width / static_cast<double>(grid_size);
    }

    std::string name() const override { return "dendrite"; }

    bool is_inside(double x, double y, double z) const override {
        if (!(std::fabs(z) <= L / 2.0)) return false;
        const double scale = static_cast<double>(grid_size) * grid_D / D;
        const double half = static_cast<double>(grid_size) / 2.0;
        double j = x * scale + half;
        const double i = npround(y * scale + half);
        if (pymod(npround(j), 2) == 1) j += 0.5;
        j = npround(j);
        const int64_t ii = static_cast<int64_t>(i), jj = static_cast<int64_t>(j);
        if (ii < 0 || ii >= grid_size || jj < 0 || jj >= grid_size) return false;
        return ice[ii * cols + jj] != 0;
    }
};

// ---- bullet rosettes ------------------------------------------------------

// Shared geometry of Rosette and Bullet (Hong 2007).
class BulletBase : public Crystal {
public:
    double D, L, a, t;

    double max_radius() const override { return D / 2.0; }

protected:
    // tan_alpha: np.tan(28 deg) as evaluated by numpy on the machine running
    // the reference (numpy's SIMD tan can differ from libm by 1 ulp). Pass
    // NaN to use std::tan.
    BulletBase(double D_, double tan_alpha) : D(D_) {
        const double alpha = 28 * (PI / 180.0);
        if (std::isnan(tan_alpha)) tan_alpha = std::tan(alpha);
        const double f = std::sqrt(3.0) * 1.552 / tan_alpha;
        const double Dum = D * 1e6;
        auto L_func = [f, Dum](double Lx) { return 2 * Lx + f * pypow(Lx, 0.63) - Dum; };
        L = brentq(L_func, 0, D * 1e6 / 2.0) * 1e-6;
        a = (1.552 * pypow(L * 1e6, 0.63)) * 1e-6;
        t = std::sqrt(3.0) * a / (2 * tan_alpha);
    }

    bool inside_bullet(double x, double y, double z, bool positive_only) const {
        if (positive_only && !(z > 0)) return false;
        if (!(std::fabs(z) < D / 2.0)) return false;
        double z_ratio = std::fabs(z) / t;
        if (z_ratio > 1) z_ratio = 1.0;
        const double ab = a * z_ratio;
        const double rb = std::sqrt(3.0 / 4.0) * ab;
        if (!(std::fabs(y) <= rb)) return false;
        static constexpr int signs[4][2] = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
        for (const auto& s : signs) {
            const double cx = s[0] * 0.75 * ab;
            const double cy = s[1] * 0.5 * rb;
            if (!((x - cx) * cx + (y - cy) * cy < 0)) return false;
        }
        return true;
    }
};

// 6-branch bullet rosette; D is the maximum diameter.
class Rosette : public BulletBase {
public:
    explicit Rosette(double D, double tan_alpha = NAN) : BulletBase(D, tan_alpha) {}
    std::string name() const override { return "rosette"; }
    bool is_inside(double x, double y, double z) const override {
        return inside_bullet(x, y, z, false) || inside_bullet(y, z, x, false) ||
               inside_bullet(z, x, y, false);
    }
};

// Single bullet; D is the maximum diameter.
class Bullet : public BulletBase {
public:
    explicit Bullet(double D, double tan_alpha = NAN) : BulletBase(D, tan_alpha) {}
    std::string name() const override { return "bullet"; }
    bool is_inside(double x, double y, double z) const override {
        return inside_bullet(x, y, z, true);
    }
};

// Oblate spheroid.
class Spheroid : public Crystal {
public:
    double a, c, axis_ratio;
    explicit Spheroid(double D_max, double axis_ratio_ = 1.0)
        : a(D_max / 2.0), c(a * axis_ratio_), axis_ratio(axis_ratio_) {}
    std::string name() const override { return "spheroid"; }
    double max_radius() const override { return a; }
    bool is_inside(double x, double y, double z) const override {
        return (sq(x) + sq(y)) / pypow(a, 2) + sq(z / c) <= 1;
    }
};

}  // namespace snowagg
