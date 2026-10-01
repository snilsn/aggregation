// Volume-element aggregate and rimed aggregate (port of
// aggregation/aggregate.py: Aggregate, RimedAggregate).
//
// Bookkeeping that is pure Python (id_tree, monomer_number, visualization)
// lives in the Python wrapper.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "generator.hpp"
#include "index.hpp"
#include "rng.hpp"
#include "rotator.hpp"

namespace snowagg {

constexpr int32_t RIME_IDENT = -1;
constexpr int32_t DEPOSITION_IDENT = -2;

// Python builtins max(a, b) / min(a, b): return a unless b is strictly
// larger / smaller.
inline double pymax(double a, double b) { return (b > a) ? b : a; }
inline double pymin(double a, double b) { return (b < a) ? b : a; }

// Stable argsort of values (numpy argsort(kind='stable')).
inline std::vector<size_t> stable_argsort(const std::vector<double>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t a, size_t b) { return v[a] < v[b]; });
    return idx;
}

struct IVec3Hash {
    size_t operator()(const IVec3& k) const {
        uint64_t h = static_cast<uint64_t>(k[0]);
        h = h * 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(k[1]);
        h = h * 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(k[2]);
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

// Offsets on the outer layer of a cube of radius r, in the order of
// aggregate.outer_layer_of_cube().
inline void outer_layer_of_cube(int64_t r, std::vector<IVec3>& out) {
    out.clear();
    for (int64_t dx = -r; dx <= r; ++dx)
        for (int64_t dy = -r; dy <= r; ++dy) {
            if (std::llabs(dx) == r || std::llabs(dy) == r) {
                for (int64_t dz = -r; dz <= r; ++dz) out.push_back({dx, dy, dz});
            } else {
                out.push_back({dx, dy, -r});
                out.push_back({dx, dy, r});
            }
        }
}

struct ProjGrid {
    int64_t nx = 0, ny = 0;
    std::vector<uint8_t> data;  // row-major (nx, ny)
    int64_t count() const {
        int64_t c = 0;
        const int64_t n = static_cast<int64_t>(data.size());
#pragma omp parallel for schedule(static) reduction(+ : c) if (n > 1000000)
        for (int64_t i = 0; i < n; ++i) c += data[i];
        return c;
    }
};

class Aggregate {
public:
    double grid_res;
    Points X;
    std::vector<int32_t> ident;
    Extent extent;
    // Memory layout the Python reference would have for its X array (numpy
    // Fortran order). It only determines the summation order of X.mean(0);
    // see mean_rows(). Aggregate.__init__ stores generate().T -> Fortran.
    bool fortran_order = true;

    Aggregate(Points X_, double grid_res_, int32_t ident_ = 0, bool fortran_order_ = true)
        : grid_res(grid_res_), X(std::move(X_)), ident(X.size(), ident_), fortran_order(fortran_order_) {
        update_extent();
    }

    Aggregate(const MonodisperseGenerator& gen, Rng& rng, int32_t ident_ = 0)
        : Aggregate(gen.generate(rng), gen.grid_res, ident_) {}

    size_t size() const { return X.size(); }

    void update_extent() {
        if (X.empty()) {
            extent = Extent{};
        } else {
            extent = extent_of(X);
        }
    }

    // Recenter and update the extent.
    void update_coordinates() {
        const Vec3 m = mean_rows(X, fortran_order);
        subtract(m);
        update_extent();
    }

    void subtract(const Vec3& m) {
        const int64_t n = static_cast<int64_t>(X.size());
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            X[i][0] -= m[0];
            X[i][1] -= m[1];
            X[i][2] -= m[2];
        }
    }

    // added_fortran: layout of the added array in the reference (np.vstack
    // keeps Fortran order only if all inputs are Fortran ordered; single rows
    // count as both, empty arrays as C order).
    void add_elements(const Points& added, const std::vector<int32_t>& added_ident, bool update = true,
                      bool added_fortran = false) {
        if (added_ident.size() != added.size())
            throw std::invalid_argument("add_elements: ident size mismatch");
        auto f_ok = [](size_t rows, bool f) { return rows == 1 ? true : (rows == 0 ? false : f); };
        fortran_order = f_ok(X.size(), fortran_order) && f_ok(added.size(), added_fortran);
        X.insert(X.end(), added.begin(), added.end());
        ident.insert(ident.end(), added_ident.begin(), added_ident.end());
        if (update) update_coordinates();
    }

    void add_elements(const Points& added, int32_t added_ident, bool update = true, bool added_fortran = false) {
        add_elements(added, std::vector<int32_t>(added.size(), added_ident), update, added_fortran);
    }

    // Remove elements within sqrt(tolerance)*grid_res of each given point.
    void remove_elements(const Points& removed, double tolerance = 0.001, bool update = true) {
        std::vector<uint8_t> keep(X.size(), 1);
        const double lim = pypow(grid_res, 2) * tolerance;
        for (const auto& re : removed)
            for (size_t i = 0; i < X.size(); ++i) {
                const double d = sq(X[i][0] - re[0]) + sq(X[i][1] - re[1]) + sq(X[i][2] - re[2]);
                if (d < lim) keep[i] = 0;
            }
        size_t k = 0;
        for (size_t i = 0; i < X.size(); ++i)
            if (keep[i]) {
                X[k] = X[i];
                ident[k] = ident[i];
                ++k;
            }
        X.resize(k);
        ident.resize(k);
        fortran_order = false;  // boolean indexing returns a C-ordered copy
        if (update) update_coordinates();
    }

    // Principal axes as columns, in descending order of length; the length
    // of each axis is the RMS extent along it.
    Mat3 principal_axes() const {
        // X^T X summed in fixed blocks, so the result does not depend on
        // the number of threads
        constexpr int64_t B = 16384;
        const int64_t n = static_cast<int64_t>(X.size());
        const int64_t nb = (n + B - 1) / B;
        std::vector<std::array<double, 6>> part(nb);
#pragma omp parallel for schedule(static) if (nb > 4)
        for (int64_t b = 0; b < nb; ++b) {
            std::array<double, 6> s{};
            for (int64_t i = b * B; i < std::min(n, (b + 1) * B); ++i) {
                const auto& p = X[i];
                s[0] += p[0] * p[0];
                s[1] += p[1] * p[0];
                s[2] += p[2] * p[0];
                s[3] += p[1] * p[1];
                s[4] += p[2] * p[1];
                s[5] += p[2] * p[2];
            }
            part[b] = s;
        }
        std::array<double, 6> t{};
        for (const auto& s : part)
            for (int k = 0; k < 6; ++k) t[k] += s[k];
        Eigen::Matrix3d cov;
        cov << t[0], t[1], t[2], t[1], t[3], t[4], t[2], t[4], t[5];
        cov /= static_cast<double>(X.size());
        const double reg = pypow(grid_res, 2) / 12.;
        for (int a = 0; a < 3; ++a) cov(a, a) += reg;
        Mat3 PA{};
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        if (es.info() != Eigen::Success) return PA;  // zeros, like the LinAlgError fallback
        const auto& l = es.eigenvalues();   // ascending
        const auto& v = es.eigenvectors();
        for (int c = 0; c < 3; ++c) {
            const double s = std::sqrt(l(2 - c));
            for (int r = 0; r < 3; ++r) PA[r][c] = v(r, 2 - c) * s;
        }
        return PA;
    }

    double aspect_ratio() const {
        const Mat3 pa = principal_axes();
        double len[3];
        for (int c = 0; c < 3; ++c) len[c] = std::sqrt(sq(pa[0][c]) + sq(pa[1][c]) + sq(pa[2][c]));
        const double width = std::sqrt(0.5 * (pypow(len[0], 2) + pypow(len[1], 2)));
        return len[2] / width;
    }

    // Rotate so that the principal axes point along x, y, z (longest first).
    void align() {
        Mat3 PA = principal_axes();
        for (int c = 0; c < 3; ++c) {
            const double n = std::sqrt(sq(PA[0][c]) + sq(PA[1][c]) + sq(PA[2][c]));
            for (int r = 0; r < 3; ++r) PA[r][c] /= n;
        }
        right_multiply(X, PA);
        fortran_order = false;  // np.dot(X, PA)
        update_extent();
    }

    void rotate(const Rotator& rot, Rng& rng) {
        const Vec3 m = mean_rows(X, fortran_order);
        if (rot.identity()) {
            subtract(m);
            fortran_order = true;
            update_extent();
            return;
        }
        const Mat3 R = rot.draw(rng);
        const int64_t n = static_cast<int64_t>(X.size());
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            auto& p = X[i];
            const double x = p[0] - m[0], y = p[1] - m[1], z = p[2] - m[2];
            p[0] = R[0][0] * x + R[0][1] * y + R[0][2] * z;
            p[1] = R[1][0] * x + R[1][1] * y + R[1][2] * z;
            p[2] = R[2][0] * x + R[2][1] * y + R[2][2] * z;
        }
        fortran_order = true;  // rotator.rotate(X.T).T
        update_extent();
    }

    // 2D projection along dim (0, 1 or 2); grid spacing grid_res.
    ProjGrid project_on_dim(int dim = 2) const {
        int ax, ay;
        if (dim == 0) {
            ax = 1;
            ay = 2;
        } else if (dim == 1) {
            ax = 0;
            ay = 2;
        } else if (dim == 2) {
            ax = 0;
            ay = 1;
        } else {
            throw std::invalid_argument("Argument dim must be 0<=dim<=2.");
        }
        ProjGrid pg;
        if (X.empty()) throw std::runtime_error("project_on_dim: empty aggregate");
        const double ox = extent(ax, 0), oy = extent(ay, 0);
        const double g = grid_res;
        const int64_t n = static_cast<int64_t>(X.size());
        double xmax = -INFINITY, ymax = -INFINITY;
#pragma omp parallel for schedule(static) reduction(max : xmax, ymax) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            xmax = std::max(xmax, (X[i][ax] - ox) / g);
            ymax = std::max(ymax, (X[i][ay] - oy) / g);
        }
        pg.nx = pyint(npround(xmax)) + 1;
        pg.ny = pyint(npround(ymax)) + 1;
        pg.data.assign(pg.nx * pg.ny, 0);
        uint8_t* data = pg.data.data();
        const int64_t ny = pg.ny;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t k = pyint(npround((X[i][ax] - ox) / g)) * ny + pyint(npround((X[i][ay] - oy) / g));
            __atomic_store_n(&data[k], uint8_t(1), __ATOMIC_RELAXED);
        }
        return pg;
    }

    // Projection from the viewpoint given by Euler angles (alpha, beta). Like
    // the Python version, X is rotated in place and rotated back afterwards.
    ProjGrid project_on_direction(double alpha, double beta) {
        const Mat3 R = rotation_matrix(alpha, beta, 0);
        right_multiply(X, R);
        update_extent();
        ProjGrid pg;
        try {
            pg = project_on_dim(0);
        } catch (...) {
            right_multiply(X, transpose(R));
            update_extent();
            throw;
        }
        right_multiply(X, transpose(R));
        fortran_order = false;  // X.dot(R).dot(R.T)
        update_extent();
        return pg;
    }

    double projected_area(int dim = 2) const {
        return static_cast<double>(project_on_dim(dim).count()) * pypow(grid_res, 2);
    }

    static double projected_aspect_ratio(const ProjGrid& pg) {
        // x_proj = proj_grid.any(axis=0) (length ny); y_proj = any(axis=1) (length nx)
        int64_t x0 = -1, x1 = -1, y0 = -1, y1 = -1;
        for (int64_t j = 0; j < pg.ny; ++j) {
            bool any = false;
            for (int64_t i = 0; i < pg.nx && !any; ++i) any = pg.data[i * pg.ny + j];
            if (any) {
                if (x0 < 0) x0 = j;
                x1 = j;
            }
        }
        for (int64_t i = 0; i < pg.nx; ++i) {
            bool any = false;
            for (int64_t j = 0; j < pg.ny && !any; ++j) any = pg.data[i * pg.ny + j];
            if (any) {
                if (y0 < 0) y0 = i;
                y1 = i;
            }
        }
        return static_cast<double>(y1 - y0 + 1) / static_cast<double>(x1 - x0 + 1);
    }

    // Merge another particle into this one: random (x, y) position,
    // attached at the bottom of this particle. Returns true on success.
    bool add_particle(const Points& particle, const std::vector<int32_t>& particle_ident,
                      bool required, double pen_depth, Rng& rng) {
        if (particle.empty()) throw std::invalid_argument("add_particle: empty particle");
        const Extent pe = extent_of(particle);
        const double g = grid_res;
        const double g2 = pypow(g, 2);

        // limits for random positioning of the other particle
        const double x0 = extent(0, 0) - pe(0, 1);
        const double x1 = extent(0, 1) - pe(0, 0);
        const double y0 = extent(1, 0) - pe(1, 1);
        const double y1 = extent(1, 1) - pe(1, 0);

        bool site_found = false;
        double x_shift = 0, y_shift = 0;
        double min_z_sep = INFINITY;
        Points overlapping;
        while (!site_found) {
            x_shift = x0 + rng.rand() * (x1 - x0);
            y_shift = y0 + rng.rand() * (y1 - y0);
            double xs_min = INFINITY, xs_max = -INFINITY, ys_min = INFINITY, ys_max = -INFINITY;
            for (const auto& p : particle) {
                const double xs = p[0] + x_shift, ys = p[1] + y_shift;
                xs_min = std::min(xs_min, xs);
                xs_max = std::max(xs_max, xs);
                ys_min = std::min(ys_min, ys);
                ys_max = std::max(ys_max, ys);
            }
            const double r0 = pymax(xs_min, extent(0, 0)) - g;
            const double r1 = pymin(xs_max, extent(0, 1)) + g;
            const double r2 = pymax(ys_min, extent(1, 0)) - g;
            const double r3 = pymin(ys_max, extent(1, 1)) + g;

            if ((r0 >= r1) || (r2 >= r3)) {  // no overlap
                if (required) continue;
                break;
            }

            overlapping.clear();
            for (const auto& p : X)
                if (p[0] >= r0 && p[0] < r1 && p[1] >= r2 && p[1] < r3) overlapping.push_back(p);
            if (overlapping.empty()) {
                if (required) continue;
                break;
            }
            const FlatIndex2D index(g, overlapping);

            min_z_sep = INFINITY;
            for (const auto& p : particle) {
                const double xs = p[0] + x_shift, ys = p[1] + y_shift;
                if (!(xs >= r0 && xs < r1 && ys >= r2 && ys < r3)) continue;
                index.for_each_near(xs, ys, g, [&](const Vec3& c) {
                    const double s = sq(c[0] - xs) + sq(c[1] - ys);
                    if (s < g2) {
                        const double z_sep = c[2] - p[2] - std::sqrt(g2 - s);
                        if (z_sep < min_z_sep) min_z_sep = z_sep;
                    }
                });
            }
            site_found = !std::isinf(min_z_sep);
            if (!required) break;
        }

        if (site_found) {
            Points shifted(particle.size());
            for (size_t i = 0; i < particle.size(); ++i)
                shifted[i] = {particle[i][0] + x_shift, particle[i][1] + y_shift,
                              particle[i][2] + min_z_sep + pen_depth};
            add_elements(shifted, particle_ident, true, /*added_fortran=*/true);
        }
        return site_found;
    }

    // Arrange elements on the integer grid X/res, relocating elements that
    // would share a grid point to the nearest free points. Returns the sorted
    // integer coordinates.
    std::vector<IVec3> grid(double res, Rng& rng) const {
        std::vector<IVec3> Xc(X.size());
        for (size_t i = 0; i < X.size(); ++i)
            for (int d = 0; d < 3; ++d) Xc[i][d] = static_cast<int64_t>(npround(X[i][d] / res));
        std::sort(Xc.begin(), Xc.end());
        std::vector<IVec3> unique, overlap;
        for (size_t i = 0; i < Xc.size(); ++i) {
            if (i + 1 < Xc.size() && Xc[i] == Xc[i + 1])
                overlap.push_back(Xc[i]);
            else
                unique.push_back(Xc[i]);
        }
        rng.shuffle(overlap);
        std::unordered_set<IVec3, IVec3Hash> occupied(unique.begin(), unique.end());
        std::vector<IVec3> shell;
        for (const auto& Xm : overlap) {
            bool placed = false;
            for (int64_t r = 1; !placed; ++r) {
                outer_layer_of_cube(r, shell);
                for (const auto& d : shell) {
                    const IVec3 c = {Xm[0] + d[0], Xm[1] + d[1], Xm[2] + d[2]};
                    if (occupied.insert(c).second) {
                        unique.push_back(c);
                        placed = true;
                        break;
                    }
                }
            }
        }
        std::sort(unique.begin(), unique.end());
        return unique;
    }

    // ---- riming (RimedAggregate) ----------------------------------------

    // Add N rime particles falling vertically onto the aggregate.
    void add_rime_particles(int64_t N, double pen_depth, double compact_dist, Rng& rng) {
        const double g = grid_res;
        const double g2 = pypow(g, 2);
        const double x0 = extent(0, 0), x1 = extent(0, 1);
        const double y0 = extent(1, 0), y1 = extent(1, 1);
        const bool use_indexing = (N > 1);

        std::optional<FlatIndex2D> index2;
        if (use_indexing) index2.emplace(g, X);
        std::optional<Index3D> index3;
        if (compact_dist > 0 && use_indexing) {
            index3.emplace(g);
            index3->insert(X);
        }

        Points added(N > 0 ? N : 0);
        Points ov;
        Points ovs;
        std::vector<double> oz;
        Points near3;

        for (int64_t num = 0; num < N; ++num) {
            bool site_found = false;
            while (!site_found) {
                double xs = x0 + rng.rand() * (x1 - x0);
                double ys = y0 + rng.rand() * (y1 - y0);

                // find_overlapping(xs, ys)
                ov.clear();
                auto test = [&](const Vec3& p) {
                    if (sq(p[0] - xs) + sq(p[1] - ys) < g2 * 1) ov.push_back(p);
                };
                if (use_indexing)
                    index2->for_each_near(xs, ys, g * 1, test);
                else
                    for (const auto& p : X) test(p);
                if (ov.empty()) continue;

                // stable sort by z (numpy argsort(kind='stable'))
                ovs = ov;
                if (ovs.size() <= 32) {
                    for (size_t k = 1; k < ovs.size(); ++k) {
                        const Vec3 v = ovs[k];
                        size_t m = k;
                        while (m > 0 && v[2] < ovs[m - 1][2]) {
                            ovs[m] = ovs[m - 1];
                            --m;
                        }
                        ovs[m] = v;
                    }
                } else {
                    std::stable_sort(ovs.begin(), ovs.end(),
                                     [](const Vec3& p, const Vec3& q) { return p[2] < q[2]; });
                }

                oz.resize(ovs.size());
                for (size_t k = 0; k < ovs.size(); ++k) oz[k] = ovs[k][2];
                const double zlim = ovs[0][2] + pen_depth;
                const int64_t last_ind = std::lower_bound(oz.begin(), oz.end(), zlim) - oz.begin();
                const int64_t last_search_ind =
                    std::lower_bound(oz.begin(), oz.end(), zlim + g) - oz.begin();
                const size_t keep = std::min<size_t>(ovs.size(), static_cast<size_t>(last_search_ind + 1));
                ovs.resize(keep);
                oz.resize(keep);

                bool overlap = true;
                double zc = 0;
                for (int64_t i = last_ind - 1; i >= 0; --i) {
                    const double d_sqr = pypow(ovs[i][0] - xs, 2) + pypow(ovs[i][1] - ys, 2);
                    // (Python raises in math.sqrt if rounding makes this negative)
                    const double dz = std::sqrt(std::max(0.0, g2 - d_sqr));
                    const double z_upper = ovs[i][2] + dz;
                    const double z_lower = ovs[i][2] - dz;
                    for (const double zcand : {z_upper, z_lower}) {
                        zc = zcand;
                        overlap = false;
                        if ((i == 0) && (zc == z_lower)) break;  // attach at the last site
                        const int64_t j0 = std::lower_bound(oz.begin(), oz.end(), zc - g) - oz.begin();
                        const int64_t j1 = std::lower_bound(oz.begin(), oz.end(), zc + g) - oz.begin();
                        for (int64_t j = j0; j < j1; ++j) {
                            if (j == i) continue;
                            if (pypow(xs - ovs[j][0], 2) + pypow(ys - ovs[j][1], 2) +
                                    pypow(zc - ovs[j][2], 2) <
                                g2) {
                                overlap = true;
                                break;
                            }
                        }
                        if (!overlap) break;
                    }

                    if (!overlap) {
                        if (compact_dist > 0) {
                            near3.clear();
                            const double lim = g2 * 4;
                            auto test3 = [&](const Vec3& p) {
                                if (sq(p[0] - xs) + sq(p[1] - ys) + sq(p[2] - zc) < lim) near3.push_back(p);
                            };
                            if (use_indexing)
                                index3->for_each_near({xs, ys, zc}, g * 2, test3);
                            else
                                for (const auto& p : X) test3(p);
                            if (!near3.empty()) {
                                const Vec3 Xp = {xs, ys, zc};
                                const double lim2 = pypow(2 * g, 2);
                                Points filtered;
                                for (const auto& p : near3)
                                    if (sq(p[0] - Xp[0]) + sq(p[1] - Xp[1]) + sq(p[2] - Xp[2]) < lim2)
                                        filtered.push_back(p);
                                const Vec3 c = compact_rime(Xp, filtered, compact_dist);
                                xs = c[0];
                                ys = c[1];
                                zc = c[2];
                            }
                        }
                        added[num] = {xs, ys, zc};
                        site_found = true;
                        extent(0, 0) = pymin(extent(0, 0), xs);
                        extent(0, 1) = pymax(extent(0, 1), xs);
                        extent(1, 0) = pymin(extent(1, 0), ys);
                        extent(1, 1) = pymax(extent(1, 1), ys);
                        extent(2, 0) = pymin(extent(2, 0), zc);
                        extent(2, 1) = pymax(extent(2, 1), zc);
                        if (use_indexing) index2->insert({xs, ys, zc});
                        break;
                    }
                }
            }
        }
        add_elements(added, RIME_IDENT, true, /*added_fortran=*/false);  // np.empty((N,3))
    }

    Vec3 compact_rime(Vec3 Xp, const Points& X_near, double max_dist = 0., double min_move = 0.01,
                      double dr = 0.1, int max_iters = 100) const {
        if (max_dist <= 0.) return Xp;
        const double g = grid_res;
        const Vec3 X_old = Xp;
        const double max_dist_sqr = pypow(max_dist * g, 2);
        const double min_move_sqr = pypow(min_move * g, 2);
        const size_t n = X_near.size();
        Points dX(n);
        std::vector<double> r_sqr(n), r_sqr_norm(n);
        for (int it = 0; it < max_iters; ++it) {
            for (size_t i = 0; i < n; ++i) {
                for (int d = 0; d < 3; ++d) dX[i][d] = X_near[i][d] - Xp[d];
                r_sqr[i] = sq(dX[i][0]) + sq(dX[i][1]) + sq(dX[i][2]);
                r_sqr_norm[i] = r_sqr[i] / pypow(g, 2);
            }
            const auto nearest = stable_argsort(r_sqr);
            Vec3 F = {0.0, 0.0, 0.0};
            for (const size_t i : nearest) {
                const double denom = std::sqrt(r_sqr[i]) * r_sqr_norm[i];
                if (r_sqr_norm[i] > 1) {
                    for (int d = 0; d < 3; ++d) F[d] += dX[i][d] / denom;
                } else if (r_sqr_norm[i] < 0.01) {  // avoid singularity
                } else {
                    for (int d = 0; d < 3; ++d) F[d] -= dX[i][d] / denom;
                }
            }
            for (int d = 0; d < 3; ++d) F[d] *= dr;
            const double F_abs_sqr = sq(F[0]) + sq(F[1]) + sq(F[2]);
            if (F_abs_sqr > pypow(dr, 2)) {
                const double s = dr / std::sqrt(F_abs_sqr);
                for (int d = 0; d < 3; ++d) F[d] *= s;
            }
            for (int d = 0; d < 3; ++d) F[d] *= g;
            const Vec3 X_last = Xp;
            for (int d = 0; d < 3; ++d) Xp[d] += F[d];
            const double dist_sqr = sq(Xp[0] - X_old[0]) + sq(Xp[1] - X_old[1]) + sq(Xp[2] - X_old[2]);
            if (dist_sqr / pypow(g, 2) > max_dist_sqr) {
                // limit distance to at most max_dist
                const double s = max_dist * g / std::sqrt(dist_sqr);
                for (int d = 0; d < 3; ++d) Xp[d] = X_old[d] + (Xp[d] - X_old[d]) * s;
                break;
            }
            if (sq(Xp[0] - X_last[0]) + sq(Xp[1] - X_last[1]) + sq(Xp[2] - X_last[2]) < min_move_sqr) break;
        }
        return Xp;
    }
};

}  // namespace snowagg
