// Spatial hash indices that reproduce aggregation/index.py exactly,
// including the order in which items_near() returns items (this order
// matters for tie-breaking in the riming and deposition code):
//   * cell index = int(coordinate / elem_size) (truncation toward zero)
//   * cells are visited x-major, then y (then z)
//   * items within a cell are returned in insertion order
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "common.hpp"

namespace snowagg {

namespace detail {
inline uint64_t key2(int32_t x, int32_t y) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
}
struct Key3 {
    int32_t x, y, z;
    bool operator==(const Key3& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct Key3Hash {
    size_t operator()(const Key3& k) const {
        uint64_t h = static_cast<uint32_t>(k.x);
        h = h * 0x9E3779B97F4A7C15ULL ^ static_cast<uint32_t>(k.y);
        h = h * 0x9E3779B97F4A7C15ULL ^ static_cast<uint32_t>(k.z);
        return static_cast<size_t>(h ^ (h >> 29));
    }
};
struct Key2Hash {
    size_t operator()(uint64_t k) const {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        return static_cast<size_t>(k);
    }
};
}  // namespace detail

// 2D index of 3D points by their (x, y) coordinates (Index2D with the
// coordinate rows as the indexed objects).
class Index2D {
public:
    explicit Index2D(double elem_size = 1.0) : elem_(elem_size) {}

    void insert(const Vec3& p) {
        const int32_t xi = static_cast<int32_t>(p[0] / elem_);
        const int32_t yi = static_cast<int32_t>(p[1] / elem_);
        cells_[detail::key2(xi, yi)].push_back(p);
    }

    void insert(const Points& X) {
        cells_.reserve(cells_.size() + X.size());
        for (const auto& p : X) insert(p);
    }

    // Calls f(item) for all items in the cells near (x, y), in the order of
    // Index2D.items_near(p, search_rad).
    template <class F>
    void for_each_near(double x, double y, double search_rad, F&& f) const {
        const double px = x / elem_, py = y / elem_;
        const double rad = search_rad / elem_;
        const int64_t x0 = pyint(px - rad), x1 = pyint(px + rad);
        const int64_t y0 = pyint(py - rad), y1 = pyint(py + rad);
        for (int64_t xi = x0; xi <= x1; ++xi)
            for (int64_t yi = y0; yi <= y1; ++yi) {
                auto it = cells_.find(detail::key2(static_cast<int32_t>(xi), static_cast<int32_t>(yi)));
                if (it == cells_.end()) continue;
                for (const auto& p : it->second) f(p);
            }
    }

private:
    double elem_;
    std::unordered_map<uint64_t, std::vector<Vec3>, detail::Key2Hash> cells_;
};

// Same semantics and item order as Index2D, but the initial points are
// indexed in a flat CSR grid of point indices (fast to build and query; the
// points must stay alive and unchanged while the index is used). Points
// inserted later are kept in a hash map and returned after the initial
// points of the same cell, which preserves insertion order. Falls back to
// hashing everything if the grid would be huge.
class FlatIndex2D {
public:
    FlatIndex2D(double elem_size, const Points& X, int64_t max_cells = int64_t(1) << 26)
        : elem_(elem_size), base_(&X) {
        const int64_t n = static_cast<int64_t>(X.size());
        if (n == 0) return;
        std::vector<int32_t> cx(n), cy(n);
        int32_t xmin = INT32_MAX, xmax = INT32_MIN, ymin = INT32_MAX, ymax = INT32_MIN;
#pragma omp parallel for schedule(static) reduction(min : xmin, ymin) reduction(max : xmax, ymax) if (n > 50000)
        for (int64_t i = 0; i < n; ++i) {
            cx[i] = static_cast<int32_t>(X[i][0] / elem_);
            cy[i] = static_cast<int32_t>(X[i][1] / elem_);
            xmin = std::min(xmin, cx[i]);
            xmax = std::max(xmax, cx[i]);
            ymin = std::min(ymin, cy[i]);
            ymax = std::max(ymax, cy[i]);
        }
        const int64_t nx = int64_t(xmax) - xmin + 1, ny = int64_t(ymax) - ymin + 1;
        if (nx * ny > max_cells) {  // too sparse for a flat grid
            for (const auto& p : X) insert(p);
            return;
        }
        cx0_ = xmin;
        cy0_ = ymin;
        nx_ = nx;
        ny_ = ny;
        const int64_t ncells = nx_ * ny_;
        std::vector<uint32_t> cell(n);
#pragma omp parallel for schedule(static) if (n > 50000)
        for (int64_t i = 0; i < n; ++i)
            cell[i] = static_cast<uint32_t>((int64_t(cx[i]) - cx0_) * ny_ + (int64_t(cy[i]) - cy0_));
        std::vector<int32_t>().swap(cx);
        std::vector<int32_t>().swap(cy);
        start_.assign(ncells + 1, 0);
#pragma omp parallel for schedule(static) if (n > 50000)
        for (int64_t i = 0; i < n; ++i) __atomic_fetch_add(&start_[cell[i] + 1], 1u, __ATOMIC_RELAXED);
        for (int64_t c = 1; c <= ncells; ++c) start_[c] += start_[c - 1];
        // stable scatter; each thread fills a range of cells, so the order
        // within a cell is the original order independent of the threads
        items_.resize(n);
#pragma omp parallel if (n > 50000)
        {
            int nt = 1, t = 0;
#ifdef _OPENMP
            nt = omp_get_num_threads();
            t = omp_get_thread_num();
#endif
            const uint32_t c_lo = static_cast<uint32_t>(ncells * t / nt);
            const uint32_t c_hi = static_cast<uint32_t>(ncells * (t + 1) / nt);
            std::vector<uint32_t> fill(start_.begin() + c_lo, start_.begin() + c_hi);
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t c = cell[i];
                if (c >= c_lo && c < c_hi) items_[fill[c - c_lo]++] = static_cast<uint32_t>(i);
            }
        }
    }

    void insert(const Vec3& p) {
        const int32_t xi = static_cast<int32_t>(p[0] / elem_);
        const int32_t yi = static_cast<int32_t>(p[1] / elem_);
        const int64_t gx = int64_t(xi) - cx0_, gy = int64_t(yi) - cy0_;
        if (gx >= 0 && gx < nx_ && gy >= 0 && gy < ny_) {
            if (dyn_mark_.empty()) dyn_mark_.assign((nx_ * ny_ + 63) / 64, 0);
            const int64_t c = gx * ny_ + gy;
            dyn_mark_[c >> 6] |= uint64_t(1) << (c & 63);
        } else {
            dyn_outside_ = true;
        }
        dyn_[detail::key2(xi, yi)].push_back(p);
    }

    template <class F>
    void for_each_near(double x, double y, double search_rad, F&& f) const {
        const double px = x / elem_, py = y / elem_;
        const double rad = search_rad / elem_;
        const int64_t x0 = pyint(px - rad), x1 = pyint(px + rad);
        const int64_t y0 = pyint(py - rad), y1 = pyint(py + rad);
        const Points& X = *base_;
        for (int64_t xi = x0; xi <= x1; ++xi)
            for (int64_t yi = y0; yi <= y1; ++yi) {
                const int64_t gx = xi - cx0_, gy = yi - cy0_;
                bool check_dyn = dyn_outside_;
                if (gx >= 0 && gx < nx_ && gy >= 0 && gy < ny_) {
                    const int64_t c = gx * ny_ + gy;
                    for (uint32_t k = start_[c]; k < start_[c + 1]; ++k) f(X[items_[k]]);
                    check_dyn = !dyn_mark_.empty() && ((dyn_mark_[c >> 6] >> (c & 63)) & 1);
                }
                if (check_dyn) {
                    auto it = dyn_.find(detail::key2(static_cast<int32_t>(xi), static_cast<int32_t>(yi)));
                    if (it != dyn_.end())
                        for (const auto& p : it->second) f(p);
                }
            }
    }

private:
    double elem_;
    const Points* base_;
    int64_t cx0_ = 0, cy0_ = 0, nx_ = 0, ny_ = 0;
    std::vector<uint32_t> start_;
    std::vector<uint32_t> items_;
    std::vector<uint64_t> dyn_mark_;
    bool dyn_outside_ = false;
    std::unordered_map<uint64_t, std::vector<Vec3>, detail::Key2Hash> dyn_;
};

// 3D index of points (Index3D), supporting removal.
class Index3D {
public:
    explicit Index3D(double elem_size = 1.0) : elem_(elem_size) {}

    detail::Key3 cell_of(const Vec3& p) const {
        return {static_cast<int32_t>(p[0] / elem_), static_cast<int32_t>(p[1] / elem_),
                static_cast<int32_t>(p[2] / elem_)};
    }

    void insert(const Vec3& p) {
        cells_[cell_of(p)].push_back(p);
        ++size_;
    }

    void insert(const Points& X) {
        cells_.reserve(cells_.size() + X.size());
        for (const auto& p : X) insert(p);
    }

    // Removes the first stored item equal to p. Throws if p is not indexed
    // (the Python version raises ValueError).
    void remove(const Vec3& p) {
        auto it = cells_.find(cell_of(p));
        if (it == cells_.end()) throw std::runtime_error("Index3D.remove: item not found");
        auto& v = it->second;
        auto pos = std::find(v.begin(), v.end(), p);
        if (pos == v.end()) throw std::runtime_error("Index3D.remove: item not found");
        v.erase(pos);
        if (v.empty()) cells_.erase(it);
        --size_;
    }

    size_t size() const { return size_; }

    template <class F>
    void for_each_near(const Vec3& p, double search_rad, F&& f) const {
        const double px = p[0] / elem_, py = p[1] / elem_, pz = p[2] / elem_;
        const double rad = search_rad / elem_;
        const int64_t x0 = pyint(px - rad), x1 = pyint(px + rad);
        const int64_t y0 = pyint(py - rad), y1 = pyint(py + rad);
        const int64_t z0 = pyint(pz - rad), z1 = pyint(pz + rad);
        for (int64_t xi = x0; xi <= x1; ++xi)
            for (int64_t yi = y0; yi <= y1; ++yi)
                for (int64_t zi = z0; zi <= z1; ++zi) {
                    auto it = cells_.find({static_cast<int32_t>(xi), static_cast<int32_t>(yi),
                                           static_cast<int32_t>(zi)});
                    if (it == cells_.end()) continue;
                    for (const auto& q : it->second) f(q);
                }
    }

private:
    double elem_;
    size_t size_ = 0;
    std::unordered_map<detail::Key3, std::vector<Vec3>, detail::Key3Hash> cells_;
};

}  // namespace snowagg
