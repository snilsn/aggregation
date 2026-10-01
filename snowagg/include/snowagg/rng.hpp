// Random number generator bit-compatible with numpy's legacy global
// RandomState (MT19937), i.e. the generator behind np.random.seed(),
// np.random.rand(), np.random.randint(), np.random.shuffle(),
// np.random.randn() and scipy.stats.expon.rvs().
//
// The state can be exchanged with numpy (np.random.get_state/set_state), so
// C++ kernels can continue exactly the random stream of Python code.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace snowagg {

class Rng {
public:
    static constexpr int N = 624;
    static constexpr int M = 397;

    std::array<uint32_t, N> key{};
    int pos = N;
    bool has_gauss = false;
    double gauss_cache = 0.0;

    explicit Rng(uint32_t s = 0) { seed(s); }

    // np.random.seed(int)
    void seed(uint32_t s) {
        for (int i = 0; i < N; ++i) {
            key[i] = s;
            s = 1812433253U * (s ^ (s >> 30)) + static_cast<uint32_t>(i) + 1U;
        }
        pos = N;
        has_gauss = false;
        gauss_cache = 0.0;
    }

    // Seed from the operating system (like np.random.seed(None)).
    void seed_random() {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        std::array<uint32_t, N> k;
        seq.generate(k.begin(), k.end());
        key = k;
        key[0] |= 0x80000000U;  // guarantee a non-zero state
        pos = N;
        has_gauss = false;
        gauss_cache = 0.0;
    }

    uint32_t next_u32() {
        if (pos == N) regenerate();
        uint32_t y = key[pos++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    uint64_t next_u64() {
        uint64_t hi = next_u32();
        return (hi << 32) | next_u32();
    }

    // np.random.rand() / random_sample(): 53-bit double in [0, 1)
    double rand() {
        const uint32_t a = next_u32() >> 5;
        const uint32_t b = next_u32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    // np.random.randint(high) (legacy, masked rejection sampling)
    int64_t randint(int64_t high) {
        if (high <= 0) throw std::invalid_argument("randint: high <= 0");
        const uint64_t rng = static_cast<uint64_t>(high - 1);
        if (rng == 0) return 0;
        const uint64_t mask = gen_mask(rng);
        if (rng <= 0xFFFFFFFFULL) {
            if (rng == 0xFFFFFFFFULL) return static_cast<int64_t>(next_u32());
            uint32_t val;
            while ((val = (next_u32() & static_cast<uint32_t>(mask))) > rng) {
            }
            return static_cast<int64_t>(val);
        }
        uint64_t val;
        while ((val = (next_u64() & mask)) > rng) {
        }
        return static_cast<int64_t>(val);
    }

    // numpy random_interval(max): uniform integer in [0, max]
    uint64_t interval(uint64_t max) {
        if (max == 0) return 0;
        const uint64_t mask = gen_mask(max);
        uint64_t value;
        if (max <= 0xFFFFFFFFULL) {
            while ((value = (next_u32() & mask)) > max) {
            }
        } else {
            while ((value = (next_u64() & mask)) > max) {
            }
        }
        return value;
    }

    // np.random.shuffle (legacy) on a sequence of rows
    template <class T>
    void shuffle(std::vector<T>& v) {
        const int64_t n = static_cast<int64_t>(v.size());
        for (int64_t i = n - 1; i >= 1; --i) {
            const uint64_t j = interval(static_cast<uint64_t>(i));
            if (static_cast<int64_t>(j) == i) continue;
            std::swap(v[i], v[j]);
        }
    }

    // legacy standard_exponential (scipy.stats.expon.rvs uses this)
    double standard_exponential() { return -std::log(1.0 - rand()); }

    // legacy_gauss: polar Box-Muller with a cached second value
    double gauss() {
        if (has_gauss) {
            const double tmp = gauss_cache;
            has_gauss = false;
            gauss_cache = 0.0;
            return tmp;
        }
        double f, x1, x2, r2;
        do {
            x1 = 2.0 * rand() - 1.0;
            x2 = 2.0 * rand() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_cache = f * x1;
        has_gauss = true;
        return f * x2;
    }

private:
    static uint64_t gen_mask(uint64_t max) {
        uint64_t mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        return mask;
    }

    void regenerate() {
        constexpr uint32_t MATRIX_A = 0x9908b0dfU;
        constexpr uint32_t UPPER = 0x80000000U;
        constexpr uint32_t LOWER = 0x7fffffffU;
        uint32_t y;
        int i;
        for (i = 0; i < N - M; ++i) {
            y = (key[i] & UPPER) | (key[i + 1] & LOWER);
            key[i] = key[i + M] ^ (y >> 1) ^ (-(y & 1U) & MATRIX_A);
        }
        for (; i < N - 1; ++i) {
            y = (key[i] & UPPER) | (key[i + 1] & LOWER);
            key[i] = key[i + (M - N)] ^ (y >> 1) ^ (-(y & 1U) & MATRIX_A);
        }
        y = (key[N - 1] & UPPER) | (key[0] & LOWER);
        key[N - 1] = key[M - 1] ^ (y >> 1) ^ (-(y & 1U) & MATRIX_A);
        pos = 0;
    }
};

}  // namespace snowagg
