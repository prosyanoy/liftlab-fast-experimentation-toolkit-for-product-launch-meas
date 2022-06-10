// Header-only multithreaded resampling kernels (no Python dependency, so Catch2 can test them).
//
// Every replicate i draws from its own RNG stream seeded from (seed, i), so the output is
// bit-identical regardless of how many threads are used.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace liftlab {

inline std::uint64_t splitmix64(std::uint64_t &state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// xoshiro256** seeded through splitmix64 (cheap to construct, unlike std::mt19937_64).
class Rng {
public:
    Rng(std::uint64_t seed, std::uint64_t stream) {
        std::uint64_t sm = seed ^ (stream * 0xD1B54A32D192ED03ULL + 0x8CB92BA72F3D8DD7ULL);
        for (auto &w : s_) w = splitmix64(sm);
    }
    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }
    // Uniform integer in [0, n) via 64x64->128 multiply (bias < n / 2^64, negligible).
    std::size_t below(std::size_t n) {
        return static_cast<std::size_t>((static_cast<unsigned __int128>(next()) * n) >> 64);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

inline unsigned resolve_threads(unsigned requested, std::size_t work_items) {
    unsigned t = requested ? requested : std::thread::hardware_concurrency();
    if (t == 0) t = 1;
    if (work_items < t) t = static_cast<unsigned>(std::max<std::size_t>(work_items, 1));
    return t;
}

// Runs fn(begin, end) over [0, n) split into contiguous ranges, one per thread.
template <class Fn>
void parallel_ranges(std::size_t n, unsigned threads, Fn fn) {
    const unsigned t = resolve_threads(threads, n);
    if (t <= 1) {
        fn(std::size_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(t);
    const std::size_t chunk = (n + t - 1) / t;
    for (unsigned k = 0; k < t; ++k) {
        const std::size_t b = k * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([=, &fn] { fn(b, e); });
    }
    for (auto &th : pool) th.join();
}

inline double mean_of(const double *x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    return s / static_cast<double>(n);
}

// Bootstrap distribution of mean(treatment) - mean(control); each group resampled independently.
inline std::vector<double> bootstrap_mean_diff(const double *control, std::size_t n_c,
                                               const double *treatment, std::size_t n_t,
                                               std::size_t n_boot, std::uint64_t seed,
                                               unsigned threads = 0) {
    if (n_c == 0 || n_t == 0) throw std::invalid_argument("both groups must be non-empty");
    std::vector<double> out(n_boot);
    parallel_ranges(n_boot, threads, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            Rng rng(seed, i);
            double sc = 0.0, st = 0.0;
            for (std::size_t j = 0; j < n_c; ++j) sc += control[rng.below(n_c)];
            for (std::size_t j = 0; j < n_t; ++j) st += treatment[rng.below(n_t)];
            out[i] = st / static_cast<double>(n_t) - sc / static_cast<double>(n_c);
        }
    });
    return out;
}

// Null distribution of mean(treatment) - mean(control) under random relabelling of the pooled data.
inline std::vector<double> permutation_mean_diff(const double *control, std::size_t n_c,
                                                 const double *treatment, std::size_t n_t,
                                                 std::size_t n_perm, std::uint64_t seed,
                                                 unsigned threads = 0) {
    if (n_c == 0 || n_t == 0) throw std::invalid_argument("both groups must be non-empty");
    const std::size_t n = n_c + n_t;
    std::vector<double> pooled(n);
    std::copy(control, control + n_c, pooled.begin());
    std::copy(treatment, treatment + n_t, pooled.begin() + static_cast<std::ptrdiff_t>(n_c));
    const double total = std::accumulate(pooled.begin(), pooled.end(), 0.0);

    std::vector<double> out(n_perm);
    parallel_ranges(n_perm, threads, [&](std::size_t b, std::size_t e) {
        std::vector<double> work(pooled);  // thread-local shuffle buffer
        for (std::size_t i = b; i < e; ++i) {
            Rng rng(seed, i);
            work = pooled;  // reset so each replicate depends only on (seed, i)
            // Partial Fisher-Yates: only the first n_t slots (the "treatment" group) are needed.
            double st = 0.0;
            for (std::size_t j = 0; j < n_t; ++j) {
                std::size_t k = j + rng.below(n - j);
                std::swap(work[j], work[k]);
                st += work[j];
            }
            out[i] = st / static_cast<double>(n_t) - (total - st) / static_cast<double>(n_c);
        }
    });
    return out;
}

}  // namespace liftlab
