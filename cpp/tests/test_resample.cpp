#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "liftlab/resample.hpp"

using Catch::Approx;

namespace {
std::vector<double> ramp(std::size_t n, double offset) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = offset + static_cast<double>(i % 17);
    return v;
}
double mean(const std::vector<double> &v) { return liftlab::mean_of(v.data(), v.size()); }
double sd(const std::vector<double> &v) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}
}  // namespace

TEST_CASE("bootstrap is centred on the observed difference with the analytic SE") {
    auto c = ramp(500, 0.0), t = ramp(400, 2.0);
    auto d = liftlab::bootstrap_mean_diff(c.data(), c.size(), t.data(), t.size(), 4000, 1, 2);
    double obs = mean(t) - mean(c);
    double se = std::sqrt(sd(c) * sd(c) / 500 + sd(t) * sd(t) / 400);
    REQUIRE(mean(d) == Approx(obs).margin(3 * se / std::sqrt(4000.0) * 3));
    REQUIRE(sd(d) == Approx(se).epsilon(0.05));
}

TEST_CASE("results do not depend on the thread count") {
    auto c = ramp(300, 0.0), t = ramp(300, 1.0);
    auto a = liftlab::bootstrap_mean_diff(c.data(), c.size(), t.data(), t.size(), 500, 7, 1);
    auto b = liftlab::bootstrap_mean_diff(c.data(), c.size(), t.data(), t.size(), 500, 7, 4);
    REQUIRE(a == b);
    auto p1 = liftlab::permutation_mean_diff(c.data(), c.size(), t.data(), t.size(), 500, 7, 1);
    auto p4 = liftlab::permutation_mean_diff(c.data(), c.size(), t.data(), t.size(), 500, 7, 4);
    REQUIRE(p1 == p4);
}

TEST_CASE("different seeds give different draws") {
    auto c = ramp(100, 0.0), t = ramp(100, 1.0);
    auto a = liftlab::bootstrap_mean_diff(c.data(), c.size(), t.data(), t.size(), 50, 1, 1);
    auto b = liftlab::bootstrap_mean_diff(c.data(), c.size(), t.data(), t.size(), 50, 2, 1);
    REQUIRE(a != b);
}

TEST_CASE("permutation null is centred at zero with the analytic SE") {
    auto c = ramp(400, 0.0), t = ramp(600, 5.0);
    auto d = liftlab::permutation_mean_diff(c.data(), c.size(), t.data(), t.size(), 4000, 3, 2);
    REQUIRE(std::abs(mean(d)) < 0.1);
    // Under the null the SE is sd_pooled * sqrt(1/nc + 1/nt) (finite-population correction ~1).
    std::vector<double> pooled(c);
    pooled.insert(pooled.end(), t.begin(), t.end());
    double se = sd(pooled) * std::sqrt(1.0 / 400 + 1.0 / 600);
    REQUIRE(sd(d) == Approx(se).epsilon(0.05));
}

TEST_CASE("permutation preserves group sizes: identical groups give exactly zero") {
    std::vector<double> c(50, 3.0), t(70, 3.0);
    auto d = liftlab::permutation_mean_diff(c.data(), c.size(), t.data(), t.size(), 20, 0, 2);
    for (double x : d) REQUIRE(x == Approx(0.0).margin(1e-12));
}

TEST_CASE("empty groups are rejected") {
    std::vector<double> c, t{1.0};
    REQUIRE_THROWS(liftlab::bootstrap_mean_diff(c.data(), 0, t.data(), 1, 10, 0, 1));
}
