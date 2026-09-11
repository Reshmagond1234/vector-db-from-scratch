#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace metrics {

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "metrics::dot - vectors must have the same dimension");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double l2_norm(const Vector& a) {
    double sum_sq = 0.0;
    for (float x : a) {
        sum_sq += static_cast<double>(x) * static_cast<double>(x);
    }
    return std::sqrt(sum_sq);
}

double euclidean_distance_sq(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "metrics::euclidean_distance_sq - vectors must have the same dimension");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

double cosine_similarity(const Vector& a, const Vector& b) {
    double na = l2_norm(a);
    double nb = l2_norm(b);
    if (na <= 1e-12 || nb <= 1e-12) {
        return 0.0;
    }
    return dot(a, b) / (na * nb);
}

void normalize_in_place(Vector& v) {
    double n = l2_norm(v);
    if (n <= 1e-12) {
        return;  // zero (or near-zero) vector: no well-defined direction
    }
    float inv = static_cast<float>(1.0 / n);
    for (float& x : v) {
        x *= inv;
    }
}

double recall_at_k(const std::vector<int>& exact_ids,
                    const std::vector<int>& approx_ids,
                    std::size_t k) {
    if (exact_ids.empty()) {
        return 1.0;
    }
    std::size_t effective_k = std::min(k, exact_ids.size());
    std::unordered_set<int> exact_set(exact_ids.begin(),
                                       exact_ids.begin() + static_cast<long>(effective_k));

    std::size_t limit = std::min(k, approx_ids.size());
    std::size_t hits = 0;
    for (std::size_t i = 0; i < limit; ++i) {
        if (exact_set.count(approx_ids[i]) > 0) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(effective_k);
}

}  // namespace metrics
