#include "kmeans.h"
#include "metrics.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

KMeans::KMeans(std::size_t k, std::size_t dimension, std::size_t max_iterations,
               std::uint32_t seed)
    : k_(k), dim_(dimension), max_iterations_(max_iterations), seed_(seed) {
    if (k_ == 0) {
        throw std::invalid_argument("KMeans - k must be > 0");
    }
    if (dim_ == 0) {
        throw std::invalid_argument("KMeans - dimension must be > 0");
    }
}

void KMeans::initialize_centroids(const std::vector<Vector>& data) {
    std::mt19937 rng(seed_);
    std::vector<std::size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    centroids_.clear();
    centroids_.reserve(k_);
    for (std::size_t i = 0; i < k_; ++i) {
        centroids_.push_back(data[indices[i]]);
    }
}

int KMeans::nearest_centroid_index(const Vector& v) const {
    int best_idx = -1;
    double best_dist = std::numeric_limits<double>::max();
    for (std::size_t c = 0; c < centroids_.size(); ++c) {
        double d = metrics::euclidean_distance_sq(v, centroids_[c]);
        if (d < best_dist) {
            best_dist = d;
            best_idx = static_cast<int>(c);
        }
    }
    return best_idx;
}

std::vector<int> KMeans::fit(const std::vector<Vector>& data) {
    if (data.empty()) {
        throw std::invalid_argument("KMeans::fit - data must not be empty");
    }
    if (data.size() < k_) {
        throw std::invalid_argument(
            "KMeans::fit - need at least k data points (k=" + std::to_string(k_) +
            ", got " + std::to_string(data.size()) + ")");
    }
    for (const auto& v : data) {
        if (v.size() != dim_) {
            throw std::invalid_argument("KMeans::fit - data vector has wrong dimension");
        }
    }

    initialize_centroids(data);

    std::vector<int> assignments(data.size(), -1);
    std::mt19937 reseed_rng(seed_ + 1);  // separate stream, used only for empty-cluster repair

    for (std::size_t iter = 0; iter < max_iterations_; ++iter) {
        bool changed = false;

        // --- Assignment step ---
        for (std::size_t i = 0; i < data.size(); ++i) {
            int nearest = nearest_centroid_index(data[i]);
            if (nearest != assignments[i]) {
                assignments[i] = nearest;
                changed = true;
            }
        }

        // --- Update step ---
        std::vector<Vector> sums(k_, Vector(dim_, 0.0f));
        std::vector<std::size_t> counts(k_, 0);
        for (std::size_t i = 0; i < data.size(); ++i) {
            int c = assignments[i];
            counts[c] += 1;
            for (std::size_t d = 0; d < dim_; ++d) {
                sums[c][d] += data[i][d];
            }
        }

        for (std::size_t c = 0; c < k_; ++c) {
            if (counts[c] == 0) {
                // Empty cluster: reseed it to a random data point so every
                // cluster stays non-empty (a "dead" centroid would never
                // recover on its own).
                std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);
                centroids_[c] = data[dist(reseed_rng)];
                changed = true;
            } else {
                for (std::size_t d = 0; d < dim_; ++d) {
                    centroids_[c][d] = sums[c][d] / static_cast<float>(counts[c]);
                }
            }
        }

        if (!changed) {
            break;  // converged: no assignment changed and no cluster was empty
        }
    }

    // Final assignment pass so the returned assignments match the final
    // centroids exactly (the loop above may have updated centroids after
    // the last assignment step).
    for (std::size_t i = 0; i < data.size(); ++i) {
        assignments[i] = nearest_centroid_index(data[i]);
    }

    return assignments;
}
