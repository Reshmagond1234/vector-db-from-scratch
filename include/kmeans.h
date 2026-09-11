#pragma once

#include "vector_types.h"
#include <cstdint>
#include <vector>

// A from-scratch Lloyd's-algorithm K-means, used only to build the
// centroids for the IVF index. Distances are Euclidean, as specified.
//
// Initialization: deterministic seeded random selection of k distinct data
// points as the starting centroids (classic "Forgy" initialization). Using
// a fixed std::mt19937 seed makes every run reproducible.
//
// Each iteration:
//   1. Assignment step: assign every point to its nearest centroid
//      (O(N * K * D)).
//   2. Update step: recompute each centroid as the mean of the points
//      assigned to it (O(N * D)).
// Stops when assignments stop changing or max_iterations is reached.
//
// If a cluster ends up empty after an update step, it is reseeded to the
// data point currently farthest from its own centroid, so no cluster is
// silently dropped.
class KMeans {
public:
    KMeans(std::size_t k, std::size_t dimension, std::size_t max_iterations = 20,
           std::uint32_t seed = 42);

    // Runs Lloyd's algorithm on `data` and returns, for each input vector,
    // the index of its assigned centroid (0..k-1). Also leaves the final
    // centroids available via centroids().
    // Throws std::invalid_argument if data is empty, any vector has the
    // wrong dimension, or data.size() < k (can't form k non-empty clusters
    // from fewer than k points).
    std::vector<int> fit(const std::vector<Vector>& data);

    const std::vector<Vector>& centroids() const { return centroids_; }
    std::size_t k() const { return k_; }
    std::size_t dimension() const { return dim_; }

private:
    std::size_t k_;
    std::size_t dim_;
    std::size_t max_iterations_;
    std::uint32_t seed_;
    std::vector<Vector> centroids_;

    void initialize_centroids(const std::vector<Vector>& data);
    int nearest_centroid_index(const Vector& v) const;
};
