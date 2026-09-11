#pragma once

#include "vector_types.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Optional stats filled in by search(), useful for benchmarking.
struct IVFSearchStats {
    std::size_t clusters_visited = 0;
    // Every inverted-list entry looked at while scanning the visited
    // clusters, ACTIVE OR NOT. Lazily-deleted entries are still walked over
    // (they just don't get scored/returned), so this number is the honest
    // cost of a query, including lazy-deletion overhead.
    std::size_t candidates_examined = 0;
};

// IVFIndex: an Inverted File Index, built completely from scratch on top of
// the from-scratch KMeans in kmeans.h.
//
//   1. build() runs K-means over the initial vector set to produce
//      `num_clusters` centroids, then assigns every vector to the inverted
//      list of its nearest centroid.
//   2. search() finds the `nprobe` centroids nearest the query, visits only
//      those inverted lists, scores each candidate with exact cosine
//      similarity, and returns the top-k. Because only a subset of the
//      clusters is visited, results are approximate: nprobe trades speed
//      for accuracy (nprobe == num_clusters searches every cluster and the
//      result becomes equivalent to brute force).
//
// Deletion is lazy (see remove()): IDs are flagged inactive rather than
// physically removed from the inverted lists. This keeps insert/remove
// O(1)-ish but means a heavily-deleted index does extra (wasted) work
// walking past dead entries -- see IVFSearchStats::candidates_examined and
// the README for the full discussion of this tradeoff.
class IVFIndex {
public:
    IVFIndex(std::size_t dimension, std::size_t num_clusters,
              std::size_t kmeans_max_iterations = 20, std::uint32_t seed = 42);

    // Builds the index from scratch: runs K-means on `vectors` to produce
    // the centroids, then fills the inverted lists. Must be called exactly
    // once, before any insert()/remove()/search() call, with at least
    // num_clusters() vectors (K-means needs at least k points).
    void build(const std::vector<int>& ids, const std::vector<Vector>& vectors,
               const std::vector<std::string>& texts = {});

    // Adds one more vector after the index has been built. The vector is
    // assigned to whichever existing centroid it is nearest to (centroids
    // themselves are NOT recomputed -- that would require a full rebuild).
    // It is searchable immediately. Throws std::invalid_argument if the
    // index has not been built yet, the dimension is wrong, or id is a
    // duplicate.
    void insert(int id, const Vector& vector, const std::string& text = "");

    // Lazily deletes `id`: it is flagged inactive and will never again be
    // returned by search(), but its slot remains physically present in its
    // inverted list. Returns true if id was found and active, false if
    // unknown or already deleted.
    bool remove(int id);

    // Returns the approximate top-k nearest neighbours of `query`, scanning
    // only the `nprobe` inverted lists whose centroids are nearest to the
    // query. Throws std::invalid_argument if the index is not built, the
    // query has the wrong dimension, or nprobe is 0 or > num_clusters().
    std::vector<SearchResult> search(const Vector& query, std::size_t k, std::size_t nprobe,
                                      IVFSearchStats* stats = nullptr) const;

    std::size_t size() const;              // active vector count
    std::size_t total_size() const;        // including lazily-deleted
    std::size_t num_clusters() const { return num_clusters_; }
    std::size_t dimension() const { return dim_; }
    bool is_built() const { return built_; }
    const std::string& get_text(int id) const;

    // Cluster sizes (active + inactive entries), useful for reporting
    // balance in the README/demo.
    std::vector<std::size_t> cluster_sizes() const;

private:
    std::size_t dim_;
    std::size_t num_clusters_;
    std::size_t kmeans_max_iter_;
    std::uint32_t seed_;
    bool built_ = false;

    std::vector<Vector> centroids_;                  // normalized, size == num_clusters_
    std::vector<std::vector<std::size_t>> inverted_lists_;  // cluster -> internal slots

    std::vector<Vector> vectors_;      // normalized, by internal slot
    std::vector<int> ids_;             // by internal slot
    std::vector<bool> active_;         // by internal slot
    std::vector<std::string> texts_;   // by internal slot
    std::unordered_map<int, std::size_t> id_to_slot_;
    std::size_t active_count_ = 0;

    int nearest_centroid(const Vector& v) const;
    std::vector<int> nearest_centroids(const Vector& v, std::size_t nprobe) const;
};
