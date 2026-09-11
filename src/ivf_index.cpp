#include "ivf_index.h"
#include "kmeans.h"
#include "metrics.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

IVFIndex::IVFIndex(std::size_t dimension, std::size_t num_clusters,
                    std::size_t kmeans_max_iterations, std::uint32_t seed)
    : dim_(dimension),
      num_clusters_(num_clusters),
      kmeans_max_iter_(kmeans_max_iterations),
      seed_(seed) {
    if (dim_ == 0) {
        throw std::invalid_argument("IVFIndex - dimension must be > 0");
    }
    if (num_clusters_ == 0) {
        throw std::invalid_argument("IVFIndex - num_clusters must be > 0");
    }
}

void IVFIndex::build(const std::vector<int>& ids, const std::vector<Vector>& vectors,
                      const std::vector<std::string>& texts) {
    if (built_) {
        throw std::logic_error("IVFIndex::build - index has already been built");
    }
    if (ids.size() != vectors.size()) {
        throw std::invalid_argument("IVFIndex::build - ids.size() must equal vectors.size()");
    }
    if (!texts.empty() && texts.size() != vectors.size()) {
        throw std::invalid_argument("IVFIndex::build - texts.size() must equal vectors.size()");
    }
    if (vectors.size() < num_clusters_) {
        throw std::invalid_argument(
            "IVFIndex::build - need at least num_clusters vectors to build (num_clusters=" +
            std::to_string(num_clusters_) + ", got " + std::to_string(vectors.size()) + ")");
    }

    // Normalize everything up front so cosine similarity at query time is a
    // plain dot product.
    std::vector<Vector> normalized(vectors.size());
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != dim_) {
            throw std::invalid_argument("IVFIndex::build - vector has wrong dimension");
        }
        normalized[i] = vectors[i];
        metrics::normalize_in_place(normalized[i]);
    }

    // K-means from scratch (src/kmeans.cpp) determines the centroids and,
    // as a side effect, the initial cluster assignment for every vector.
    KMeans kmeans(num_clusters_, dim_, kmeans_max_iter_, seed_);
    std::vector<int> assignments = kmeans.fit(normalized);
    centroids_ = kmeans.centroids();
    for (auto& c : centroids_) {
        metrics::normalize_in_place(c);
    }

    inverted_lists_.assign(num_clusters_, {});
    vectors_.clear();
    ids_.clear();
    active_.clear();
    texts_.clear();
    id_to_slot_.clear();
    active_count_ = 0;

    vectors_.reserve(vectors.size());
    ids_.reserve(vectors.size());
    active_.reserve(vectors.size());
    texts_.reserve(vectors.size());

    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (id_to_slot_.count(ids[i]) > 0) {
            throw std::invalid_argument("IVFIndex::build - duplicate id " +
                                         std::to_string(ids[i]));
        }
        std::size_t slot = vectors_.size();
        vectors_.push_back(std::move(normalized[i]));
        ids_.push_back(ids[i]);
        active_.push_back(true);
        texts_.push_back(texts.empty() ? std::string() : texts[i]);
        id_to_slot_[ids[i]] = slot;

        int cluster = assignments[i];
        inverted_lists_[static_cast<std::size_t>(cluster)].push_back(slot);
        ++active_count_;
    }

    built_ = true;
}

int IVFIndex::nearest_centroid(const Vector& v) const {
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

std::vector<int> IVFIndex::nearest_centroids(const Vector& v, std::size_t nprobe) const {
    std::vector<std::pair<double, int>> dists;
    dists.reserve(centroids_.size());
    for (std::size_t c = 0; c < centroids_.size(); ++c) {
        dists.emplace_back(metrics::euclidean_distance_sq(v, centroids_[c]), static_cast<int>(c));
    }
    std::size_t effective = std::min(nprobe, dists.size());
    std::partial_sort(dists.begin(), dists.begin() + static_cast<long>(effective), dists.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> result;
    result.reserve(effective);
    for (std::size_t i = 0; i < effective; ++i) {
        result.push_back(dists[i].second);
    }
    return result;
}

void IVFIndex::insert(int id, const Vector& vector, const std::string& text) {
    if (!built_) {
        throw std::logic_error("IVFIndex::insert - build() must be called before insert()");
    }
    if (vector.size() != dim_) {
        throw std::invalid_argument("IVFIndex::insert - vector has wrong dimension");
    }
    if (id_to_slot_.count(id) > 0) {
        throw std::invalid_argument("IVFIndex::insert - duplicate id " + std::to_string(id));
    }

    Vector normalized = vector;
    metrics::normalize_in_place(normalized);

    int cluster = nearest_centroid(normalized);

    std::size_t slot = vectors_.size();
    vectors_.push_back(std::move(normalized));
    ids_.push_back(id);
    active_.push_back(true);
    texts_.push_back(text);
    id_to_slot_[id] = slot;

    inverted_lists_[static_cast<std::size_t>(cluster)].push_back(slot);
    ++active_count_;
}

bool IVFIndex::remove(int id) {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;
    }
    std::size_t slot = it->second;
    if (!active_[slot]) {
        return false;
    }
    active_[slot] = false;  // lazy delete: stays in its inverted list, just flagged inactive
    --active_count_;
    return true;
}

std::vector<SearchResult> IVFIndex::search(const Vector& query, std::size_t k,
                                            std::size_t nprobe, IVFSearchStats* stats) const {
    if (!built_) {
        throw std::logic_error("IVFIndex::search - index has not been built yet");
    }
    if (query.size() != dim_) {
        throw std::invalid_argument("IVFIndex::search - query has wrong dimension");
    }
    if (nprobe == 0 || nprobe > num_clusters_) {
        throw std::invalid_argument(
            "IVFIndex::search - nprobe must be in [1, num_clusters] (num_clusters=" +
            std::to_string(num_clusters_) + ", got " + std::to_string(nprobe) + ")");
    }
    if (k == 0 || active_count_ == 0) {
        return {};
    }

    Vector q = query;
    metrics::normalize_in_place(q);

    std::vector<int> probe_clusters = nearest_centroids(q, nprobe);

    std::vector<SearchResult> candidates;
    std::size_t examined = 0;
    for (int c : probe_clusters) {
        for (std::size_t slot : inverted_lists_[static_cast<std::size_t>(c)]) {
            ++examined;  // counted whether active or not: the real cost of lazy deletion
            if (!active_[slot]) continue;
            float score = static_cast<float>(metrics::dot(q, vectors_[slot]));
            candidates.push_back(SearchResult{ids_[slot], score});
        }
    }

    if (stats != nullptr) {
        stats->clusters_visited = probe_clusters.size();
        stats->candidates_examined = examined;
    }

    std::size_t effective_k = std::min(k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<long>(effective_k),
                       candidates.end(),
                       [](const SearchResult& a, const SearchResult& b) {
                           return a.score > b.score;
                       });
    candidates.resize(effective_k);
    return candidates;
}

std::size_t IVFIndex::size() const { return active_count_; }

std::size_t IVFIndex::total_size() const { return vectors_.size(); }

const std::string& IVFIndex::get_text(int id) const {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        throw std::invalid_argument("IVFIndex::get_text - unknown id " + std::to_string(id));
    }
    return texts_[it->second];
}

std::vector<std::size_t> IVFIndex::cluster_sizes() const {
    std::vector<std::size_t> sizes;
    sizes.reserve(inverted_lists_.size());
    for (const auto& list : inverted_lists_) {
        sizes.push_back(list.size());
    }
    return sizes;
}
