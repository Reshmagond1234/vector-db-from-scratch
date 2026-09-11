#include "exact_index.h"
#include "metrics.h"

#include <algorithm>
#include <stdexcept>

ExactIndex::ExactIndex(std::size_t dimension) : dim_(dimension) {
    if (dim_ == 0) {
        throw std::invalid_argument("ExactIndex - dimension must be > 0");
    }
}

void ExactIndex::insert(int id, const Vector& vector, const std::string& text) {
    if (vector.size() != dim_) {
        throw std::invalid_argument(
            "ExactIndex::insert - vector dimension (" + std::to_string(vector.size()) +
            ") does not match index dimension (" + std::to_string(dim_) + ")");
    }
    if (id_to_slot_.count(id) > 0) {
        throw std::invalid_argument("ExactIndex::insert - duplicate id " + std::to_string(id));
    }

    Vector normalized = vector;
    metrics::normalize_in_place(normalized);

    std::size_t slot = vectors_.size();
    vectors_.push_back(std::move(normalized));
    ids_.push_back(id);
    active_.push_back(true);
    texts_.push_back(text);
    id_to_slot_[id] = slot;
    ++active_count_;
}

bool ExactIndex::remove(int id) {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;  // unknown id
    }
    std::size_t slot = it->second;
    if (!active_[slot]) {
        return false;  // already deleted
    }
    active_[slot] = false;
    --active_count_;
    return true;
}

std::vector<SearchResult> ExactIndex::search(const Vector& query, std::size_t k) const {
    if (query.size() != dim_) {
        throw std::invalid_argument(
            "ExactIndex::search - query dimension (" + std::to_string(query.size()) +
            ") does not match index dimension (" + std::to_string(dim_) + ")");
    }
    if (k == 0 || active_count_ == 0) {
        return {};
    }

    Vector q = query;
    metrics::normalize_in_place(q);

    // Every active vector is compared -- this is the brute-force guarantee.
    std::vector<SearchResult> all;
    all.reserve(active_count_);
    for (std::size_t slot = 0; slot < vectors_.size(); ++slot) {
        if (!active_[slot]) continue;
        // Both q and vectors_[slot] are unit-normalized, so their dot
        // product IS the cosine similarity.
        float score = static_cast<float>(metrics::dot(q, vectors_[slot]));
        all.push_back(SearchResult{ids_[slot], score});
    }

    std::size_t effective_k = std::min(k, all.size());
    std::partial_sort(
        all.begin(), all.begin() + static_cast<long>(effective_k), all.end(),
        [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    all.resize(effective_k);
    return all;
}

std::size_t ExactIndex::size() const { return active_count_; }

std::size_t ExactIndex::total_size() const { return vectors_.size(); }

bool ExactIndex::contains(int id) const {
    auto it = id_to_slot_.find(id);
    return it != id_to_slot_.end() && active_[it->second];
}

const std::string& ExactIndex::get_text(int id) const {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        throw std::invalid_argument("ExactIndex::get_text - unknown id " + std::to_string(id));
    }
    return texts_[it->second];
}

Vector ExactIndex::get_vector(int id) const {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        throw std::invalid_argument("ExactIndex::get_vector - unknown id " + std::to_string(id));
    }
    return vectors_[it->second];
}
