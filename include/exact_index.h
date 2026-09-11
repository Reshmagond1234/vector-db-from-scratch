#pragma once

#include "vector_types.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// ExactIndex is the ground-truth brute-force nearest-neighbour index.
//
// Every insert() call stores the vector (normalized to unit length so that
// cosine similarity reduces to a dot product at query time). Every search()
// call compares the query against *every currently active* stored vector --
// there is no shortcut, no clustering, no pruning. This is what makes it
// suitable as ground truth: it cannot silently miss a true nearest
// neighbour the way an approximate index can.
//
// Deletion is lazy: remove() flips an "active" flag. The slot's memory is
// reused for nothing else, but the vector is skipped by every future
// search(). This keeps IDs stable and remove() O(1).
class ExactIndex {
public:
    explicit ExactIndex(std::size_t dimension);

    // Inserts a new vector under `id`. `text` is optional metadata (e.g. the
    // source sentence) purely for display purposes.
    // Throws std::invalid_argument if:
    //   - vector.size() != dimension()
    //   - id already exists (active or previously deleted)
    void insert(int id, const Vector& vector, const std::string& text = "");

    // Marks `id` inactive so it can never again be returned by search().
    // Returns true if `id` was found and was active, false if unknown or
    // already deleted.
    bool remove(int id);

    // Returns the top-k active vectors by cosine similarity to `query`,
    // sorted by descending score. Checks EVERY active vector -- O(N * D).
    // If k > number of active vectors, returns all active vectors.
    // If the index has zero active vectors, returns an empty vector.
    // Throws std::invalid_argument if query.size() != dimension().
    std::vector<SearchResult> search(const Vector& query, std::size_t k) const;

    // Number of currently active (non-deleted) vectors.
    std::size_t size() const;

    // Number of vectors ever inserted, including deleted ones.
    std::size_t total_size() const;

    std::size_t dimension() const { return dim_; }

    bool contains(int id) const;
    const std::string& get_text(int id) const;
    Vector get_vector(int id) const;  // returns the *normalized* stored vector

private:
    std::size_t dim_;

    std::vector<Vector> vectors_;        // normalized vectors, by internal slot
    std::vector<int> ids_;               // external id, by internal slot
    std::vector<bool> active_;           // active flag, by internal slot
    std::vector<std::string> texts_;     // optional metadata, by internal slot
    std::unordered_map<int, std::size_t> id_to_slot_;
    std::size_t active_count_ = 0;
};
