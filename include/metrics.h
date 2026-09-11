#pragma once

#include "vector_types.h"
#include <vector>
#include <cstddef>

// All distance / similarity math lives here. Nothing in this file uses any
// external numerical library -- it is plain loops over std::vector<float>.
namespace metrics {

// Dot product of two equal-length vectors. Throws std::invalid_argument on
// a dimension mismatch.
double dot(const Vector& a, const Vector& b);

// L2 (Euclidean) norm of a vector.
double l2_norm(const Vector& a);

// Squared Euclidean distance between two vectors (avoids an unnecessary
// sqrt when only used for nearest-centroid comparisons).
double euclidean_distance_sq(const Vector& a, const Vector& b);

// Cosine similarity between two arbitrary (not necessarily unit-length)
// vectors: dot(a,b) / (||a|| * ||b||). Returns 0.0 if either vector has
// zero norm (defined that way to avoid division by zero on degenerate
// input rather than throwing, since an all-zero vector is a valid, if
// unusual, input).
double cosine_similarity(const Vector& a, const Vector& b);

// Normalizes a vector to unit L2 norm in place. If the vector has (near)
// zero norm it is left unchanged (there is no meaningful direction).
void normalize_in_place(Vector& v);

// Recall@K between an exact (ground truth) ID list and an approximate ID
// list, both already restricted/sorted to their top-K.
//   recall = |approx_ids ∩ exact_ids| / min(k, exact_ids.size())
// Returns 1.0 if exact_ids is empty (nothing to recall).
double recall_at_k(const std::vector<int>& exact_ids,
                    const std::vector<int>& approx_ids,
                    std::size_t k);

}  // namespace metrics
