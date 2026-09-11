#pragma once

#include <vector>
#include <string>
#include <cstdint>

// A dense vector of 32-bit floats. Kept as a plain std::vector<float> so the
// rest of the codebase only depends on the C++ standard library.
using Vector = std::vector<float>;

// A single search hit: the original ID the caller supplied at insert time,
// and the similarity score (cosine similarity, higher is better).
struct SearchResult {
    int id;
    float score;
};
