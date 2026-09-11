// tests/tests.cpp
//
// A small, dependency-free test runner (no GoogleTest / Catch2 -- just
// std::iostream and macros) exercising every requirement called out in the
// assignment. Run the resulting `tests` binary; it prints PASS/FAIL per
// test and exits non-zero if anything failed.

#include "dataset.h"
#include "exact_index.h"
#include "ivf_index.h"
#include "kmeans.h"
#include "metrics.h"

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_tests_run = 0;
int g_tests_failed = 0;
std::string g_current_test;

#define TEST(name) void name()
#define RUN_TEST(name)                                                     \
    do {                                                                   \
        g_current_test = #name;                                           \
        ++g_tests_run;                                                    \
        try {                                                              \
            name();                                                       \
            std::cout << "[PASS] " << #name << "\n";                     \
        } catch (const std::exception& e) {                               \
            ++g_tests_failed;                                             \
            std::cout << "[FAIL] " << #name << " -- exception: " << e.what() << "\n"; \
        } catch (...) {                                                    \
            ++g_tests_failed;                                             \
            std::cout << "[FAIL] " << #name << " -- unknown exception\n"; \
        }                                                                  \
    } while (0)

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            throw std::runtime_error(std::string("CHECK failed: ") + #cond + \
                                      " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                              \
    do {                                                                   \
        double _a = (a), _b = (b), _eps = (eps);                          \
        if (std::fabs(_a - _b) > _eps) {                                  \
            throw std::runtime_error("CHECK_NEAR failed: " + std::to_string(_a) + \
                                      " vs " + std::to_string(_b) + " at " __FILE__ ":" + \
                                      std::to_string(__LINE__));           \
        }                                                                   \
    } while (0)

#define CHECK_THROWS(expr)                                                 \
    do {                                                                   \
        bool _threw = false;                                              \
        try { expr; } catch (...) { _threw = true; }                      \
        if (!_threw) {                                                     \
            throw std::runtime_error("expected exception from: " #expr " at " __FILE__ ":" + \
                                      std::to_string(__LINE__));           \
        }                                                                   \
    } while (0)

Vector make_vec(std::initializer_list<float> vals) { return Vector(vals); }

// ---------------------------------------------------------------------
// 1. Cosine similarity correctness (hand-computed).
// ---------------------------------------------------------------------
TEST(cosine_similarity_hand_computed) {
    Vector a = make_vec({1.0f, 0.0f});
    Vector b = make_vec({0.0f, 1.0f});
    CHECK_NEAR(metrics::cosine_similarity(a, b), 0.0, 1e-6);

    Vector c = make_vec({1.0f, 0.0f});
    Vector d = make_vec({1.0f, 0.0f});
    CHECK_NEAR(metrics::cosine_similarity(c, d), 1.0, 1e-6);

    Vector e = make_vec({1.0f, 0.0f});
    Vector f = make_vec({-1.0f, 0.0f});
    CHECK_NEAR(metrics::cosine_similarity(e, f), -1.0, 1e-6);

    // 3-4-5 style vector, hand computed: a=(3,4), b=(4,3)
    // dot = 12+12=24, |a|=5, |b|=5 => cos = 24/25 = 0.96
    Vector g = make_vec({3.0f, 4.0f});
    Vector h = make_vec({4.0f, 3.0f});
    CHECK_NEAR(metrics::cosine_similarity(g, h), 0.96, 1e-6);
}

TEST(cosine_similarity_dimension_mismatch_throws) {
    Vector a = make_vec({1.0f, 0.0f});
    Vector b = make_vec({1.0f, 0.0f, 0.0f});
    CHECK_THROWS(metrics::dot(a, b));
}

// ---------------------------------------------------------------------
// 2. Exact search against a tiny, manually-verifiable dataset.
// ---------------------------------------------------------------------
TEST(exact_search_tiny_hand_verified_dataset) {
    ExactIndex idx(2);
    idx.insert(0, make_vec({1.0f, 0.0f}), "east");
    idx.insert(1, make_vec({0.0f, 1.0f}), "north");
    idx.insert(2, make_vec({-1.0f, 0.0f}), "west");
    idx.insert(3, make_vec({0.7071f, 0.7071f}), "northeast");

    // Query pointing east should rank: east (1.0) > northeast (~0.707) >
    // north (0.0) > west (-1.0).
    auto results = idx.search(make_vec({1.0f, 0.0f}), 4);
    CHECK(results.size() == 4);
    CHECK(results[0].id == 0);
    CHECK(results[1].id == 3);
    CHECK(results[2].id == 1);
    CHECK(results[3].id == 2);
    CHECK_NEAR(results[0].score, 1.0, 1e-4);
    CHECK_NEAR(results[3].score, -1.0, 1e-4);
}

// ---------------------------------------------------------------------
// 3. Exact index: basic correctness / edge cases.
// ---------------------------------------------------------------------
TEST(exact_index_empty_search_returns_empty) {
    ExactIndex idx(4);
    auto results = idx.search(make_vec({1, 0, 0, 0}), 5);
    CHECK(results.empty());
}

TEST(exact_index_k_larger_than_available_returns_all) {
    ExactIndex idx(2);
    idx.insert(0, make_vec({1, 0}));
    idx.insert(1, make_vec({0, 1}));
    auto results = idx.search(make_vec({1, 0}), 100);
    CHECK(results.size() == 2);
}

TEST(exact_index_duplicate_id_throws) {
    ExactIndex idx(2);
    idx.insert(0, make_vec({1, 0}));
    CHECK_THROWS(idx.insert(0, make_vec({0, 1})));
}

TEST(exact_index_invalid_dimension_throws) {
    ExactIndex idx(3);
    CHECK_THROWS(idx.insert(0, make_vec({1, 0})));  // wrong length
    CHECK_THROWS(idx.search(make_vec({1, 0}), 1));  // wrong length query
}

TEST(exact_index_zero_dimension_constructor_throws) {
    CHECK_THROWS(ExactIndex bad(0));
}

TEST(exact_index_delete_then_search_never_returns_it) {
    ExactIndex idx(2);
    idx.insert(0, make_vec({1, 0}), "a");
    idx.insert(1, make_vec({0.99f, 0.01f}), "b");
    CHECK(idx.remove(0) == true);
    CHECK(idx.remove(0) == false);   // already deleted
    CHECK(idx.remove(999) == false); // unknown id

    auto results = idx.search(make_vec({1, 0}), 5);
    for (const auto& r : results) {
        CHECK(r.id != 0);
    }
    CHECK(idx.size() == 1);
    CHECK(idx.total_size() == 2);
}

// ---------------------------------------------------------------------
// 4. K-means.
// ---------------------------------------------------------------------
TEST(kmeans_recovers_well_separated_clusters) {
    // Three tight, well-separated 2D blobs; k-means with k=3 should recover
    // them (every point lands with its own blob-mates).
    std::vector<Vector> data;
    std::vector<Vector> centers = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}};
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::vector<int> true_cluster;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 20; ++i) {
            data.push_back(make_vec({centers[c][0] + noise(rng), centers[c][1] + noise(rng)}));
            true_cluster.push_back(c);
        }
    }

    KMeans km(3, 2, 20, 42);
    auto assignments = km.fit(data);
    CHECK(km.centroids().size() == 3);

    // Points with the same true cluster must share the same assignment
    // (label identity can differ from true_cluster, but grouping must
    // match).
    for (int c = 0; c < 3; ++c) {
        int first_idx = -1;
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (true_cluster[i] != c) continue;
            if (first_idx == -1) {
                first_idx = assignments[i];
            } else {
                CHECK(assignments[i] == first_idx);
            }
        }
    }
}

TEST(kmeans_rejects_fewer_points_than_k) {
    std::vector<Vector> data = {make_vec({0, 0}), make_vec({1, 1})};
    KMeans km(5, 2, 10, 1);
    CHECK_THROWS(km.fit(data));
}

TEST(kmeans_no_empty_clusters_on_convergence) {
    std::vector<Vector> data;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (int i = 0; i < 200; ++i) data.push_back(make_vec({d(rng), d(rng)}));

    KMeans km(10, 2, 30, 42);
    auto assignments = km.fit(data);
    std::vector<int> counts(10, 0);
    for (int a : assignments) counts[a]++;
    for (int c : counts) CHECK(c > 0);
}

// ---------------------------------------------------------------------
// 5. IVF construction / search / insert / delete.
// ---------------------------------------------------------------------
std::vector<Vector> synthetic_ivf_data(std::size_t n, std::size_t dim, std::size_t clusters,
                                        std::uint32_t seed, std::vector<Vector>* out_queries,
                                        std::size_t num_queries) {
    auto ds = dataset::generate_clustered_dataset(n, num_queries, dim, clusters, seed, 0.1f);
    if (out_queries) *out_queries = ds.query_vectors;
    return ds.database_vectors;
}

TEST(ivf_build_and_search_basic) {
    std::vector<Vector> queries;
    auto data = synthetic_ivf_data(2000, 16, 10, 99, &queries, 20);
    std::vector<int> ids(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) ids[i] = static_cast<int>(i);

    IVFIndex ivf(16, 10, 20, 99);
    CHECK(ivf.is_built() == false);
    ivf.build(ids, data);
    CHECK(ivf.is_built() == true);
    CHECK(ivf.size() == data.size());

    auto results = ivf.search(queries[0], 5, 10);
    CHECK(results.size() == 5);
}

TEST(ivf_search_before_build_throws) {
    IVFIndex ivf(8, 4, 10, 1);
    CHECK_THROWS(ivf.search(Vector(8, 1.0f), 3, 2));
}

TEST(ivf_invalid_nprobe_throws) {
    std::vector<Vector> queries;
    auto data = synthetic_ivf_data(500, 8, 5, 3, &queries, 5);
    std::vector<int> ids(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) ids[i] = static_cast<int>(i);
    IVFIndex ivf(8, 5, 10, 3);
    ivf.build(ids, data);

    CHECK_THROWS(ivf.search(queries[0], 3, 0));   // nprobe = 0 invalid
    CHECK_THROWS(ivf.search(queries[0], 3, 6));   // nprobe > num_clusters invalid
    // nprobe == num_clusters is valid (searches everything).
    auto results = ivf.search(queries[0], 3, 5);
    CHECK(results.size() == 3);
}

TEST(ivf_insert_makes_vector_immediately_searchable) {
    auto data = synthetic_ivf_data(500, 8, 5, 11, nullptr, 0);
    std::vector<int> ids(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) ids[i] = static_cast<int>(i);
    IVFIndex ivf(8, 5, 10, 11);
    ivf.build(ids, data);

    Vector new_vec(8, 0.0f);
    new_vec[0] = 1.0f;
    int new_id = 999999;
    ivf.insert(new_id, new_vec, "brand new vector");
    CHECK(ivf.size() == data.size() + 1);

    // Searching for something very close to new_vec at full nprobe should
    // find it.
    auto results = ivf.search(new_vec, 1, 5);
    CHECK(results.size() == 1);
    CHECK(results[0].id == new_id);
}

TEST(ivf_delete_prevents_result_forever) {
    auto data = synthetic_ivf_data(500, 8, 5, 21, nullptr, 0);
    std::vector<int> ids(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) ids[i] = static_cast<int>(i);
    IVFIndex ivf(8, 5, 10, 21);
    ivf.build(ids, data);

    int victim = ids[0];
    Vector victim_vec = data[0];
    CHECK(ivf.remove(victim) == true);
    CHECK(ivf.remove(victim) == false);  // already gone

    auto results = ivf.search(victim_vec, static_cast<std::size_t>(data.size()), 5);
    for (const auto& r : results) {
        CHECK(r.id != victim);
    }
}

TEST(ivf_duplicate_id_on_build_throws) {
    std::vector<Vector> data = {Vector(4, 1.0f), Vector(4, 2.0f), Vector(4, 3.0f), Vector(4, 4.0f)};
    std::vector<int> ids = {0, 1, 1, 2};  // duplicate id "1"
    IVFIndex ivf(4, 2, 10, 5);
    CHECK_THROWS(ivf.build(ids, data));
}

TEST(ivf_nprobe_equals_num_clusters_approaches_exact) {
    // With nprobe == num_clusters, IVF must scan every cluster, so its
    // top-k must exactly match brute force top-k (same candidate set).
    std::vector<Vector> queries;
    auto data = synthetic_ivf_data(3000, 16, 20, 77, &queries, 30);

    ExactIndex exact(16);
    std::vector<int> ids(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        ids[i] = static_cast<int>(i);
        exact.insert(ids[i], data[i]);
    }

    IVFIndex ivf(16, 20, 20, 77);
    ivf.build(ids, data);

    double total_recall = 0.0;
    for (const auto& q : queries) {
        auto exact_results = exact.search(q, 10);
        std::vector<int> exact_ids;
        for (auto& r : exact_results) exact_ids.push_back(r.id);

        auto ivf_results = ivf.search(q, 10, 20);  // nprobe == num_clusters
        std::vector<int> ivf_ids;
        for (auto& r : ivf_results) ivf_ids.push_back(r.id);

        total_recall += metrics::recall_at_k(exact_ids, ivf_ids, 10);
    }
    double avg_recall = total_recall / static_cast<double>(queries.size());
    CHECK(avg_recall > 0.999);  // should be (numerically) exact
}

// ---------------------------------------------------------------------
// 6. Recall calculation.
// ---------------------------------------------------------------------
TEST(recall_at_k_basic_cases) {
    std::vector<int> exact = {1, 2, 3, 4, 5};
    std::vector<int> perfect = {1, 2, 3, 4, 5};
    CHECK_NEAR(metrics::recall_at_k(exact, perfect, 5), 1.0, 1e-9);

    std::vector<int> half = {1, 2, 99, 98, 97};
    CHECK_NEAR(metrics::recall_at_k(exact, half, 5), 0.4, 1e-9);

    std::vector<int> none = {10, 11, 12, 13, 14};
    CHECK_NEAR(metrics::recall_at_k(exact, none, 5), 0.0, 1e-9);

    std::vector<int> empty_exact = {};
    CHECK_NEAR(metrics::recall_at_k(empty_exact, half, 5), 1.0, 1e-9);
}

// ---------------------------------------------------------------------
// 7. Dataset helpers: round-trip serialization + text embedding sanity.
// ---------------------------------------------------------------------
TEST(vector_binary_round_trip) {
    std::vector<Vector> vecs = {make_vec({1, 2, 3}), make_vec({4, 5, 6}), make_vec({-1, 0, 1})};
    dataset::save_vectors_binary("vdb_test_vectors.bin", vecs);
    auto loaded = dataset::load_vectors_binary("vdb_test_vectors.bin");
    CHECK(loaded.size() == vecs.size());
    for (std::size_t i = 0; i < vecs.size(); ++i) {
        CHECK(loaded[i].size() == vecs[i].size());
        for (std::size_t d = 0; d < vecs[i].size(); ++d) {
            CHECK_NEAR(loaded[i][d], vecs[i][d], 1e-6);
        }
    }
}

TEST(embed_text_shared_vocabulary_scores_higher) {
    Vector a = dataset::embed_text("how can I reset my password", 64);
    Vector b = dataset::embed_text("how do I reset my password please", 64);
    Vector c = dataset::embed_text("what temperature should I fry chicken", 64);

    double sim_ab = metrics::cosine_similarity(a, b);
    double sim_ac = metrics::cosine_similarity(a, c);
    CHECK(sim_ab > sim_ac);
}

TEST(embed_text_is_deterministic) {
    Vector a = dataset::embed_text("deterministic please", 32);
    Vector b = dataset::embed_text("deterministic please", 32);
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-9);
}

}  // namespace

int main() {
    std::cout << "Running Vector Database From Scratch test suite...\n\n";

    RUN_TEST(cosine_similarity_hand_computed);
    RUN_TEST(cosine_similarity_dimension_mismatch_throws);
    RUN_TEST(exact_search_tiny_hand_verified_dataset);
    RUN_TEST(exact_index_empty_search_returns_empty);
    RUN_TEST(exact_index_k_larger_than_available_returns_all);
    RUN_TEST(exact_index_duplicate_id_throws);
    RUN_TEST(exact_index_invalid_dimension_throws);
    RUN_TEST(exact_index_zero_dimension_constructor_throws);
    RUN_TEST(exact_index_delete_then_search_never_returns_it);
    RUN_TEST(kmeans_recovers_well_separated_clusters);
    RUN_TEST(kmeans_rejects_fewer_points_than_k);
    RUN_TEST(kmeans_no_empty_clusters_on_convergence);
    RUN_TEST(ivf_build_and_search_basic);
    RUN_TEST(ivf_search_before_build_throws);
    RUN_TEST(ivf_invalid_nprobe_throws);
    RUN_TEST(ivf_insert_makes_vector_immediately_searchable);
    RUN_TEST(ivf_delete_prevents_result_forever);
    RUN_TEST(ivf_duplicate_id_on_build_throws);
    RUN_TEST(ivf_nprobe_equals_num_clusters_approaches_exact);
    RUN_TEST(recall_at_k_basic_cases);
    RUN_TEST(vector_binary_round_trip);
    RUN_TEST(embed_text_shared_vocabulary_scores_higher);
    RUN_TEST(embed_text_is_deterministic);

    std::cout << "\n" << g_tests_run << " tests run, " << (g_tests_run - g_tests_failed)
               << " passed, " << g_tests_failed << " failed.\n";

    return g_tests_failed == 0 ? 0 : 1;
}
