// benchmark.cpp
//
// Builds the ExactIndex (ground truth) and IVFIndex over the 50,000-vector
// synthetic dataset, runs the same 500 queries through IVF at several
// nprobe settings, and reports measured Recall@10, QPS, and latency for
// each. Writes results/benchmark.csv. No number in this file is hard-coded
// -- every value in the CSV comes from an actual timed run against actual
// computed ground truth.

#include "dataset.h"
#include "exact_index.h"
#include "ivf_index.h"
#include "metrics.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
constexpr std::size_t kDimension = 128;
constexpr std::size_t kNumClusters = 100;
constexpr std::size_t kK = 10;
constexpr std::uint32_t kSeed = 42;

std::vector<Vector> load_or_generate_database(std::vector<Vector>& queries_out) {
    const std::string db_path = "data/database_vectors.bin";
    const std::string q_path = "data/query_vectors.bin";

    if (std::filesystem::exists(db_path) && std::filesystem::exists(q_path)) {
        std::cout << "Loading existing dataset from data/...\n";
        auto db = dataset::load_vectors_binary(db_path);
        queries_out = dataset::load_vectors_binary(q_path);
        return db;
    }

    std::cout << "No dataset found on disk -- generating it now (seed=" << kSeed << ")...\n";
    auto ds = dataset::generate_clustered_dataset(50000, 500, kDimension, kNumClusters, kSeed,
                                                    /*cluster_std=*/1.0f);
    std::filesystem::create_directories("data");
    dataset::save_vectors_binary(db_path, ds.database_vectors);
    dataset::save_vectors_binary(q_path, ds.query_vectors);
    queries_out = ds.query_vectors;
    return ds.database_vectors;
}
}  // namespace

int main() {
    std::vector<Vector> queries;
    std::vector<Vector> database = load_or_generate_database(queries);

    std::cout << "database vectors: " << database.size() << "\n";
    std::cout << "query vectors   : " << queries.size() << "\n";
    std::cout << "dimension       : " << (database.empty() ? 0 : database[0].size()) << "\n\n";

    // ------------------------------------------------------------------
    // Build the exact (ground truth) index.
    // ------------------------------------------------------------------
    std::cout << "Building ExactIndex (ground truth, brute force)...\n";
    ExactIndex exact(kDimension);
    for (std::size_t i = 0; i < database.size(); ++i) {
        exact.insert(static_cast<int>(i), database[i]);
    }
    std::cout << "ExactIndex built with " << exact.size() << " active vectors.\n\n";

    // ------------------------------------------------------------------
    // Compute ground truth top-10 for all 500 queries, timing it too (this
    // time is NOT part of the IVF benchmark, just reported for interest).
    // ------------------------------------------------------------------
    std::cout << "Computing exact ground truth for " << queries.size() << " queries (k=" << kK
               << ")... this is brute force, so it is the slow part.\n";
    std::vector<std::vector<int>> ground_truth(queries.size());
    auto gt_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        auto results = exact.search(queries[i], kK);
        ground_truth[i].reserve(results.size());
        for (const auto& r : results) ground_truth[i].push_back(r.id);
    }
    auto gt_end = std::chrono::steady_clock::now();
    double gt_seconds = std::chrono::duration<double>(gt_end - gt_start).count();
    std::cout << "Ground truth computed in " << std::fixed << std::setprecision(2) << gt_seconds
               << "s (" << queries.size() / gt_seconds << " QPS brute force).\n\n";

    // ------------------------------------------------------------------
    // Build the IVF index over the same data.
    // ------------------------------------------------------------------
    std::cout << "Building IVFIndex (K-means with " << kNumClusters << " clusters)...\n";
    std::vector<int> ids(database.size());
    for (std::size_t i = 0; i < database.size(); ++i) ids[i] = static_cast<int>(i);

    IVFIndex ivf(kDimension, kNumClusters, /*kmeans_max_iterations=*/20, kSeed);
    auto build_start = std::chrono::steady_clock::now();
    ivf.build(ids, database);
    auto build_end = std::chrono::steady_clock::now();
    std::cout << "IVFIndex built in "
               << std::chrono::duration<double>(build_end - build_start).count() << "s.\n";

    auto sizes = ivf.cluster_sizes();
    std::size_t min_size = *std::min_element(sizes.begin(), sizes.end());
    std::size_t max_size = *std::max_element(sizes.begin(), sizes.end());
    double avg_size = static_cast<double>(database.size()) / sizes.size();
    std::cout << "Cluster sizes -- min: " << min_size << ", max: " << max_size
               << ", avg: " << avg_size << "\n\n";

    // ------------------------------------------------------------------
    // Benchmark at multiple nprobe settings. Index construction time is
    // excluded; only the search() calls are timed. Same 500 queries and
    // same dataset are reused for every setting.
    // ------------------------------------------------------------------
    std::vector<std::size_t> nprobe_settings = {1, 2, 5, 10, 20};
    // nprobe cannot exceed num_clusters; clip defensively in case someone
    // changes kNumClusters below 20.
    for (auto& n : nprobe_settings) n = std::min(n, kNumClusters);

    std::filesystem::create_directories("results");
    std::ofstream csv("results/benchmark.csv");
    csv << "nprobe,recall_at_10,qps,avg_latency_ms,candidates_examined\n";

    std::cout << std::left << std::setw(10) << "nprobe" << std::setw(14) << "recall@10"
               << std::setw(14) << "qps" << std::setw(18) << "avg_latency_ms"
               << "avg_candidates\n";
    std::cout << std::string(66, '-') << "\n";

    for (std::size_t nprobe : nprobe_settings) {
        // Warm-up pass (not timed): touches memory/pages so the timed pass
        // reflects steady-state performance, not first-touch cache misses.
        for (std::size_t i = 0; i < std::min<std::size_t>(20, queries.size()); ++i) {
            IVFSearchStats warm_stats;
            ivf.search(queries[i], kK, nprobe, &warm_stats);
        }

        double total_recall = 0.0;
        std::size_t total_candidates = 0;

        auto t_start = std::chrono::steady_clock::now();
        std::vector<std::vector<int>> all_results(queries.size());
        for (std::size_t i = 0; i < queries.size(); ++i) {
            IVFSearchStats stats;
            auto results = ivf.search(queries[i], kK, nprobe, &stats);
            total_candidates += stats.candidates_examined;
            all_results[i].reserve(results.size());
            for (const auto& r : results) all_results[i].push_back(r.id);
        }
        auto t_end = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

        for (std::size_t i = 0; i < queries.size(); ++i) {
            total_recall += metrics::recall_at_k(ground_truth[i], all_results[i], kK);
        }

        double avg_recall = total_recall / static_cast<double>(queries.size());
        double qps = static_cast<double>(queries.size()) / elapsed_seconds;
        double avg_latency_ms = (elapsed_seconds * 1000.0) / static_cast<double>(queries.size());
        double avg_candidates =
            static_cast<double>(total_candidates) / static_cast<double>(queries.size());

        std::cout << std::left << std::setw(10) << nprobe << std::setw(14)
                   << std::fixed << std::setprecision(4) << avg_recall << std::setw(14)
                   << std::setprecision(1) << qps << std::setw(18) << std::setprecision(4)
                   << avg_latency_ms << std::setprecision(1) << avg_candidates << "\n";

        csv << nprobe << "," << std::fixed << std::setprecision(6) << avg_recall << ","
            << std::setprecision(3) << qps << "," << std::setprecision(6) << avg_latency_ms << ","
            << total_candidates / queries.size() << "\n";
    }

    std::cout << "\nWrote results/benchmark.csv\n";
    std::cout << "Run `python3 plot_results.py` to generate results/recall_vs_qps.png\n";

    return 0;
}
