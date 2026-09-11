// generate_data.cpp
//
// Generates and saves the reproducible benchmark dataset and the real-text
// demo corpus to disk under data/. Run this once before benchmark/demo (they
// will also auto-generate on first run if the files are missing, but running
// this explicitly makes the pipeline steps visible).

#include "dataset.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {
constexpr std::size_t kNumDatabaseVectors = 50000;
constexpr std::size_t kNumQueries = 500;
constexpr std::size_t kDimension = 128;
constexpr std::size_t kNumClusters = 100;
constexpr std::uint32_t kSeed = 42;

constexpr std::size_t kNumTexts = 5000;
constexpr std::size_t kTextDimension = 128;
constexpr std::uint32_t kTextSeed = 7;
}  // namespace

int main() {
    std::filesystem::create_directories("data");

    std::cout << "==========================================\n";
    std::cout << " Generating synthetic clustered benchmark\n";
    std::cout << "==========================================\n";
    std::cout << "database vectors : " << kNumDatabaseVectors << "\n";
    std::cout << "query vectors    : " << kNumQueries << "\n";
    std::cout << "dimension        : " << kDimension << "\n";
    std::cout << "clusters         : " << kNumClusters << "\n";
    std::cout << "seed             : " << kSeed << "\n\n";

    // cluster_std=1.0 is deliberately chosen (see README "Dataset") so that
    // clusters overlap enough for nprobe to matter -- with tighter clusters
    // recall saturates to 1.0 at nprobe=2 and the speed/accuracy tradeoff
    // becomes uninteresting to look at.
    auto ds = dataset::generate_clustered_dataset(kNumDatabaseVectors, kNumQueries, kDimension,
                                                    kNumClusters, kSeed, /*cluster_std=*/1.0f);

    dataset::save_vectors_binary("data/database_vectors.bin", ds.database_vectors);
    dataset::save_vectors_binary("data/query_vectors.bin", ds.query_vectors);

    std::cout << "Wrote data/database_vectors.bin (" << ds.database_vectors.size()
              << " vectors)\n";
    std::cout << "Wrote data/query_vectors.bin (" << ds.query_vectors.size() << " vectors)\n\n";

    std::cout << "==========================================\n";
    std::cout << " Generating real-text demo corpus\n";
    std::cout << "==========================================\n";
    std::cout << "texts     : " << kNumTexts << "\n";
    std::cout << "dimension : " << kTextDimension << "\n";
    std::cout << "seed      : " << kTextSeed << "\n\n";

    auto corpus = dataset::generate_text_corpus(kNumTexts, kTextDimension, kTextSeed);
    dataset::save_text_corpus("data/text_corpus.txt", corpus);
    std::cout << "Wrote data/text_corpus.txt (" << corpus.texts.size() << " lines)\n";
    std::cout << "\nDone. Vectors for the text corpus are re-derived deterministically from\n";
    std::cout << "text_corpus.txt via dataset::embed_text() rather than stored separately.\n";

    return EXIT_SUCCESS;
}
