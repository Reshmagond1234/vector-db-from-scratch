// demo.cpp
//
// Interactive terminal demo: type any statement, the program embeds it with
// the same deterministic hashing-trick embedding used to build the corpus
// (dataset::embed_text) and returns the top-5 most similar stored
// statements by cosine similarity, using ExactIndex under the hood.
//
// This is real vector similarity search over real (if simply-embedded) text
// vectors -- not a keyword substring search. Two inputs that share no exact
// substrings but overlap in hashed vocabulary will still score high if
// their word sets overlap; it will NOT capture true semantic synonymy (e.g.
// "car" vs "automobile") because the embedding has no learned meaning, only
// shared-token statistics. See README.md "Limitations" for how to plug in a
// real neural embedding model without touching the index code at all --
// ExactIndex/IVFIndex only ever see plain std::vector<float>.

#include "dataset.h"
#include "exact_index.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {
constexpr std::size_t kDimension = 128;
constexpr std::size_t kTopK = 5;
}  // namespace

int main() {
    const std::string corpus_path = "data/text_corpus.txt";
    dataset::TextCorpus corpus;

    if (std::filesystem::exists(corpus_path)) {
        corpus = dataset::load_text_corpus(corpus_path, kDimension);
    } else {
        std::cout << "No corpus found at " << corpus_path << " -- generating 5000 texts now...\n";
        std::filesystem::create_directories("data");
        corpus = dataset::generate_text_corpus(5000, kDimension, /*seed=*/7);
        dataset::save_text_corpus(corpus_path, corpus);
    }

    std::cout << "========================================\n";
    std::cout << "      VECTOR DATABASE FROM SCRATCH\n";
    std::cout << "========================================\n";
    std::cout << "Loaded " << corpus.texts.size() << " statements ("
               << kDimension << "-dim hashing-trick embeddings).\n";
    std::cout << "Type a statement and press Enter. Type 'quit' to exit.\n\n";

    ExactIndex index(kDimension);
    for (std::size_t i = 0; i < corpus.texts.size(); ++i) {
        index.insert(static_cast<int>(i), corpus.vectors[i], corpus.texts[i]);
    }

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        Vector q = dataset::embed_text(line, kDimension);
        auto results = index.search(q, kTopK);

        if (results.empty()) {
            std::cout << "  (no results -- index is empty)\n\n";
            continue;
        }

        std::cout << "\nTop " << results.size() << " matches:\n\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            std::cout << "  " << (i + 1) << ". Score: " << results[i].score << "\n";
            std::cout << "     Text: \"" << index.get_text(results[i].id) << "\"\n\n";
        }
    }

    std::cout << "Goodbye.\n";
    return 0;
}
