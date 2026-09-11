#include "dataset.h"
#include "metrics.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace dataset {

ClusteredDataset generate_clustered_dataset(std::size_t num_database_vectors,
                                             std::size_t num_queries, std::size_t dimension,
                                             std::size_t num_clusters, std::uint32_t seed,
                                             float cluster_std) {
    if (dimension == 0) throw std::invalid_argument("generate_clustered_dataset - dimension must be > 0");
    if (num_clusters == 0) throw std::invalid_argument("generate_clustered_dataset - num_clusters must be > 0");

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, cluster_std);
    std::uniform_int_distribution<std::size_t> cluster_pick(0, num_clusters - 1);

    // Step 1: generate cluster centers.
    std::vector<Vector> centers(num_clusters, Vector(dimension));
    for (auto& c : centers) {
        for (std::size_t d = 0; d < dimension; ++d) {
            c[d] = center_dist(rng);
        }
    }

    auto sample_around_cluster = [&]() -> Vector {
        std::size_t c = cluster_pick(rng);
        Vector v(dimension);
        for (std::size_t d = 0; d < dimension; ++d) {
            v[d] = centers[c][d] + noise_dist(rng);
        }
        return v;
    };

    ClusteredDataset ds;
    ds.dimension = dimension;
    ds.num_clusters = num_clusters;

    ds.database_vectors.reserve(num_database_vectors);
    ds.database_ids.reserve(num_database_vectors);
    for (std::size_t i = 0; i < num_database_vectors; ++i) {
        ds.database_vectors.push_back(sample_around_cluster());
        ds.database_ids.push_back(static_cast<int>(i));
    }

    ds.query_vectors.reserve(num_queries);
    for (std::size_t i = 0; i < num_queries; ++i) {
        ds.query_vectors.push_back(sample_around_cluster());
    }

    return ds;
}

void save_vectors_binary(const std::string& path, const std::vector<Vector>& vectors) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("save_vectors_binary - could not open '" + path + "' for writing");
    }
    std::uint32_t count = static_cast<std::uint32_t>(vectors.size());
    std::uint32_t dim = vectors.empty() ? 0 : static_cast<std::uint32_t>(vectors[0].size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    for (const auto& v : vectors) {
        if (v.size() != dim) {
            throw std::invalid_argument("save_vectors_binary - all vectors must share the same dimension");
        }
        out.write(reinterpret_cast<const char*>(v.data()),
                   static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("save_vectors_binary - write failed for '" + path + "'");
    }
}

std::vector<Vector> load_vectors_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_vectors_binary - could not open '" + path + "' for reading");
    }
    std::uint32_t count = 0, dim = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in) {
        throw std::runtime_error("load_vectors_binary - truncated header in '" + path + "'");
    }
    std::vector<Vector> vectors(count, Vector(dim));
    for (std::uint32_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(vectors[i].data()),
                static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in) {
            throw std::runtime_error("load_vectors_binary - truncated data in '" + path + "'");
        }
    }
    return vectors;
}

// ---------------------------------------------------------------------
// Text embedding: the hashing trick.
// ---------------------------------------------------------------------

Vector embed_text(const std::string& text, std::size_t dimension) {
    Vector v(dimension, 0.0f);

    std::string lower;
    lower.reserve(text.size());
    for (char ch : text) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    std::istringstream iss(lower);
    std::string word;
    while (iss >> word) {
        // Strip simple punctuation from word edges so "password?" and
        // "password" hash the same.
        while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.front()))) {
            word.erase(word.begin());
        }
        while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.back()))) {
            word.pop_back();
        }
        if (word.empty()) continue;

        std::size_t h = std::hash<std::string>{}(word);
        std::size_t index = h % dimension;
        // Use a second, independent bit of the hash to pick a sign, which
        // is standard practice for the hashing trick: it makes unrelated
        // words partially cancel instead of only ever adding up, so raw
        // vocabulary size doesn't just inflate every score.
        bool sign_bit = ((h / dimension) & 1u) != 0;
        v[index] += sign_bit ? 1.0f : -1.0f;
    }

    metrics::normalize_in_place(v);
    return v;
}

namespace {

// A small set of support/FAQ-style sentence templates across topics, with
// slot-fillers that get deterministically combined to produce thousands of
// distinct-but-related short texts. This keeps the corpus self-contained
// (no network access / no external dataset download needed) while still
// giving nearby vectors real shared vocabulary to key off of.
struct TemplateSet {
    std::string topic;
    std::vector<std::string> sentences;
};

const std::vector<TemplateSet>& topic_templates() {
    static const std::vector<TemplateSet> templates = {
        {"account", {
            "How can I reset my password",
            "I forgot my account password and need to reset it",
            "Where do I change my email address on my account",
            "How do I update my profile information",
            "I want to delete my account permanently",
            "How can I enable two factor authentication",
            "My account got locked after too many login attempts",
            "How do I recover a suspended account",
            "I cannot log in with my username and password",
            "How do I change my display name",
        }},
        {"billing", {
            "How do I update my credit card on file",
            "Why was I charged twice this month",
            "I want to cancel my subscription",
            "How do I get a refund for my last payment",
            "Where can I download my invoice",
            "How do I switch from monthly to annual billing",
            "My payment failed, what should I do",
            "How do I apply a discount code to my order",
            "I need a receipt for my last purchase",
            "How do I change my billing address",
        }},
        {"shipping", {
            "Where is my package right now",
            "How long does standard shipping take",
            "Can I change the delivery address after ordering",
            "My order arrived damaged, what should I do",
            "How do I track my shipment",
            "Do you ship internationally",
            "My package says delivered but I never received it",
            "How much does express shipping cost",
            "Can I schedule a specific delivery time",
            "How do I return an item for a refund",
        }},
        {"technical", {
            "The app keeps crashing on startup",
            "How do I clear the cache in the mobile app",
            "The website is loading very slowly today",
            "I am getting an error code when I try to save",
            "How do I update the app to the latest version",
            "The search feature is not returning any results",
            "How do I enable dark mode",
            "My uploads keep failing halfway through",
            "How do I connect the app to my calendar",
            "The page keeps freezing when I scroll",
        }},
        {"machine_learning", {
            "Explain how machine learning algorithms work",
            "What is supervised learning in simple terms",
            "How does a neural network learn from data",
            "What is the difference between classification and regression",
            "How do recommendation systems choose what to show me",
            "What is overfitting in a machine learning model",
            "How does gradient descent optimize a model",
            "What is the purpose of a validation dataset",
            "How do clustering algorithms group similar data points",
            "What makes deep learning different from traditional machine learning",
        }},
        {"cooking", {
            "How long should I bake chicken breast in the oven",
            "What is a good substitute for buttermilk in a recipe",
            "How do I keep pasta from sticking together",
            "What temperature should I fry chicken at",
            "How do I make a simple tomato sauce from scratch",
            "What is the best way to store fresh basil",
            "How do I know when steak is medium rare",
            "What can I use instead of eggs in baking",
            "How do I make rice fluffy instead of sticky",
            "What spices go well with roasted vegetables",
        }},
        {"travel", {
            "What documents do I need to travel internationally",
            "How early should I arrive at the airport",
            "What is the best way to find cheap flights",
            "Do I need a visa to visit this country",
            "How much luggage can I bring on a flight",
            "What should I pack for a week long trip",
            "How do I exchange currency before traveling",
            "What is the best time of year to visit",
            "How do I get from the airport to downtown",
            "Is travel insurance worth buying for this trip",
        }},
    };
    return templates;
}

}  // namespace

TextCorpus generate_text_corpus(std::size_t num_texts, std::size_t dimension,
                                 std::uint32_t seed) {
    const auto& templates = topic_templates();

    // Flatten all base sentences.
    std::vector<std::string> base_sentences;
    for (const auto& t : templates) {
        for (const auto& s : t.sentences) {
            base_sentences.push_back(s);
        }
    }

    // Small set of deterministic paraphrase-ish suffixes/prefixes used to
    // generate variety beyond the base sentences without needing a network
    // call or an external dataset. This is purely templated text
    // augmentation, documented here and in the README -- it is not claimed
    // to be a rich, naturalistic corpus, just a reproducible stand-in large
    // enough to exercise the index.
    std::vector<std::string> variations = {
        "", " please", " thanks", " today", " right now", " again",
        " on mobile", " on the website", " urgently", " for my account",
    };

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> base_pick(0, base_sentences.size() - 1);
    std::uniform_int_distribution<std::size_t> var_pick(0, variations.size() - 1);

    TextCorpus corpus;
    corpus.dimension = dimension;
    corpus.texts.reserve(num_texts);
    corpus.vectors.reserve(num_texts);

    for (std::size_t i = 0; i < num_texts; ++i) {
        std::string text = base_sentences[base_pick(rng)] + variations[var_pick(rng)];
        corpus.texts.push_back(text);
        corpus.vectors.push_back(embed_text(text, dimension));
    }

    return corpus;
}

void save_text_corpus(const std::string& path, const TextCorpus& corpus) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("save_text_corpus - could not open '" + path + "' for writing");
    }
    for (const auto& t : corpus.texts) {
        // Sentences are generated without embedded newlines, so one line
        // per text is a safe, simple format.
        out << t << "\n";
    }
}

TextCorpus load_text_corpus(const std::string& path, std::size_t dimension) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_text_corpus - could not open '" + path + "' for reading");
    }
    TextCorpus corpus;
    corpus.dimension = dimension;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        corpus.texts.push_back(line);
        corpus.vectors.push_back(embed_text(line, dimension));
    }
    return corpus;
}

}  // namespace dataset
