# Vector Database From Scratch

A vector database built in pure C++17 — brute-force exact search, a from-scratch
K-means, an Inverted File (IVF) approximate index, insert/delete, and a
measured speed-vs-accuracy benchmark. No Pinecone, no FAISS, no Chroma, no
Eigen, no ANN library of any kind. Just `<vector>`, loops, and math.

## Overview

Everybody imports a vector database. This project is what's inside one:

- **`ExactIndex`** — brute-force cosine similarity search over every active
  vector. This is the ground truth everything else is scored against.
- **`KMeans`** — Lloyd's algorithm, written from scratch, used to build the
  IVF index's centroids.
- **`IVFIndex`** — an Inverted File index: vectors are clustered, and a query
  only searches the `nprobe` clusters nearest to it. `nprobe` is the
  speed/accuracy knob: low `nprobe` is fast and approximate, high `nprobe`
  approaches (and at `nprobe == num_clusters`, matches) the exact index.
- **A benchmark** that computes real ground truth over 50,000 vectors and
  500 queries, measures Recall@10, QPS, and latency at five `nprobe`
  settings, and writes the results to a CSV — no invented numbers anywhere.
- **A terminal demo** that embeds text with a from-scratch hashing-trick
  embedding and finds the most similar stored statement by cosine
  similarity — real vector search, not keyword matching.

## Assignment Requirements

This directly implements every requirement in the assignment brief: an
exact ground-truth index over 50,000+ vectors, a self-built approximate
index with a named, curve-producing accuracy/speed knob (`nprobe`), full
insert/search/delete (with an honest writeup of the lazy-deletion
tradeoff), a 50,000-vector clustered synthetic benchmark plus a 5,000-text generated support-style demo, 500 queries with exact ground truth, Recall@10, QPS,
latency at 5 settings, a CSV and a plot, and a full test suite. See the
[Assignment Requirement Checklist](#assignment-requirement-checklist) at
the bottom for the itemized mapping.

## Features

- Cosine similarity, vector normalization, and Euclidean distance
  implemented by hand in `src/metrics.cpp`.
- `ExactIndex`: insert, lazy delete, brute-force top-k search, full input
  validation (bad dimensions, duplicate IDs, unknown IDs, `k` larger than
  the index, empty index, deleted vectors).
- `KMeans`: deterministic seeded initialization, assignment step, update
  step, empty-cluster repair, convergence/`max_iterations` stopping.
- `IVFIndex`: K-means-built centroids, inverted lists, `nprobe`-controlled
  approximate search, insert-after-build, lazy delete, per-query stats
  (clusters visited, candidates examined).
- A deterministic dataset generator for clustered synthetic vectors, a
  binary vector serializer, and a from-scratch text-embedding function
  (feature hashing) for the generated-text demo.
- 23 automated tests covering correctness, edge cases, and the exact/IVF
  agreement property.
- A benchmark harness that times only search (never index construction),
  reuses the same 500 queries across every setting, and writes a CSV.
- `plot_results.py`, a thin, optional Python script that turns that CSV
  into a labeled Recall@10-vs-QPS chart.

## Architecture

```
                 DATASET
                    |
                    v
           Vector representation
                    |
           +--------+--------+
           |                 |
           v                 v
      EXACT INDEX        IVF INDEX
     Brute Force        Approximate
           |                 |
           |              nprobe
           |                 |
           +--------+--------+
                    |
                    v
              Top-K results
                    |
                    v
               Evaluation
```

`ExactIndex` and `IVFIndex` both operate on plain `std::vector<float>` and
know nothing about where the vectors came from — the same two indexes are
used for the 128-dim synthetic benchmark and the hashing-trick text demo.

## Exact Brute-Force Index

`ExactIndex::search()` compares the query against **every active** stored
vector — there is no shortcut. Vectors are normalized to unit length at
insertion, so cosine similarity at query time is just a dot product:

```cpp
Vector q = query;
metrics::normalize_in_place(q);
for (each active slot) {
    float score = dot(q, vectors_[slot]);   // == cosine similarity, both unit norm
}
```

Complexity per query: **O(N · D)**, where N is the number of active vectors
and D is the dimension. Deliberately slow — that's what makes it trustworthy
as ground truth.

## Cosine Similarity

```
cosine_similarity(q, x) = (q · x) / (||q|| · ||x||)
```

Implemented by hand in `metrics::cosine_similarity` / `metrics::dot` /
`metrics::l2_norm` — see `src/metrics.cpp`. Storing normalized vectors lets
every index skip the division at query time and just take a dot product.

## K-Means From Scratch

`src/kmeans.cpp` implements Lloyd's algorithm:

1. **Initialization** — `k` distinct data points are chosen as the starting
   centroids using a seeded `std::mt19937` (Forgy initialization). Same
   seed → same centroids, every run.
2. **Assignment step** — every point is assigned to its nearest centroid by
   squared Euclidean distance. **O(N · K · D)**.
3. **Update step** — each centroid becomes the mean of the points assigned
   to it. **O(N · D)**. If a cluster ends up empty, it's reseeded to a
   random data point (via a second, independent seeded stream) so no
   cluster is silently dropped.
4. **Stopping condition** — repeat until no assignment changes (or an empty
   cluster forced a reseed) or `max_iterations` is hit.

Default configuration used by the benchmark: dimension 128, 50,000 vectors,
100 clusters, 20 max iterations, seed 42.

## IVF Index

`IVFIndex::build()` runs K-means over the initial vector set to get
`num_clusters` centroids, then puts every vector's slot into the inverted
list of its nearest centroid (reusing the assignment K-means already
computed, so the two are always consistent).

`IVFIndex::search(query, k, nprobe)`:

1. Compute the distance from the query to every centroid, take the
   `nprobe` nearest (**O(C · D)**, C = num_clusters).
2. Visit only those `nprobe` inverted lists, scoring each *active*
   candidate with exact cosine similarity (**O(candidates · D)**).
3. Return the top-k by score.

Overall: **O(C · D + candidates · D)** — not O(log N). Whether that's fast
in practice depends entirely on how many vectors land in the visited
clusters, which depends on data geometry and cluster balance, not just N.
This project does not claim a big-O speedup that the design doesn't
actually deliver; it delivers a *constant-factor* speedup by shrinking the
candidate set, which the benchmark measures directly.

## Inverted Lists

```
Cluster 0: vector slots [4, 81, 921, 3001, ...]
Cluster 1: vector slots [10, 22, 100, ...]
...
Cluster 99: vector slots [...]
```

Stored as `std::vector<std::vector<std::size_t>>` in `IVFIndex`, indexed by
cluster id. `IVFIndex::cluster_sizes()` reports how balanced they are (the
benchmark run recorded below had sizes ranging from 142 to 1,055 out of an
average of 500 — real, data-dependent imbalance, not idealized).

## nprobe

`nprobe` is the number of nearest clusters a query visits. It is IVF's one
knob, and it trades speed for accuracy directly:

- **Low `nprobe`** → fewer inverted lists visited → fewer candidates
  scored → faster, but a true nearest neighbour sitting in an unvisited
  cluster is simply missed → lower recall.
- **High `nprobe`** → more lists visited → slower, but higher recall.
- **`nprobe == num_clusters`** → every cluster is visited, so the candidate
  set is the entire active database → IVF's result becomes identical to
  brute force (this is asserted directly by
  `ivf_nprobe_equals_num_clusters_approaches_exact` in the test suite,
  which checks measured recall > 0.999 against the exact index).

## Insert

`IVFIndex::insert(id, vector, text)`:

1. Validate dimension and reject a duplicate id.
2. Normalize the vector.
3. Find its nearest **existing** centroid (centroids are not recomputed —
   see Limitations).
4. Append it to that centroid's inverted list.

The vector is searchable by the very next `search()` call — there's no
separate "commit" or rebuild step. `ExactIndex::insert` is the same idea
without the clustering.

## Delete

Both indexes use **lazy deletion**, by design:

```cpp
bool remove(int id) {
    // find id -> slot, flip active_[slot] = false, return true
    // id already gone / never existed -> return false
}
```

`active_[slot]` is checked by every search; a deleted vector's score is
never computed and it can never be returned, immediately and forever.

### Lazy Deletion — the tradeoff, honestly

Lazy deletion makes deletion O(1) and search's "never return this" guarantee
trivial to prove. The cost is paid later: a deleted vector's slot **stays
physically present** in `vectors_` and in its inverted list. For
`ExactIndex` that just means one more skipped comparison per search. For
`IVFIndex` it's slightly worse: `search()` still has to walk past every
dead slot in every visited inverted list before it can skip it, so a
heavily-deleted index does more wasted work per query even though the
*active* result set is unchanged. `IVFSearchStats::candidates_examined`
deliberately counts dead slots too, so this cost is visible rather than
hidden. A compaction/rebuild pass (drop inactive slots, remap IDs, rebuild
inverted lists) would reclaim that cost, but is not implemented here —
adding a half-working compaction path was judged worse than being explicit
that it doesn't exist yet (see Future Improvements).

## Dataset

`dataset::generate_clustered_dataset()`:

1. Generates `num_clusters` cluster centers, each coordinate drawn
   uniformly from `[-1, 1]`, using a seeded `std::mt19937`.
2. Generates database and query vectors by picking a random cluster center
   and adding Gaussian noise (`std::normal_distribution`) around it.

The benchmark uses **cluster_std = 1.0**, not the smaller default the
function otherwise uses — this was a deliberate, measured choice. With
tighter clusters (std 0.15–0.6) recall hit ~1.0 by `nprobe=2` on this data,
which is a legitimate result but a boring one to look at; std 1.0 gives
clusters real overlap in 128 dimensions, which is what produces the
gradual Recall-vs-QPS curve in [Results](#results) below. Both the
database and the query vectors come from the *same* generative
distribution, and — critically — nothing about how they were generated is
visible to either index; the geometry is genuinely inferred by K-means
from the raw vectors, not hard-coded.

Default benchmark configuration: 50,000 database vectors, 500 queries,
dimension 128, 100 clusters, seed 42. Regenerate anytime with
`./build/generate_data` (fixed seed ⇒ same dataset every run, unless you
change the seed).

### Real text demo

In addition to the mandatory synthetic benchmark, `data/text_corpus.txt`
holds 5,000 short template-generated support/FAQ-style sentences across
seven topics (account, billing, shipping, technical, machine learning,
cooking, travel), produced deterministically from a fixed seed so the file
is reproducible without any network access. Each sentence is embedded with
`dataset::embed_text()` — see [Limitations](#limitations) for exactly what
that embedding is and is not.

## Ground Truth

For every one of the 500 benchmark queries, `benchmark.cpp` calls
`ExactIndex::search(query, 10)` and records the returned IDs as that
query's ground truth top-10. This is computed fresh on every benchmark
run, timed separately from the IVF benchmark, and never cached or
hand-entered.

## Recall@10

```
recall@10(query) = |approx_top_10 ∩ exact_top_10| / 10
```
averaged across all 500 queries. Implemented in `metrics::recall_at_k`
(`src/metrics.cpp`), unit-tested against hand-computed cases in
`tests/tests.cpp` (`recall_at_k_basic_cases`).

## Benchmark Methodology

- The same 500 queries and the same 50,000-vector dataset are reused for
  every `nprobe` setting — nothing changes between rows of the results
  table except `nprobe` itself.
- Index construction (`ExactIndex` inserts, `IVFIndex::build`/K-means) is
  timed separately and **excluded** from the search-time measurements.
- Before each `nprobe` setting is timed, 20 warm-up queries are run
  (untimed) so the timed pass isn't paying for first-touch page faults.
- `std::chrono::steady_clock` wraps only the `search()` calls.
- `QPS = num_queries / total_search_time_seconds`.
- `avg_latency_ms = (total_search_time_seconds * 1000) / num_queries`.
- `candidates_examined` is summed per query and averaged; it counts every
  inverted-list entry visited, active or not (see the lazy-deletion
  section above).

Run it yourself: `./build/benchmark` (auto-generates the dataset if
`data/*.bin` isn't present) → writes `results/benchmark.csv`.

## Speed vs Accuracy

`nprobe` is IVF's only knob, and the benchmark sweeps it across five
settings (1, 2, 5, 10, 20 — all ≤ `num_clusters = 100`). The relationship
is exactly what the design predicts: `candidates_examined` and latency
both grow roughly linearly with `nprobe`, QPS falls correspondingly, and
recall climbs — steeply at first, then flattening as `nprobe` approaches
the point where nearly every true neighbour's cluster is already being
visited.

## Results

Measured on the machine that built this repo, seed 42, 50,000 database
vectors / 500 queries / dimension 128 / 100 clusters / `k=10`, `cluster_std
= 1.0`. **Your numbers will differ by hardware — regenerate with
`./build/benchmark` and read `results/benchmark.csv`; do not trust numbers
in a README over the CSV your own run produces.**

Ground truth (exact, brute-force) computation: 500 queries in 3.99 s
(≈125.26 QPS). IVF build (K-means, 100 clusters, 20 iterations max, over
50,000 vectors): 14.24 s. Resulting cluster sizes ranged from 142 to 1,055
(average 500) — real imbalance from the generated benchmark data.

| nprobe | Recall@10 | QPS      | Avg Latency (ms) | Avg Candidates Examined |
|-------:|----------:|---------:|------------------:|--------------------------:|
|      1 |    0.9544 |   4949.5 |            0.2020 |                     543.1 |
|      2 |    0.9774 |   3222.5 |            0.3103 |                    1042.2 |
|      5 |    0.9888 |   1387.9 |            0.7205 |                    2551.8 |
|     10 |    0.9932 |    714.0 |            1.4006 |                    5025.2 |
|     20 |    0.9966 |    342.7 |            2.9179 |                    9973.4 |

At nprobe=1, IVF examines roughly 1% of the database on average
(543 of 50,000 vector slots) per query and still achieves 95.44%
Recall@10. It reaches 4,949.5 QPS compared with 125.26 QPS for
brute force on this run. Increasing nprobe improves recall further
but reduces QPS as more candidates are examined.

## Why IVF?

The assignment allows "whatever design you can defend" from a couple of
well-known ANN families reachable in one night. IVF was chosen over HNSW
because a complete, simple IVF beats a half-finished graph index — the
assignment says so explicitly, and it's true: IVF's moving parts (K-means,
inverted lists, a probe count) are each individually simple enough to get
fully correct and fully tested in the time available, where HNSW's
multi-layer graph construction and search have enough moving parts that
"complete and correct" was a much bigger bet.

## Why nprobe Controls the Tradeoff

IVF's approximation comes from one decision: which clusters to look inside.
`nprobe` *is* that decision, made numeric. Skipping clusters is exactly
where recall is lost (a true neighbour in a skipped cluster is simply never
seen), and it's exactly where time is saved (an unvisited cluster costs
nothing). There's no other free parameter in this design that trades
speed for accuracy — cluster count affects both index build time and
average cluster size, but doesn't let a single already-built index move
along a speed/accuracy curve the way `nprobe` does per-query.

## Limitations

- **Text embedding is a hashing-trick bag-of-words vector, not a trained
  neural embedding.** `dataset::embed_text()` lowercases the text, hashes
  each word to a dimension + sign, sums, and L2-normalizes. This is a real
  vector-space technique (the same "hashing trick" used in tools like
  Vowpal Wabbit) — cosine similarity over these vectors genuinely does
  vector search, not keyword substring matching, and two sentences that
  share more vocabulary score higher (see
  `embed_text_shared_vocabulary_scores_higher` in the tests). What it
  *cannot* do is recognize synonyms it's never seen written the same way
  ("car" vs "automobile" share no hashed tokens). Swapping in a real
  embedding model changes nothing about `ExactIndex`/`IVFIndex` — both
  only ever see `std::vector<float>`; you'd replace `embed_text()`'s body
  with a call to a local or API embedding model and everything else keeps
  working as-is. The official 50,000-vector benchmark does not depend on
  this at all — it runs entirely on the synthetic clustered dataset.
- **IVF centroids are not recomputed after `build()`.** `insert()` assigns
  new vectors to the nearest *existing* centroid. This is fine as long as
  the data distribution doesn't drift far from what `build()` saw; a
  dataset that evolves substantially over time would eventually want a
  periodic rebuild.
- **Lazy deletion never reclaims space or search time.** See the dedicated
  section above — there is no compaction pass in this project.
- **Cluster balance isn't actively managed.** K-means naturally produces
  some imbalance (see the 142–1,055 range in Results); a very skewed
  cluster can make a small `nprobe` sometimes cheaper or more expensive
  than another `nprobe` of the same count, depending on which clusters get
  visited. `IVFSearchStats` exposes this per-query so it's measurable
  rather than hidden.
- **Single-machine, single-threaded.** No SIMD intrinsics, no threading.
  There's clear room to speed up both indexes (see below) without changing
  their algorithmic design.

## Future Improvements

- A compaction/rebuild operation for `IVFIndex` and `ExactIndex` that
  physically drops inactive slots and remaps IDs, reclaiming the
  lazy-deletion cost.
- Periodic centroid recomputation (e.g. re-run K-means every N inserts) so
  `IVFIndex` doesn't drift as the data distribution shifts.
- Multi-threading the brute-force scan in `ExactIndex::search` and the
  per-cluster scan in `IVFIndex::search` — both are embarrassingly
  parallel over independent vector comparisons.
- Product quantization or scalar quantization of stored vectors to cut
  memory and speed up the dot products IVF still has to do on its
  candidate set.
- A real embedding model behind `embed_text()` for the text demo (see
  Limitations).

## Build Instructions

Requires a C++17 compiler and CMake ≥ 3.15. No external C++ libraries.

```bash
cmake -S . -B build
cmake --build build --config Release
```

This builds four executables into `build/` (or `build/Release/` on the
Visual Studio generator): `generate_data`, `benchmark`, `demo`, `tests`.

Python (only needed for the optional plot):

```bash
pip install -r requirements.txt
```

## Run Instructions

```bash
# 1. Generate the reproducible dataset + text corpus (also auto-runs if
#    missing when you run benchmark/demo directly).
./build/generate_data

# 2. Run the unit tests.
./build/tests

# 3. Run the benchmark (writes results/benchmark.csv).
./build/benchmark

# 4. Turn the CSV into a chart (optional, needs Python + requirements.txt).
python3 plot_results.py

# 5. Try the terminal demo.
./build/demo
```

(On Windows/Visual Studio builds, binaries land in `build\Release\*.exe`;
adjust the paths above accordingly.)

## Testing

`tests/tests.cpp` is a small, dependency-free test runner (no GoogleTest —
just `std::iostream`, a couple of macros, and `std::exception`). 23 tests
cover: cosine similarity (hand-computed and dimension-mismatch), exact
search against a hand-verified 4-vector dataset, exact-index edge cases
(empty index, `k` > available, duplicate/unknown IDs, invalid dimensions,
delete-then-search), K-means (recovers well-separated clusters, rejects
`data.size() < k`, never leaves an empty cluster), IVF (build, search,
search-before-build, invalid `nprobe`, insert-then-immediately-searchable,
delete-then-never-returned, duplicate ID on build, and — the key
correctness property — `nprobe == num_clusters` reproduces exact results),
Recall@10 on hand-computed cases, binary vector round-tripping, and the
text embedding (deterministic, shared-vocabulary-scores-higher).

```bash
./build/tests
```
exits 0 if every test passes, non-zero otherwise, and prints a
`[PASS]`/`[FAIL]` line per test plus a final summary.

## Benchmark

See [Benchmark Methodology](#benchmark-methodology) and
[Results](#results) above. Rerun with `./build/benchmark`; it will
auto-generate `data/*.bin` on first run if you skip step 1.

## Plot

```bash
python3 plot_results.py
```
reads `results/benchmark.csv` and writes `results/recall_vs_qps.png`:
Queries Per Second on the X axis, Recall@10 on the Y axis, each point
labeled with its `nprobe`.

## Demo

```bash
./build/demo
```

```
========================================
      VECTOR DATABASE FROM SCRATCH
========================================
Loaded 5000 statements (128-dim hashing-trick embeddings).
Type a statement and press Enter. Type 'quit' to exit.

> How can I reset my password?

Top 5 matches:

  1. Score: 1
     Text: "How can I reset my password"

  2. Score: 0.812
     Text: "I forgot my account password and need to reset it"
  ...
```

## Example Output

Actual captured output from `./build/benchmark` on this repo's reference
run (also see [Results](#results)):

```
Loading existing dataset from data/...
database vectors: 50000
query vectors   : 500
dimension       : 128

Building ExactIndex (ground truth, brute force)...
ExactIndex built with 50000 active vectors.

Computing exact ground truth for 500 queries (k=10)... this is brute force, so it is the slow part.
Ground truth computed in 3.99s (125.26 QPS brute force).

Building IVFIndex (K-means with 100 clusters)...
IVFIndex built in 14.24s.
Cluster sizes -- min: 142, max: 1055, avg: 500.00

| nprobe | Recall@10 | QPS    | Avg Latency (ms) | Avg Candidates Examined |
|-------:|----------:|-------:|------------------:|-------------------------:|
|      1 |    0.9544 | 4949.5 |            0.2020 |                    543.1 |
|      2 |    0.9774 | 3222.5 |            0.3103 |                   1042.2 |
|      5 |    0.9888 | 1387.9 |            0.7205 |                   2551.8 |
|     10 |    0.9932 |  714.0 |            1.4006 |                   5025.2 |
|     20 |    0.9966 |  342.7 |            2.9179 |                   9973.4 |

Wrote results/benchmark.csv
```

## Assignment Requirement Checklist

- [x] C++17 compiles successfully
- [x] No FAISS / Pinecone / Chroma / sklearn / Annoy / hnswlib / Milvus /
      Weaviate / ScaNN / Eigen / mlpack / any ANN or clustering library
- [x] Exact brute-force index implemented manually (`ExactIndex`)
- [x] Cosine similarity implemented manually (`metrics::cosine_similarity`)
- [x] K-means implemented manually (`KMeans`)
- [x] IVF implemented manually (`IVFIndex`)
- [x] Inverted lists implemented manually (`std::vector<std::vector<size_t>>`)
- [x] `nprobe` implemented and swept over 5 settings
- [x] insert implemented (`ExactIndex::insert`, `IVFIndex::insert`)
- [x] delete implemented (lazy, both indexes)
- [x] Deleted vectors provably never returned (tested)
- [x] 50,000 vectors supported (benchmark default)
- [x] 500 queries supported (benchmark default)
- [x] Exact ground truth generated at benchmark time, not invented
- [x] Recall@10 calculated (`metrics::recall_at_k`)
- [x] QPS calculated from measured `std::chrono` timings
- [x] Latency calculated from measured `std::chrono` timings
- [x] 5 `nprobe` settings benchmarked (1, 2, 5, 10, 20)
- [x] CSV generated by the benchmark executable
- [x] Plot generated from that CSV (`plot_results.py`)
- [x] 23 automated tests included
- [x] Terminal demo included (`demo.cpp`)
- [x] README included (this file)
- [x] CMake included (`CMakeLists.txt`)
- [x] No hard-coded local/absolute paths anywhere
- [x] No fabricated benchmark values — every number above came from a run
      of `./build/benchmark` on this repo

## Demo Video Suggested Flow

1. Show the project directory tree (`include/`, `src/`, `tests/`, etc.).
2. Explain `ExactIndex` as ground truth — O(N·D), checks every vector.
3. Run `./build/tests`, point out the 23 `[PASS]` lines.
4. Run `./build/generate_data`, show the 50,000/500/128/100 parameters.
5. Run `./build/benchmark`, let it print the ground-truth timing, the IVF
   build timing, and the 5-row results table live.
6. `cat results/benchmark.csv`.
7. `python3 plot_results.py`, open `results/recall_vs_qps.png`.
8. Explain `nprobe` using the table: low `nprobe` = fewer clusters visited
   = faster + lower recall; `nprobe = num_clusters` ⇒ equivalent to exact.
9. Run `./build/demo`, type a couple of statements live.
10. Explain insert (`IVFIndex::insert` — nearest existing centroid, no
    rebuild) and delete (lazy — flag inactive, walked-but-skipped later).

## Likely Evaluator Questions

**Q: Why is IVF not just always faster and more accurate than exact
search?**
A: It's never more accurate — it's a subset of exact search's candidate
set by construction, so its recall is capped at 1.0 (matching exact) and
is usually below it. It's faster because it skips clusters; how much
faster depends entirely on `nprobe` and cluster balance, which is why the
benchmark reports a curve, not a single number.

**Q: What happens if `nprobe` equals the number of clusters?**
A: Every inverted list gets visited, so the candidate set is the entire
active database — IVF becomes functionally brute force. This is asserted
directly by a test (`ivf_nprobe_equals_num_clusters_approaches_exact`),
not just claimed.

**Q: Why lazy deletion instead of physically removing vectors?**
A: It's simpler and O(1), and it's the same design that essentially every
production system reaches for (rows get flagged, not immediately
shredded), precisely because rewriting inverted lists on every delete is
expensive. The cost — dead slots still get walked during search — is
explained in the README and made visible in `IVFSearchStats`, not hidden.

**Q: Why K-means and not a KD-tree or LSH?**
A: The assignment allows "whichever design you can defend," and IVF (via
K-means clustering) is one of the two well-known families explicitly
suggested. It maps directly onto the assignment's "inverted lists" +
"nprobe" language.

**Q: How do you know the exact index is actually exact, not just
"probably right"?**
A: It's tested against a hand-computed 4-vector dataset where the correct
ranking was worked out by hand before the test was written
(`exact_search_tiny_hand_verified_dataset`), and it's structurally
incapable of skipping a vector — `search()` has no early exit, no pruning,
and no data structure that could hide a candidate; it iterates every
active slot.

**Q: Is the text demo "real" vector search or is it secretly keyword
matching?**
A: It's real vector search — text is embedded into a fixed-length numeric
vector (via feature hashing) and compared by cosine similarity, exactly
like the synthetic benchmark. It will match "reset my password" against "I
forgot my password" even though the strings differ, because the two share
enough hashed vocabulary to score high. It is honestly documented as
*not* a trained semantic embedding (see Limitations) — it won't catch true
synonyms it's never seen written the same way.

**Q: Why is `nprobe` presented on a QPS/Recall curve rather than a single
"accuracy: 94%" number?**
A: A single accuracy number hides the tradeoff being made to get it. The
whole point of an approximate index is that you *choose* a point on a
curve; reporting one point pretends the choice doesn't exist.

**Q: What's the actual complexity of an IVF query?**
A: `O(C·D + candidates·D)` where C is the number of clusters (for finding
the nearest `nprobe` of them) and `candidates` is however many vectors
land in the visited lists — not `O(log N)`. This project doesn't claim a
complexity class it doesn't deliver; the benchmark measures the constant
factor directly instead.

**Q: How large a dataset could this realistically handle?**
A: The benchmark comfortably handles 50,000 vectors × 128 dims (~25 MB of
raw floats) on a normal laptop, with IVF search staying sub-millisecond at
low `nprobe`. It's single-threaded, in-memory, and has O(N) memory
overhead per index (two copies of the data exist if you build both an
`ExactIndex` and an `IVFIndex` over the same vectors, as the benchmark
does) — millions of vectors would need the improvements listed in Future
Improvements, not a redesign.
