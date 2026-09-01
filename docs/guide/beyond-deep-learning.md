# Tropical algebra beyond deep learning

The min-plus semiring did not start in deep learning — it is the native algebra of
**dynamic programming, shortest paths, and scheduling**, and it underpins a small
family of **classical ML** methods. The punchline of this page (and of
[notebook 06 — *Beyond deep learning*](https://github.com/nyuolab/tikitorch/blob/pc-morphssm/notebooks/06_beyond_deep_learning.ipynb)):
the *exact same* `min`-plus matmul that is a tropical NN layer is also the
Floyd–Warshall update — so an algorithm and a network layer are the same primitive.

## Shortest paths and DP *are* tropical matmul

Tropical matrix multiply is `C_ij = min_k (A_ik + B_kj)`. Read `A` as an edge-weight
matrix and that is exactly "best one-hop-extended path from i to j". Iterate it and you
relax all paths — which is why classic DP algorithms are tropical linear algebra.
`tikitorch.algorithmic` ships them as plain tensor ops:

```python
from tikitorch.algorithmic import (
    bellman_ford, floyd_warshall, lis_lengths,
    min_coin_selection, knapsack_selection,
)
# all ops are batched: edge-weight matrices are (B, N, N), sequences (B, L), etc.

dist = floyd_warshall(W)              # (B,N,N) all-pairs shortest path = min-plus closure
d0   = bellman_ford(W, src=0)         # (B,N) single-source
lens = lis_lengths(x)                 # (B,L) longest-increasing-subsequence DP table
used = min_coin_selection(coins, amount)            # (B,K) coins-per-denomination
take = knapsack_selection(weights, values, capacity)  # (B,K) 0/1 membership
```

Each is a min-plus (or max-plus) recurrence; the "selection" variants walk the DP table
back through its argmin/argmax to recover the actual solution — the same argmin routing
that carries gradients in a tropical layer.

### …and a network can *learn* them

Because the layer and the algorithm share a primitive, a tropical net has the right
inductive bias to **learn** these algorithms from input/output examples.
`tikitorch.algorithmic` also provides the task generators and a ready model
(`get_task`, `build_reasoner`, `TropicalReasoner`, `evaluate_task`) used in the
README's *algorithmic reasoning* benchmarks — head-matched against dense baselines.

## Classical tropical machine learning

`tikitorch.classical` provides experimental order-based analogues of familiar algorithms, built on the
tropical (projective) metric `tropical_distance`:

```python
from tikitorch.classical import (
    tropical_distance, tropical_projective_normalize,
    tropical_kmeans, TropicalNearestCentroid, TropicalSVM, tropical_pca,
)

Xn = tropical_projective_normalize(X)         # tropical projective space (mod the all-ones line)
labels, centroids = tropical_kmeans(X, k=3)   # clustering under the tropical metric
clf = TropicalNearestCentroid().fit(X, y)     # min-plus nearest-centroid classifier
svm = TropicalSVM().fit(X, y)                 # tropical max-margin separator
Z   = tropical_pca(X, k=2)                    # tropical principal components
```

- **Tropical k-means / nearest centroid** cluster and classify under the tropical
  distance instead of Euclidean distance.
- **Tropical SVM** finds a max-margin separator in tropical projective space — a
  polytope-based decision boundary, connecting back to
  [Tropical geometry](tropical-geometry.md).
- **Tropical PCA** finds a best-fit tropical-linear subspace (a min-plus span).

## Optimization, scheduling, and control

The same algebra describes **discrete-event systems**: max-plus linear recurrences model
the timing of synchronized processes (manufacturing lines, train schedules, dataflow),
where `x(k+1) = A ⊗ x(k)` with `⊗` = max-plus matmul gives the next event times. The
shortest-path machinery above is the optimization face of the same coin.

## The through-line

A tropical net layer, a shortest-path relaxation, a DP table, and a max-plus schedule
are **one operation** — `min`/`max` of sums — wearing different hats. That is why this
substrate can be useful for add/compare hardware studies and combinatorial reasoning,
and why TikiTorch ships classical and algorithmic modules next to the neural ones. This
is an algebraic connection, not a general performance guarantee. For the runnable tour,
open [notebook 06](https://github.com/nyuolab/tikitorch/blob/pc-morphssm/notebooks/06_beyond_deep_learning.ipynb).
