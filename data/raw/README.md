# Raw datasets

Everything under this directory is **gitignored** except this README. It holds
NoC-topology datasets you generate locally:

```bash
python scripts/generate_dataset.py --connections 8 18 --samples-per-class 10000 \
    --seed 0 --out data/raw/nocs.npz
```

Each run writes a single compressed `.npz`:

| key | dtype / shape | meaning |
|---|---|---|
| `matrices` | int8 `(N, 9, 9)` | binary symmetric adjacency matrices, zero diagonal |
| `n_connections` | int32 `(N,)` | physical connection count (undirected edges) per sample |
| `n_routers` | int scalar | routers per topology (9) |

The paper's dataset: **9 routers**, physical connection counts **8..18**, about
**10k unique topologies per count** (~110k total), built purely by random
construction (RAPIDO 2021 / PhD thesis Chapter 5, "Algorithm 1"), every sample
is a connected 9×9 graph with all routers wired and degree ≤ 4. No NoC
simulator is involved; these datasets carry graph structure only. Latency
labels are not produced here.
