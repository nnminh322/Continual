# Routing Geometry — Experiment Scripts

## Requirements

```bash
pip install numpy scipy scikit-learn
```

All scripts run on **CPU only** (no GPU needed). They operate on pre-extracted `.npz` embeddings.

## Expected Directory Structure

```
routing_analysis/
├── embeddings/
│   ├── T5EncoderModel/           # or LlamaForCausalLM, etc.
│   │   ├── Long_Sequence/
│   │   │   ├── yelp/
│   │   │   │   ├── train.npz     # keys: embeddings (N, d), labels (N,)
│   │   │   │   ├── dev.npz
│   │   │   │   └── test.npz
│   │   │   ├── amazon/
│   │   │   │   └── ...
│   │   │   └── ...  (15 tasks)
│   │   └── SuperNI/
│   │       └── ...  (15 tasks)
│   └── LlamaForCausalLM/
│       └── ...
├── results/                      # auto-created output directory
├── analyze_geometry.py           # Phase A
├── compare_routing.py            # Phase B+C
├── ablation_psr.py               # Phase D
└── validate_theory.py            # Phase E
```

## Running Experiments

All commands should be run from the `routing_analysis/` directory.

### Phase A — Geometric EDA

```bash
# T5 + Long_Sequence
python analyze_geometry.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --subspace_k 8

# LLaMA + SuperNI
python analyze_geometry.py \
  --emb_dir embeddings/LlamaForCausalLM \
  --benchmark SuperNI \
  --subspace_k 8
```

Output: `results/geometry_{backbone}_{benchmark}.json`

### Phase B+C — Distance Metrics & Classifiers

```bash
# T5 + Long_Sequence
python compare_routing.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --subspace_k 8

# Skip sklearn classifiers (Phase C) if only comparing distances
python compare_routing.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --skip_sklearn
```

Output: `results/routing_{backbone}_{benchmark}.json` + confusion matrices as `.npy`

### Phase D — PSR Ablation

```bash
# Single backbone
python ablation_psr.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --subspace_k 8

# Compare T5 vs LLaMA (pass parent embeddings/ dir)
python ablation_psr.py \
  --emb_dir embeddings \
  --benchmark Long_Sequence \
  --compare_backbones
```

Output: `results/ablation_{backbone}_{benchmark}.json`

### Phase E — Theory Validation

```bash
python validate_theory.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --subspace_k 8
```

Output: `results/theory_{backbone}_{benchmark}.json` + KL/confusion/overlap `.npy` matrices

## Run All (Quick Script)

```bash
cd routing_analysis

for backbone in T5EncoderModel LlamaForCausalLM; do
  for bench in Long_Sequence SuperNI; do
    DIR="embeddings/${backbone}/${bench}"
    if [ -d "$DIR" ]; then
      echo "=== ${backbone} / ${bench} ==="
      python analyze_geometry.py  --emb_dir embeddings/$backbone --benchmark $bench
      python compare_routing.py   --emb_dir embeddings/$backbone --benchmark $bench
      python ablation_psr.py      --emb_dir embeddings/$backbone --benchmark $bench
      python validate_theory.py   --emb_dir embeddings/$backbone --benchmark $bench
    fi
  done
done
```

## CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--emb_dir` | Path to backbone embedding dir (e.g., `embeddings/T5EncoderModel`) | Required |
| `--benchmark` | `Long_Sequence` or `SuperNI` | Required |
| `--subspace_k` | Rank for subspace analysis (matches LoRA rank) | `8` |
| `--out_dir` | Output directory for JSON/npy results | `results` |
| `--skip_sklearn` | (compare_routing only) Skip sklearn classifiers | `false` |
| `--compare_backbones` | (ablation_psr only) Compare all backbones under `--emb_dir` | `false` |

## What Each Phase Tests

| Phase | Script | Key Question |
|-------|--------|-------------|
| A | `analyze_geometry.py` | What is the geometry of embeddings? (dimensionality, Gaussianity, anisotropy) |
| B | `compare_routing.py` | Which distance metric routes best? (L2, cosine, Mahalanobis, spectral, PSR) |
| C | `compare_routing.py` | Which algorithm routes best? (LDA, QDA, SVM, Ridge) |
| D | `ablation_psr.py` | Which PSR components matter? Rank sensitivity? Domain effects? |
| E | `validate_theory.py` | Does KL predict confusion? Grassmann bound holds? Shrinkage helps? |
| F | `simulate_gpm_routing.py` | GPM (ROOT) vs RLS (SpecRoute) vs baselines — incremental simulation |


---

## Phase F — Learned Routing Simulation (GPM vs RLS)

Simulates the actual routing mechanisms from the CL pipeline on pre-extracted embeddings, incrementally (task by task). This is the key experiment to compare learned vs non-parametric routing.

```bash
python simulate_gpm_routing.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark Long_Sequence \
  --mlp_hidden_dim 100 \
  --transthreshold 0.995

# With whitening
python simulate_gpm_routing.py \
  --emb_dir embeddings/T5EncoderModel \
  --benchmark SuperNI \
  --whiten
```

Output: `results/learned_routing_{backbone}_{benchmark}.json`

### Methods Compared in Phase F

| Method | Type | Incremental? | Description |
|--------|------|:---:|-------------|
| **GPM_ROOT** | Learned | ✅ | ROOT/GainLoRA routing: trans_input MLP + prompt_key + GPM null-space protection |
| **RLS_Woodbury** | Analytical | ✅ | SpecRoute: frozen random features + Woodbury ridge regression |
| **PSR** | Non-parametric | ✅ | PPCA generative routing (our proposed) |
| **NearestCentroid** | Non-parametric | ✅ | L2 distance to task mean |
| **CosineNearestCentroid** | Non-parametric | ✅ | Cosine similarity to task mean |


---

## Parameter Glossary

### GPM/ROOT Routing Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--mlp_hidden_dim` | `100` | Hidden dimension of `trans_input` MLP. Architecture: `Linear(d→100)→SiLU→Linear(100→d)→SiLU`. Larger = more expressive routing but more parameters to protect via GPM. ROOT uses 100 for T5-small (d=512). |
| `--transthreshold` | `0.995` | GPM energy threshold for the routing subspace. Controls how many SVD bases are kept after each task. Higher = more bases kept = stronger protection but less room for new tasks. **Adaptive scheduling**: `threshold_t = (1 - base) × t/T + base`, so it grows linearly from `base` toward 1.0 as tasks accumulate. |
| `--lr` | `1e-3` | Learning rate for proxy routing training. In ROOT, routing is trained end-to-end with the seq2seq loss; in offline simulation, we use cross-entropy on task identity as proxy. |
| `--epochs` | `30` | Number of proxy training epochs per task. Only affects task t≥2 (task 1 is trivially correct). |
| `--chunk` | `1` | Number of chunks for GPM covariance. ROOT uses `chunk=1` by default, meaning full d×d covariance. Higher chunk = block-diagonal approximation. |

### GPM "Skip Update" Mechanism

When adding task t>1, GPM computes how much variance the **existing bases** already explain for the new task's activations:

```
accumulated = (total_variance - residual_variance) / total_variance
```

If `accumulated ≥ transthreshold` (existing bases are already sufficient), then **r=0** and GPM **skips** adding new bases for that layer. This means:
- The existing null-space constraint is already tight enough
- No additional memory cost for this task/layer
- The message `"Skip Updating GPM for layer: X"` appears in ROOT's logs

**Implication**: High threshold → more bases → more "skips" in later tasks (subspace fills up).

### RLS Routing Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--rls_expansion` | `2048` | Dimension of random feature expansion (Cover's theorem: higher dim = more linearly separable). SpecRoute default: 2048. |
| `--rls_lambda` | `0.1` | Ridge regularization λ. Controls bias-variance tradeoff of the analytical router. |

### Embedding Layer Selection

| Flag | Script | Choices | Default | Notes |
|------|--------|---------|---------|-------|
| `--layer` | `extract_embeddings_t5.py` | `encoder`, `embedding` | `encoder` | `embedding` = word embedding (before attention) = what CL router uses. `encoder` = last hidden state (after attention). |
| `--layer` | `extract_embeddings_llama.py` | `hidden`, `embedding` | `hidden` | `embedding` = `model.embed_tokens()`. Output dir gets `_wordemb` suffix. |

**Critical**: The CL router operates on **word embeddings** (`avg_inputs_embeds = mean-pooled embed_tokens output`), NOT encoder output. Use `--layer embedding` for faithful comparison.


---

## Routing Accuracy Comparison Table

> Fill in after running experiments. Each cell = routing accuracy (%).

### T5-small (d=512)

| Method | Long_Sequence | SuperNI |
|--------|:---:|:---:|
| **NearestCentroid (L2)** |  |  |
| **CosineNearestCentroid** |  |  |
| **Mahalanobis (pooled)** |  |  |
| **SpectralAffinity** |  |  |
| **PSR_full (k=8)** |  |  |
| **PSR_no_penalty** |  |  |
| **LDA** |  |  |
| **QDA** |  |  |
| **LinearSVM** |  |  |
| **RidgeClassifier (batch)** |  |  |
| **GPM_ROOT (learned)** |  |  |
| **RLS_Woodbury (SpecRoute)** |  |  |

### T5-large (d=1024)

| Method | Long_Sequence | SuperNI |
|--------|:---:|:---:|
| **NearestCentroid (L2)** |  |  |
| **CosineNearestCentroid** |  |  |
| **Mahalanobis (pooled)** |  |  |
| **SpectralAffinity** |  |  |
| **PSR_full (k=8)** |  |  |
| **PSR_no_penalty** |  |  |
| **LDA** |  |  |
| **QDA** |  |  |
| **LinearSVM** |  |  |
| **RidgeClassifier (batch)** |  |  |
| **GPM_ROOT (learned)** |  |  |
| **RLS_Woodbury (SpecRoute)** |  |  |

### LLaMA-7B (d=4096)

| Method | Long_Sequence | SuperNI |
|--------|:---:|:---:|
| **NearestCentroid (L2)** |  |  |
| **CosineNearestCentroid** |  |  |
| **Mahalanobis (pooled)** |  |  |
| **SpectralAffinity** |  |  |
| **PSR_full (k=8)** |  |  |
| **PSR_no_penalty** |  |  |
| **LDA** |  |  |
| **QDA** |  |  |
| **LinearSVM** |  |  |
| **RidgeClassifier (batch)** |  |  |
| **GPM_ROOT (learned)** |  |  |
| **RLS_Woodbury (SpecRoute)** |  |  |

### With ZCA Whitening (T5-small)

| Method | Long_Sequence | SuperNI |
|--------|:---:|:---:|
| **NearestCentroid (L2)** |  |  |
| **PSR_full (k=8)** |  |  |
| **GPM_ROOT (learned)** |  |  |
| **RLS_Woodbury (SpecRoute)** |  |  |

### Word Embedding Layer vs Encoder Output (T5-small, Long_Sequence)

| Method | encoder output | word embedding |
|--------|:---:|:---:|
| **NearestCentroid (L2)** |  |  |
| **PSR_full (k=8)** |  |  |
| **GPM_ROOT (learned)** |  |  |
| **RLS_Woodbury (SpecRoute)** |  |  |

> **Note**: `word embedding` column uses `--layer embedding` during extraction (matches CL router's actual input).


---

## Method Classification

| Category | Methods | Learns? | Incremental? | Memory |
|----------|---------|:---:|:---:|--------|
| Non-parametric | NearestCentroid, Cosine, SpectralAffinity, PSR | ❌ | ✅ | O(d) per task |
| Discriminative (batch) | LDA, QDA, SVM, RidgeClassifier | ✅ | ❌ | O(N×d) retrain |
| Analytical incremental | RLS_Woodbury | ✅ | ✅ | O(E²) fixed |
| Learned + GPM protected | GPM_ROOT | ✅ | ✅ | O(d×h) per task + GPM bases |
