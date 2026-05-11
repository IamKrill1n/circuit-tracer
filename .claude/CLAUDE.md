# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a fork of `safety-research/circuit-tracer`, extended with a **summarization pipeline** that automates graph pruning, clustering into supernodes, and uploading results to Neuronpedia. The core library computes attribution graphs (circuits) for transformer language models using MLP transcoders.

## Installation

```bash
pip install .
# For dev + visualization deps:
pip install ".[dev,viz]"
```

## Commands

**Run tests:**
```bash
pytest tests/
# Single test file:
pytest tests/test_prune.py
# Single test by name:
pytest tests/test_prune.py::test_validate_threshold_valid_values
# Skip disk-heavy tests:
pytest -m "not requires_disk"
```

**Lint / type-check:**
```bash
ruff check .
pyright
```

**Run the summarization pipeline (prune + cluster + upload):**
```bash
python -m summarization --prompt "..." --slug my-slug --graph-json path/to/graph.json
# With auto-k selection and upload:
python -m summarization --use-existing-graph --graph-json demos/flow_analysis.json --auto-k --upload
```

**Run the CLI attribution pipeline:**
```bash
circuit-tracer attribute --prompt "..." --transcoder_set gemma --slug my-slug --graph_file_dir ./out --server
```

**Evaluation workflow (two steps):**
```bash
# Step 1: produce PruneGraph .pt files + manifest.json
python eval/prune_graphs.py --graphs-root demos/temp_graph_files --output-root eval_outputs/prune/subgraph

# Step 2: compute metrics from the manifest produced above
python eval/eval_prune.py --manifest eval_outputs/prune/subgraph/<source-set>/manifest.json

# Cluster quality sweep (reads PruneGraph .pt files directly):
python eval/eval_cluster.py --prune-dir eval_outputs/prune/subgraph/<source-set>
```

**Generate graphs from Neuronpedia:**
```bash
python generate_new_graphs.py
```

## Environment / Config

API keys are loaded from `.env` in the repo root (see [config.py](config.py)):
- `NEURONPEDIA_API_KEY` — required for graph download, feature lookup, and upload
- `OPENAI_API_KEY` — used by LLM-based grouping in `summarization/group_llm.py`
- `GEMINI_API_KEY` / `GENAI_API_KEY` — alternate LLM backend
- `HUGGINGFACE_API_KEY` — for private HF repos

## Architecture

### Upstream `circuit_tracer/` package

The original library. Key abstractions:

- **`circuit_tracer/replacement_model/`** — wraps HuggingFace models with transcoders. `TransformerLens` backend (default) and `nnsight` backend (experimental).
- **`circuit_tracer/transcoder/`** — `CrossLayerTranscoder` and `SingleLayerTranscoder` implementations.
- **`circuit_tracer/attribution/`** — attribution algorithm computing direct effects between nodes (forward + backward passes).
- **`circuit_tracer/graph.py`** — `Graph` dataclass holding the adjacency matrix. Node ordering is always: `[active_features..., error_nodes..., embedding_tokens..., logit_tokens...]`. `adj[target, source]` convention.
- **`circuit_tracer/utils/create_graph_files.py`** — converts a `Graph` to frontend JSON for visualization.

### Summarization pipeline (`summarization/`)

This fork's main contribution. Pipeline stages:

1. **`summarization/attr_graph.py`** — `AttrGraph`: canonical representation shared by pruning and clustering. Loads from frontend JSON (`AttrGraph.from_graph_file`) or converts from `circuit_tracer.graph.Graph` (`AttrGraph.from_graph`). Adjacency convention: `adj[target, source]`.

2. **`summarization/graph_utils.py`** — core math for influence/relevance propagation. `compute_influence` propagates backward from logits via power iteration; `compute_relevance` propagates forward from embeddings. Also provides score normalization (`normalize_scores_min_max`, `normalize_scores_rank`) and score combination helpers (`combine_scores_geometric`, `combined_scores_arithmetic`, `combined_scores_harmonic`).

3. **`summarization/prune.py`** — `PruneGraph` + `prune_attr_graph()`: prunes nodes/edges using combined influence+relevance scores against thresholds. `PruneGraph` serializes via `torch.save`/`torch.load`. `LogitWeightMode` (`"probs"` or `"target"`) controls how logit nodes are weighted during influence computation.

4. **`summarization/cluster.py`** — `cluster_graph()`: clusters middle feature nodes into supernodes using spectral clustering on a cosine similarity matrix. `compute_similarity()` builds affinity from shared in/out neighbors weighted by edge scores. Enforces DAG constraints and `max_layer_span`.

5. **`summarization/cluster_scoring.py`** — composite quality metrics for a given clustering: silhouette score over middle nodes, DAG score, intra-cluster cohesion, attribution faithfulness. Consumed by `auto_grouping.py` to select k.

6. **`summarization/supernode_graph.py`** — `Node`, `Supernode`, `SummarizationGraph` dataclasses. `SummarizationGraph.sn_adj` is the supernode-level adjacency (same `[target, source]` convention). `node_from_prune_graph()` enriches a `Node` with influence/relevance scores from a `PruneGraph`.

7. **`summarization/flow_analysis.py`** — flow faithfulness metrics on the supernode graph.

8. **`summarization/auto_grouping.py`** — `find_best_k()`: estimates k range via normalized-Laplacian eigengap, sweeps candidates, and picks the highest-scoring k using `cluster_scoring.score_k`.

9. **`summarization/__main__.py`** — `run_pipeline()`: full orchestration (download → prune → cluster → score → optionally upload). All CLI flags are documented in `build_parser()`.

### Supporting modules

- **`summarization/utils.py`** — `get_data_from_json()` loads frontend JSON → `(adj, nodes, metadata)`; node classification helpers (`node_is_embedding`, `node_is_logit`, `node_is_fixed`); `layer_index_from_node` / `layer_index_from_node_id`.
- **`summarization/classify.py`** — LLM-free heuristic supernode labelling.
- **`summarization/token_attribution.py`** — per-token attribution utilities used by the prune evaluator.
- **`summarization/group_llm.py`** — LLM-based grouping (OpenAI / Gemini backends).
- **`api.py`** — thin Neuronpedia REST client (`get_feature`, `generate_graph`, `save_subgraph`).
- **`config.py`** — loads env vars from `.env`.
- **`eval/prune_graphs.py`** — mass-prune sweep over graph JSON files; writes `PruneGraph` `.pt` files and `manifest.json`. Must run before `eval/eval_prune.py`.
- **`eval/eval_prune.py`** — computes prune quality metrics from a manifest; outputs CSV.
- **`eval/eval_cluster.py`** — sweeps clustering methods across saved `PruneGraph` files; outputs summary CSV.
- **`generate_new_graphs.py`** / **`generate_shap_values.py`** — batch graph/SHAP generation scripts.
- **`streamlit_app.py`** — interactive Streamlit UI for the pipeline.

### Node ID conventions

- Feature nodes: `"{layer}_{feature_idx}_{ctx_position}"` (e.g., `"10_22605_3"`)
- Error nodes: `"0_{layer}_{ctx_position}"` (e.g., `"0_5_2"`)
- Embedding nodes: `"E_{vocab_id}_{ctx_position}"` (e.g., `"E_6037_0"`)
- Logit nodes: `"{n_layers+1}_{vocab_id}_{rank}"` (e.g., `"27_1234_0"`)

### Ruff rules

`typing.Union`, `typing.Optional`, `typing.Dict`, `typing.Tuple`, `typing.List` are banned — use the modern Python 3.10+ equivalents (`X | Y`, `X | None`, `dict`, `tuple`, `list`).
