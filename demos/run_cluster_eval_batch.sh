#!/usr/bin/env bash
# Run clustering evaluation on all .pt pruned graphs under demos/subgraph/clt
# and demos/subgraph/gemma-scope-16k. Writes one JSON per graph next to the file.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "Skip (missing): $dir"
    return 0
  fi
  shopt -s nullglob
  local f
  for f in "$dir"/*.pt; do
    base="$(basename "$f" .pt)"
    out="${dir}/${base}_cluster_eval.json"
    echo "==> $f -> $out"
    python3 demos/run_cluster_eval.py --graph "$f" --out-json "$out"
  done
  shopt -u nullglob
}

run_dir "demos/subgraph/clt"
run_dir "demos/subgraph/gemma-scope-16k"
