import os
import json

from circuit_tracer.subgraph.prune import prune_graph_pipeline
from circuit_tracer.subgraph.api import save_subgraph
from circuit_tracer.subgraph.utils import get_clerp
# from dotenv import load_dotenv
# load_dotenv()

def generate_subgraphs(
    graph_path: str,
    logit_weights: str = "target",
    token_weights=None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = False,
    save: bool = False,
    model_id: str = "",
    slug: str = "",
    display_name: str = "",
):
    """
    Generate a subgraph from a graph JSON file.

    Args:
        graph_path:                  path to the graph JSON file
        logit_weights:               "target" or "probs"
        token_weights:               per-token relevance weights; None => uniform
        node_threshold:              cumulative node-score fraction to keep
        edge_threshold:              cumulative edge-score fraction to keep
        keep_all_tokens_and_logits:  keep all embeddings/logits or only target logit
        save:                        upload subgraph to Neuronpedia
        model_id:                    model identifier for save (e.g. "gemma-2-2b")
        slug:                        parent graph slug for save
        display_name:                display name for saved subgraph
    """
    # 1. Prune
    kept_ids, pruned_adj, attr, metadata = prune_graph_pipeline(
        json_path=graph_path,
        logit_weights=logit_weights,
        token_weights=token_weights,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    print(f"Subgraph has {len(kept_ids)} nodes, {int((pruned_adj != 0).sum())} edges")

    # 2. Optionally save to Neuronpedia
    if save:
        _model_id = model_id or metadata.get("model_id", "")
        _slug = slug or metadata.get("slug", "")
        _display_name = display_name or f"subgraph-{os.path.basename(graph_path).split('.')[0]}"

        print(f"{_model_id}, {_slug}, {_display_name}'")
        if not _model_id or not _slug:
            print("Warning: model_id and slug are required to save. Skipping save.")
        else:
            status, response = save_subgraph(
                modelId=_model_id,
                slug=_slug,
                displayName=_display_name,
                pinnedIds=kept_ids,
                pruningThreshold=node_threshold,
                densityThreshold=edge_threshold,
            )
            print(f"Save status: {status}")
            print(response)

    return kept_ids, pruned_adj, attr, metadata


if __name__ == "__main__":
    token_weights = [0.00198786, 0.03153391, 0.00083086, 0.01473883, 0.22338926, 0.00649094,
  0.00222269, 0.01996207, 0.0052309, 0.67869559, 0.01491708]
    
    kept_ids, pruned_adj, attr, metadata = generate_subgraphs(
        graph_path="demos/temp_graph_files/austin.json",
        logit_weights="target",
        token_weights=token_weights,
        node_threshold=0.6,
        edge_threshold=0.95,
        keep_all_tokens_and_logits=False,
        model_id="gemma-2-2b",
        display_name="Austin Subgraph",
        save=True,
    )

    print(f"\nKept {len(kept_ids)} nodes:")
    for nid in kept_ids:
        clerp = attr.get(nid, {}).get("clerp", "")
        print(f"  {nid}: {clerp[:60]}")