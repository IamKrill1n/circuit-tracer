import os
import json

from circuit_tracer.subgraph.prune import prune_graph_pipeline
from circuit_tracer.subgraph.api import save_subgraph
from circuit_tracer.subgraph.utils import get_clerp


def generate_subgraphs(
    graph_path: str,
    logit_weights: str = "target",
    token_weights=None,
    node_influence_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    node_relevance_threshold: float = 0.8,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = False,
    include_clerp: bool = False,
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
        node_influence_threshold:    cumulative influence fraction to keep
        edge_influence_threshold:    cumulative edge-influence fraction to keep
        node_relevance_threshold:    cumulative relevance fraction to keep
        edge_relevance_threshold:    cumulative edge-relevance fraction to keep
        keep_all_tokens_and_logits:  keep all embeddings/logits or only target logit
        include_clerp:               fetch clerp descriptions from API
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
        node_influence_threshold=node_influence_threshold,
        edge_influence_threshold=edge_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    print(f"Subgraph has {len(kept_ids)} nodes, {int((pruned_adj != 0).sum())} edges")

    # 2. Optionally fetch clerp descriptions
    if include_clerp:
        get_clerp(metadata, attr)

    # 3. Optionally save to Neuronpedia
    if save:
        _model_id = model_id or metadata.get("model_id", "")
        _slug = slug or metadata.get("slug", "")
        _display_name = display_name or f"subgraph-{os.path.basename(graph_path).split('.')[0]}"

        if not _model_id or not _slug:
            print("Warning: model_id and slug are required to save. Skipping save.")
        else:
            status, response = save_subgraph(
                modelId=_model_id,
                slug=_slug,
                displayName=_display_name,
                pinnedIds=kept_ids,
                pruningThreshold=node_influence_threshold,
                densityThreshold=edge_influence_threshold,
            )
            print(f"Save status: {status}")
            print(response)

    return kept_ids, pruned_adj, attr, metadata


if __name__ == "__main__":
    token_weights = [0, 0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0]

    kept_ids, pruned_adj, attr, metadata = generate_subgraphs(
        graph_path="demos/temp_graph_files/austin.json",
        logit_weights="target",
        token_weights=token_weights,
        node_influence_threshold=0.7,
        edge_influence_threshold=0.7,
        node_relevance_threshold=0.7,
        edge_relevance_threshold=0.7,
        keep_all_tokens_and_logits=True,
        include_clerp=False,
        save=True,
    )

    print(f"\nKept {len(kept_ids)} nodes:")
    for nid in kept_ids[:15]:
        clerp = attr.get(nid, {}).get("clerp", "")
        print(f"  {nid}: {clerp[:60]}")