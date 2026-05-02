# graph.prune_graph for json files
import logging
from typing import Any, Dict, List, Tuple, Optional, Literal, NamedTuple
from summarization.utils import _build_index_sets
from api import get_feature
from dataclasses import dataclass

import torch
import json
from circuit_tracer.graph import (
    compute_node_influence,
    compute_node_relevance,
    compute_edge_influence,
    compute_edge_relevance,
    find_threshold,
)

from summarization.utils import get_data_from_json
logger = logging.getLogger(__name__)

LogitWeightMode = Literal["probs", "target"]
class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative scores for each node

@dataclass
class PruneGraph:
    kept_ids: List[str]
    pruned_adj: torch.Tensor
    attr: Dict[str, Any]
    metadata: Dict[str, Any]
    node_influence: torch.Tensor | None = None
    node_relevance: torch.Tensor | None = None
    edge_influence: torch.Tensor | None = None
    edge_relevance: torch.Tensor | None = None
    graph_scores: float | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.kept_ids)

    @property
    def num_edges(self) -> int:
        return int((self.pruned_adj != 0).sum().item())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept_ids": self.kept_ids,
            "pruned_adj": self.pruned_adj,
            "node_influence": self.node_influence,
            "node_relevance": self.node_relevance,
            "edge_influence": self.edge_influence,
            "edge_relevance": self.edge_relevance,
            "attr": self.attr,
            "metadata": self.metadata,
            "graph_scores": self.graph_scores,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PruneGraph":
        required = {
            "kept_ids",
            "pruned_adj",
            "node_influence",
            "node_relevance",
            "edge_influence",
            "edge_relevance",
            "attr",
            "metadata",
            "graph_scores",
        }
        missing = required - set(payload.keys())
        if missing:
            raise ValueError(f"Invalid PruneGraph payload. Missing keys: {sorted(missing)}")
        return cls(
            kept_ids=payload["kept_ids"],
            pruned_adj=payload["pruned_adj"],
            node_influence=payload.get("node_influence"),
            node_relevance=payload.get("node_relevance"),
            edge_influence=payload.get("edge_influence"),
            edge_relevance=payload.get("edge_relevance"),
            attr=payload["attr"],
            metadata=payload["metadata"],
            graph_scores=payload.get("graph_scores"),
        )


def save_prune_graph(prune_graph: PruneGraph, output_path: str) -> None:
    """
    Save a PruneGraph to a .pt file.

    Args:
        prune_graph: PruneGraph object to serialize.
        output_path: Destination path (typically ending with .pt).
    """
    torch.save(prune_graph.to_dict(), output_path)


def load_prune_graph(
    input_path: str,
    map_location: Optional[str | torch.device] = "cpu",
) -> PruneGraph:
    """
    Load a PruneGraph from a .pt file produced by `save_prune_graph`.

    Args:
        input_path: Path to the .pt file.
        map_location: Device mapping passed to torch.load.
    """
    payload = torch.load(input_path, map_location=map_location)
    if isinstance(payload, PruneGraph):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload type in {input_path}: {type(payload)}")
    return PruneGraph.from_dict(payload)



def _validate_threshold(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_inputs(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    logit_weights: LogitWeightMode,
    token_weights: Optional[List[float]],
) -> None:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D tensor, got shape={tuple(adj.shape)}")
    if adj.shape[0] != len(node_ids):
        raise ValueError(
            f"adj size and node_ids length mismatch: {adj.shape[0]} vs {len(node_ids)}"
        )
    missing = [nid for nid in node_ids if nid not in attr]
    if missing:
        raise ValueError(f"attr missing entries for {len(missing)} node_ids")
    if logit_weights not in ("probs", "target"):
        raise ValueError(f"logit_weights must be 'probs' or 'target', got {logit_weights}")
    if token_weights is not None and any((not isinstance(x, (int, float))) for x in token_weights):
        raise ValueError("token_weights must be a list of floats if provided")


def remove_dangling_nodes(node_mask: torch.Tensor, edge_mask: torch.Tensor, feature_idx: torch.Tensor, non_boundary: torch.Tensor) -> torch.Tensor:
    # Iteratively remove nodes that become dangling after edge removals.
    old = node_mask.clone()
    while not torch.all(node_mask == old):
        old[:] = node_mask
        # remove edges touching removed nodes
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        # features that are non-boundary should have at least one incoming edge
        if feature_idx.numel() > 0 and non_boundary.numel() > 0:
            node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
            node_mask[feature_idx] &= edge_mask[feature_idx].any(1)
        else:
            if non_boundary.numel() > 0:
                node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
            if feature_idx.numel() > 0:
                node_mask[feature_idx] &= edge_mask[feature_idx].any(1)
    return node_mask

def prune_combined(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    logit_weights: LogitWeightMode,
    token_weights: Optional[List[float]] = None,
    node_influence_threshold: float = 0.8,
    node_relevance_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    n = adj.shape[0]
    idx = _build_index_sets(node_ids, attr)

    logits_seed = torch.zeros(n, device=adj.device)
    if logit_weights == "probs":
        for i in idx["logit"]:
            nid = node_ids[i]
            logits_seed[i] = float(attr.get(nid, {}).get("token_prob", 0.0))
    else:  # "target"
        if not idx["target_logit"]:
            raise ValueError("No target logit node found in graph attributes.")
        for i in idx["target_logit"]:
            logits_seed[i] = 1.0

    emb_weights = torch.zeros(n, device=adj.device)
    emb_idx = idx["embedding"]
    if token_weights is None:
        n_emb = len(emb_idx)
        for i in emb_idx:
            emb_weights[i] = 1.0 / n_emb
    else:
        if len(token_weights) != len(emb_idx):
            raise ValueError(
                f"token_weights length ({len(token_weights)}) must equal number of embedding nodes ({len(emb_idx)})"
            )
        for k, i in enumerate(emb_idx):
            emb_weights[i] = float(token_weights[k])

    node_inf = compute_node_influence(adj, logits_seed)
    node_rel = compute_node_relevance(adj, emb_weights)

    full_graph_scores = torch.sum(node_inf[idx["feature"]] * node_rel[idx["feature"]]).item()

    node_inf_mask = node_inf >= find_threshold(node_inf, node_influence_threshold)
    node_rel_mask = node_rel >= find_threshold(node_rel, node_relevance_threshold)
    # Final node selection keeps only nodes that are both influential and relevant.
    node_mask = (node_inf_mask & node_rel_mask).bool()

    if keep_all_tokens_and_logits:
        for i in emb_idx:
            node_mask[i] = True
        for i in idx["logit"]:
            node_mask[i] = True
    else:
        for i in idx["target_logit"]:
            node_mask[i] = True

    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    edge_inf = compute_edge_influence(pruned, logits_seed)
    edge_rel = compute_edge_relevance(pruned, emb_weights)
    edge_inf_mask = edge_inf >= find_threshold(edge_inf.flatten(), edge_influence_threshold)
    edge_rel_mask = edge_rel >= find_threshold(edge_rel.flatten(), edge_relevance_threshold)
    # Final edge selection keeps only edges that are both influential and relevant.
    edge_mask = (edge_inf_mask & edge_rel_mask).bool()

    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)
    # Use helper to iteratively remove dangling nodes after combining masks
    node_mask = remove_dangling_nodes(node_mask, edge_mask, feature_idx, non_boundary)
    # only consider feature nodes for graph scores
    feature_node_mask = node_mask[idx["feature"]]
    if feature_node_mask.numel() == 0:
        pruned_graph_scores = 0.0
    else:
        inf_f = node_inf[idx["feature"]]
        rel_f = node_rel[idx["feature"]]
        pruned_graph_scores = torch.sum(inf_f[feature_node_mask] * rel_f[feature_node_mask]).item()

    return node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel, pruned_graph_scores/full_graph_scores


def prune_graph_pipeline(
    json_path: str,
    logit_weights: LogitWeightMode,
    token_weights: Optional[List[float]] = None,
    node_threshold: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    node_influence_threshold: float = 0.8,
    node_relevance_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
    filter_act_density: bool = False,
    act_density_lb: float = 2e-5,
    act_density_ub: float = 0.1,
) -> PruneGraph:
    # Backward compatibility: if shared thresholds are provided, use them for both.
    if node_threshold is not None:
        node_influence_threshold = node_threshold
        node_relevance_threshold = node_threshold
    if edge_threshold is not None:
        edge_influence_threshold = edge_threshold
        edge_relevance_threshold = edge_threshold

    _validate_threshold("node_influence_threshold", node_influence_threshold)
    _validate_threshold("node_relevance_threshold", node_relevance_threshold)
    _validate_threshold("edge_influence_threshold", edge_influence_threshold)
    _validate_threshold("edge_relevance_threshold", edge_relevance_threshold)

    adj, node_ids, attr, metadata = get_data_from_json(json_path)
    _validate_inputs(adj, node_ids, attr, logit_weights, token_weights)

    node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel, graph_scores = prune_combined(
        adj, node_ids, attr,
        logit_weights=logit_weights,
        token_weights=token_weights,
        node_influence_threshold=node_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_influence_threshold=edge_influence_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    kept_indices = node_mask.nonzero(as_tuple=True)[0]
    kept_ids = [node_ids[i] for i in kept_indices.tolist()]

    
    if filter_act_density:
        modelId = metadata.get("scan", "")
        info = metadata.get("info", {})
        source_set = info.get("neuronpedia_source_set") or info.get("source_urls", [""])[0].split("/")[-1]
        for node_id in kept_ids:
            if attr[node_id].get("feature_type") != 'cross layer transcoder':
                continue

            # node_ids are in the format {layer}_{node_id}_{ctx_id}
            layer, index = node_id.split("_")[:2]
            index = int(index)
            layer = layer + "-" + source_set
            # print(layer)
            status, data = get_feature(modelId=modelId, layer=layer, index=index)
            if status != 200:
                print(f"Failed node={node_id} modelId={modelId} layer={layer} status={status} body={data[:200]}")
                continue

            json_data = json.loads(data)
            clerp = json_data.get("explanations", [])[0].get("description", "") 
            act_density = json_data.get("frac_nonzero", 0)
            if attr[node_id].get("clerp", "") == "":
                attr[node_id]['clerp'] = clerp
            # remove too frequently activated features
            if act_density > act_density_ub or act_density < act_density_lb:
                idx = node_ids.index(node_id)
                node_mask[idx] = False
                edge_mask[idx, :] = False
                edge_mask[:, idx] = False

    # after removing nodes due to activation density, remove any newly-dangling nodes
    idx2 = _build_index_sets(node_ids, attr)
    feature_idx = torch.tensor(idx2["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx2["feature"] + idx2["error"], dtype=torch.long, device=adj.device)
    node_mask = remove_dangling_nodes(node_mask, edge_mask, feature_idx, non_boundary)

    kept_indices = node_mask.nonzero(as_tuple=True)[0]
    kept_ids = [node_ids[i] for i in kept_indices.tolist()]

    pruned_adj = adj[kept_indices][:, kept_indices].clone()
    kept_edge_mask = edge_mask[kept_indices][:, kept_indices]
    pruned_adj[~kept_edge_mask] = 0.0
    kept_node_inf = node_inf[kept_indices]
    kept_node_rel = node_rel[kept_indices]
    kept_edge_inf = edge_inf[kept_indices][:, kept_indices]
    kept_edge_rel = edge_rel[kept_indices][:, kept_indices]
    kept_edge_inf[~kept_edge_mask] = 0.0
    kept_edge_rel[~kept_edge_mask] = 0.0

    out_attr = {nid: attr[nid] for nid in kept_ids}
    logger.info("Pruned graph: %d nodes, %d edges", len(kept_ids), int((pruned_adj != 0).sum().item()))

    return PruneGraph(
        kept_ids,
        pruned_adj,
        out_attr,
        metadata,
        kept_node_inf,
        kept_node_rel,
        kept_edge_inf,
        kept_edge_rel,
        graph_scores,
    )


if __name__ == "__main__":

    prune_graph = prune_graph_pipeline(
        json_path="demos/temp_graph_files/austin_clt.json",
        logit_weights='target',
        token_weights=[0, 0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0],
        node_influence_threshold=1,
        node_relevance_threshold=1,
        edge_influence_threshold=1,
        edge_relevance_threshold=1,
        keep_all_tokens_and_logits=False,
    )
    
    print(prune_graph.num_nodes)
    print(prune_graph.num_edges)
    print(prune_graph.graph_scores)
    print(prune_graph.node_influence)
    print(prune_graph.node_relevance)
    print(prune_graph.edge_influence)
    print(prune_graph.edge_relevance)