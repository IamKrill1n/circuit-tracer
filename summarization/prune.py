# Unified pruning: AttrGraph -> PruneGraph; shared core with circuit_tracer.graph.prune_graph
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from api import get_feature
from circuit_tracer.graph import (
    combine_scores_geometric,
    combined_scores_arithmetic,
    combined_scores_harmonic,
    compute_edge_influence,
    compute_edge_relevance,
    compute_node_influence,
    compute_node_relevance,
    find_threshold,
    normalize_matrix,
)
from summarization.attr_graph import AttrGraph
from summarization.utils import _build_index_sets

logger = logging.getLogger(__name__)

LogitWeightMode = Literal["probs", "target"]


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
        if missing and "graph_scores" not in payload:
            payload = dict(payload)
            payload["graph_scores"] = 0.0
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
    torch.save(prune_graph.to_dict(), output_path)


def load_prune_graph(
    input_path: str,
    map_location: Optional[str | torch.device] = "cpu",
) -> PruneGraph:
    payload = torch.load(input_path, map_location=map_location)
    if isinstance(payload, PruneGraph):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload type in {input_path}: {type(payload)}")
    return PruneGraph.from_dict(payload)


def compute_combined_prune_graph_scores(
    prune_graph: PruneGraph,
    method: Literal["geometric", "arithmetic", "harmonic"] = "geometric",
    normalization: Literal["min_max", "rank"] = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
) -> tuple[float, float]:
    """Compute retention/completeness style scores directly from a saved PruneGraph."""
    node_influence = prune_graph.node_influence
    node_relevance = prune_graph.node_relevance
    if node_influence is None or node_relevance is None:
        raise ValueError("PruneGraph is missing node influence/relevance tensors.")

    ni = node_influence.to(dtype=torch.float32)
    nr = node_relevance.to(dtype=torch.float32)
    if method == "geometric":
        combined = combine_scores_geometric(ni, nr, normalization=normalization, alpha=alpha, eps=eps)
    elif method == "arithmetic":
        combined = combined_scores_arithmetic(ni, nr, normalization=normalization, alpha=alpha, eps=eps)
    else:
        combined = combined_scores_harmonic(ni, nr, normalization=normalization, alpha=alpha, eps=eps)

    # For saved pruned graphs, the full-graph denominator is tracked as graph_scores.
    combined_retention = float(prune_graph.graph_scores) if prune_graph.graph_scores is not None else float("nan")

    idx = _build_index_sets(prune_graph.kept_ids, prune_graph.attr)
    error_idx = idx["error"]
    pruned_norm = normalize_matrix(prune_graph.pruned_adj.clone())
    if error_idx:
        non_error_fractions = 1.0 - pruned_norm[:, error_idx].sum(dim=-1)
    else:
        non_error_fractions = torch.ones(pruned_norm.shape[0], dtype=pruned_norm.dtype, device=pruned_norm.device)

    denom = combined.sum().clamp(min=eps)
    combined_completeness_score = float(((non_error_fractions * combined).sum() / denom).item())
    return combined_retention, combined_completeness_score


def _validate_threshold(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_inputs(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    logit_weights: LogitWeightMode | None,
    token_weights: Optional[List[float]],
    logits_seed: torch.Tensor | None,
    emb_weights_seed: torch.Tensor | None,
) -> None:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D tensor, got shape={tuple(adj.shape)}")
    if adj.shape[0] != len(node_ids):
        raise ValueError(f"adj size and node_ids length mismatch: {adj.shape[0]} vs {len(node_ids)}")
    missing = [nid for nid in node_ids if nid not in attr]
    if missing:
        raise ValueError(f"attr missing entries for {len(missing)} node_ids")
    if logits_seed is None:
        if logit_weights not in ("probs", "target"):
            raise ValueError(f"logit_weights must be 'probs' or 'target', got {logit_weights}")
    if token_weights is not None and emb_weights_seed is None:
        if any(not isinstance(x, (int, float)) for x in token_weights):
            raise ValueError("token_weights must be a list of floats if provided")
    if logits_seed is not None and logits_seed.numel() != adj.shape[0]:
        raise ValueError(
            f"logits_seed length {logits_seed.numel()} != num_nodes {adj.shape[0]}"
        )
    if emb_weights_seed is not None and emb_weights_seed.numel() != adj.shape[0]:
        raise ValueError(
            f"emb_weights_seed length {emb_weights_seed.numel()} != num_nodes {adj.shape[0]}"
        )


def remove_dangling_nodes(
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    feature_idx: torch.Tensor,
    non_boundary: torch.Tensor,
) -> torch.Tensor:
    old = node_mask.clone()
    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
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
    logit_weights: LogitWeightMode | None = "target",
    token_weights: Optional[List[float]] = None,
    logits_seed: torch.Tensor | None = None,
    emb_weights_seed: torch.Tensor | None = None,
    node_influence_threshold: float = 0.8,
    node_relevance_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    n = adj.shape[0]
    idx = _build_index_sets(node_ids, attr)

    if logits_seed is not None:
        logits_seed_t = logits_seed.to(device=adj.device, dtype=torch.float32).reshape(n)
    else:
        logits_seed_t = torch.zeros(n, device=adj.device, dtype=torch.float32)
        if logit_weights == "probs":
            for i in idx["logit"]:
                nid = node_ids[i]
                logits_seed_t[i] = float(attr.get(nid, {}).get("token_prob", 0.0))
        else:
            if not idx["target_logit"]:
                raise ValueError("No target logit node found in graph attributes.")
            for i in idx["target_logit"]:
                logits_seed_t[i] = 1.0

    if emb_weights_seed is not None:
        emb_weights_t = emb_weights_seed.to(device=adj.device, dtype=torch.float32).reshape(n)
    else:
        emb_weights_t = torch.zeros(n, device=adj.device, dtype=torch.float32)
        emb_idx = idx["embedding"]
        if token_weights is None:
            n_emb = len(emb_idx)
            denom = max(n_emb, 1)
            for i in emb_idx:
                emb_weights_t[i] = 1.0 / denom
        else:
            if len(token_weights) != len(emb_idx):
                raise ValueError(
                    f"token_weights length ({len(token_weights)}) must equal number of embedding nodes ({len(emb_idx)})"
                )
            for k, i in enumerate(emb_idx):
                emb_weights_t[i] = float(token_weights[k])

    node_inf = compute_node_influence(adj, logits_seed_t)
    node_rel = compute_node_relevance(adj, emb_weights_t)

    full_graph_scores = torch.sum(node_inf[idx["feature"]] * node_rel[idx["feature"]]).item()

    node_inf_mask = node_inf >= find_threshold(node_inf, node_influence_threshold)
    node_rel_mask = node_rel >= find_threshold(node_rel, node_relevance_threshold)
    node_mask = (node_inf_mask & node_rel_mask).bool()

    if keep_all_tokens_and_logits:
        for i in idx["embedding"]:
            node_mask[i] = True
        for i in idx["logit"]:
            node_mask[i] = True
    else:
        for i in idx["target_logit"]:
            node_mask[i] = True

    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    edge_inf = compute_edge_influence(pruned, logits_seed_t)
    edge_rel = compute_edge_relevance(pruned, emb_weights_t)
    edge_inf_mask = edge_inf >= find_threshold(edge_inf.flatten(), edge_influence_threshold)
    edge_rel_mask = edge_rel >= find_threshold(edge_rel.flatten(), edge_relevance_threshold)
    edge_mask = (edge_inf_mask & edge_rel_mask).bool()

    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)
    node_mask = remove_dangling_nodes(node_mask, edge_mask, feature_idx, non_boundary)

    feature_node_mask = node_mask[idx["feature"]]
    if feature_node_mask.numel() == 0:
        pruned_graph_scores = 0.0
    else:
        inf_f = node_inf[idx["feature"]]
        rel_f = node_rel[idx["feature"]]
        pruned_graph_scores = torch.sum(inf_f[feature_node_mask] * rel_f[feature_node_mask]).item()

    denom = full_graph_scores if abs(full_graph_scores) > 1e-12 else 1.0
    ratio = pruned_graph_scores / denom

    return node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel, ratio


def prune_attr_graph(
    attr_graph: AttrGraph,
    logit_weights: LogitWeightMode | None = "target",
    token_weights: Optional[List[float]] = None,
    logits_seed: torch.Tensor | None = None,
    emb_weights_seed: torch.Tensor | None = None,
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
    """
    Prune from a canonical ``AttrGraph``.
    """
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

    nodes = attr_graph.nodes
    node_ids = [n.node_id for n in nodes]
    attr = {n.node_id: asdict(n) for n in nodes}
    adj = attr_graph.adj
    metadata = attr_graph.metadata

    _validate_inputs(adj, node_ids, attr, logit_weights, token_weights, logits_seed, emb_weights_seed)

    node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel, graph_scores = prune_combined(
        adj,
        node_ids,
        attr,
        logit_weights=logit_weights,
        token_weights=token_weights,
        logits_seed=logits_seed,
        emb_weights_seed=emb_weights_seed,
        node_influence_threshold=node_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_influence_threshold=edge_influence_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    kept_indices = node_mask.nonzero(as_tuple=True)[0]
    kept_ids = [node_ids[i] for i in kept_indices.tolist()]

    if filter_act_density:
        model_id = metadata.get("scan", "")
        info = metadata.get("info", {})
        source_set = info.get("neuronpedia_source_set") or (
            info.get("source_urls", [""])[0].split("/")[-1] if info.get("source_urls") else ""
        )
        for node_id in list(kept_ids):
            if attr[node_id].get("feature_type") == "embedding":
                ptoks = metadata.get("prompt_tokens", [])
                cidx = attr[node_id].get("ctx_idx", 0)
                try:
                    cidx = int(cidx)
                except (TypeError, ValueError):
                    cidx = 0
                if cidx < len(ptoks):
                    attr[node_id]["clerp"] = f"Emb: {ptoks[cidx]}"
                continue
            if attr[node_id].get("feature_type") != "cross layer transcoder":
                continue

            layer, index = node_id.split("_")[:2]
            index = int(index)
            layer = layer + "-" + source_set
            status, data = get_feature(modelId=model_id, layer=layer, index=index)
            if status != 200:
                logger.warning(
                    "Failed node=%s modelId=%s layer=%s status=%s",
                    node_id,
                    model_id,
                    layer,
                    status,
                )
                continue

            json_data = json.loads(data)
            explanations = json_data.get("explanations", [])
            clerp = ""
            if isinstance(explanations, list) and explanations:
                first_explanation = explanations[0]
                if isinstance(first_explanation, dict):
                    clerp = first_explanation.get("description", "")
            act_density = json_data.get("frac_nonzero", 0)
            if attr[node_id].get("clerp", "") == "":
                attr[node_id]["clerp"] = clerp
            if act_density > act_density_ub or act_density < act_density_lb:
                idx_local = node_ids.index(node_id)
                node_mask[idx_local] = False
                edge_mask[idx_local, :] = False
                edge_mask[:, idx_local] = False

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
    ag = AttrGraph.from_graph_file(json_path)
    return prune_attr_graph(
        ag,
        logit_weights=logit_weights,
        token_weights=token_weights,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        node_influence_threshold=node_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_influence_threshold=edge_influence_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
        filter_act_density=filter_act_density,
        act_density_lb=act_density_lb,
        act_density_ub=act_density_ub,
    )


def prune_masks_from_attr_graph(
    attr_graph: AttrGraph,
    *,
    token_weights: Optional[torch.Tensor] = None,
    logit_weights: Optional[torch.Tensor] = None,
    logit_weights_mode: LogitWeightMode | None = "probs",
    token_weights_list: Optional[List[float]] = None,
    node_influence_threshold: float = 0.8,
    node_relevance_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Shared pruning step returning masks and score tensors (used by ``circuit_tracer.graph.prune_graph``)."""
    adj = attr_graph.adj
    nodes = attr_graph.nodes
    node_ids = [n.node_id for n in nodes]
    attr = {n.node_id: asdict(n) for n in nodes}

    logits_seed = logit_weights
    emb_seed = token_weights
    lw_mode = None if logits_seed is not None else logit_weights_mode
    tw_list = None if emb_seed is not None else token_weights_list

    _validate_inputs(adj, node_ids, attr, lw_mode, tw_list, logits_seed, emb_seed)

    return prune_combined(
        adj,
        node_ids,
        attr,
        logit_weights=lw_mode,
        token_weights=tw_list,
        logits_seed=logits_seed,
        emb_weights_seed=emb_seed,
        node_influence_threshold=node_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_influence_threshold=edge_influence_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )


if __name__ == "__main__":
    prune_graph = prune_graph_pipeline(
        json_path="demos/temp_graph_files/austin_clt.json",
        logit_weights="target",
        token_weights=[0, 0, 0, 0, 1 / 3, 0, 0, 1 / 3, 0, 1 / 3, 0],
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
