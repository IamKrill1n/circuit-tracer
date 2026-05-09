# Unified pruning: AttrGraph -> PruneGraph
import json
import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch

from api import get_feature
from summarization.graph_utils import (
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
from summarization.supernode_graph import Node
from summarization.utils import _build_index_sets, _node_from_json_dict

logger = logging.getLogger(__name__)

LogitWeightMode = Literal["probs", "target"]
CombineMethod = Literal["geometric", "arithmetic", "harmonic"]
NormalizationMethod = Literal["min_max", "rank"]


def _combine_scores(
    method: CombineMethod,
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: NormalizationMethod,
    alpha: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    if method == "geometric":
        return combine_scores_geometric(influence, relevance, normalization=normalization, alpha=alpha, eps=eps)
    if method == "arithmetic":
        return combined_scores_arithmetic(influence, relevance, normalization=normalization, alpha=alpha, eps=eps)
    if method == "harmonic":
        return combined_scores_harmonic(influence, relevance, normalization=normalization, alpha=alpha, eps=eps)
    raise ValueError(f"Invalid combine_method: {method}")


def _nodes_from_payload(raw_nodes: Any) -> list[Node]:
    if not isinstance(raw_nodes, list):
        raise TypeError(f"nodes must be a list, got {type(raw_nodes)}")
    out: list[Node] = []
    for item in raw_nodes:
        if isinstance(item, Node):
            out.append(item)
        elif isinstance(item, dict):
            out.append(_node_from_json_dict(item))
        else:
            raise TypeError(f"invalid node entry: {type(item)}")
    for idx, node in enumerate(out):
        if getattr(node, "node_idx", -1) < 0:
            node.set_node_idx(idx)
    return out


@dataclass
class PruneGraph:
    nodes: list[Node]
    pruned_adj: torch.Tensor
    metadata: dict[str, Any]
    node_influence: torch.Tensor | None = None
    node_relevance: torch.Tensor | None = None
    edge_influence: torch.Tensor | None = None
    edge_relevance: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return int((self.pruned_adj != 0).sum().item())

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "pruned_adj": self.pruned_adj,
            "node_influence": self.node_influence,
            "node_relevance": self.node_relevance,
            "edge_influence": self.edge_influence,
            "edge_relevance": self.edge_relevance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PruneGraph":
        required = {
            "nodes",
            "pruned_adj",
            "metadata",
        }
        missing = required - set(payload.keys())
        if missing:
            raise ValueError(f"Invalid PruneGraph payload. Missing keys: {sorted(missing)}")
        nodes = _nodes_from_payload(payload["nodes"])
        return cls(
            nodes=nodes,
            pruned_adj=payload["pruned_adj"],
            node_influence=payload.get("node_influence"),
            node_relevance=payload.get("node_relevance"),
            edge_influence=payload.get("edge_influence"),
            edge_relevance=payload.get("edge_relevance"),
            metadata=payload["metadata"],
        )


def save_prune_graph(prune_graph: PruneGraph, output_path: str) -> None:
    torch.save(prune_graph.to_dict(), output_path)


def load_prune_graph(
    input_path: str,
    map_location: str | torch.device | None = "cpu",
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
) -> float:
    """Compute completeness style score directly from a saved PruneGraph."""
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

    idx = _build_index_sets(prune_graph.nodes)
    error_idx = idx["error"]
    pruned_norm = normalize_matrix(prune_graph.pruned_adj.clone())
    if error_idx:
        non_error_fractions = 1.0 - pruned_norm[:, error_idx].sum(dim=-1)
    else:
        non_error_fractions = torch.ones(pruned_norm.shape[0], dtype=pruned_norm.dtype, device=pruned_norm.device)

    denom = combined.sum().clamp(min=eps)
    combined_completeness_score = float(((non_error_fractions * combined).sum() / denom).item())
    return combined_completeness_score


def _validate_threshold(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_inputs(
    adj: torch.Tensor,
    nodes: list[Node],
    logit_weights: LogitWeightMode | None,
    token_weights: list[float] | None,
    logits_seed: torch.Tensor | None,
    emb_weights_seed: torch.Tensor | None,
) -> None:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D tensor, got shape={tuple(adj.shape)}")
    if adj.shape[0] != len(nodes):
        raise ValueError(f"adj size and nodes length mismatch: {adj.shape[0]} vs {len(nodes)}")
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
        if non_boundary.numel() > 0:
            node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
        if feature_idx.numel() > 0:
            node_mask[feature_idx] &= edge_mask[feature_idx].any(1)
    return node_mask


def prune_combined(
    adj: torch.Tensor,
    nodes: list[Node],
    logit_weights: LogitWeightMode | None = "target",
    token_weights: list[float] | None = None,
    logits_seed: torch.Tensor | None = None,
    emb_weights_seed: torch.Tensor | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    combine_method: CombineMethod = "geometric",
    normalization: NormalizationMethod = "rank",
    alpha: float = 0.5,
    keep_all_tokens_and_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = adj.shape[0]
    idx = _build_index_sets(nodes)

    if logits_seed is not None:
        logits_seed_t = logits_seed.to(device=adj.device, dtype=torch.float32).reshape(n)
    else:
        logits_seed_t = torch.zeros(n, device=adj.device, dtype=torch.float32)
        if logit_weights == "probs":
            for i in idx["logit"]:
                logits_seed_t[i] = float(nodes[i].token_prob)
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

    node_score = _combine_scores(combine_method, node_inf, node_rel, normalization, alpha)
    node_mask = (node_score >= find_threshold(node_score, node_threshold)).bool()

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
    # Flatten before combining: rank-normalization needs a single 1D index space across all edges.
    edge_score_flat = _combine_scores(
        combine_method, edge_inf.flatten(), edge_rel.flatten(), normalization, alpha
    )
    edge_score = edge_score_flat.reshape(edge_inf.shape)
    edge_mask = (edge_score >= find_threshold(edge_score_flat, edge_threshold)).bool()

    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)
    node_mask = remove_dangling_nodes(node_mask, edge_mask, feature_idx, non_boundary)

    return node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel


def prune_attr_graph(
    attr_graph: AttrGraph,
    logit_weights: LogitWeightMode | None = "target",
    token_weights: list[float] | None = None,
    logits_seed: torch.Tensor | None = None,
    emb_weights_seed: torch.Tensor | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    combine_method: CombineMethod = "geometric",
    normalization: NormalizationMethod = "rank",
    alpha: float = 0.5,
    keep_all_tokens_and_logits: bool = True,
    filter_act_density: bool = False,
    act_density_lb: float = 2e-5,
    act_density_ub: float = 0.1,
) -> PruneGraph:
    """
    Prune from a canonical ``AttrGraph``.
    """
    _validate_threshold("node_threshold", node_threshold)
    _validate_threshold("edge_threshold", edge_threshold)

    nodes = [replace(n) for n in attr_graph.nodes]
    node_ids = [n.node_id for n in nodes]
    id_index = {nid: i for i, nid in enumerate(node_ids)}
    adj = attr_graph.adj
    metadata = attr_graph.metadata

    _validate_inputs(adj, nodes, logit_weights, token_weights, logits_seed, emb_weights_seed)

    node_mask, edge_mask, node_inf, node_rel, edge_inf, edge_rel = prune_combined(
        adj,
        nodes,
        logit_weights=logit_weights,
        token_weights=token_weights,
        logits_seed=logits_seed,
        emb_weights_seed=emb_weights_seed,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        combine_method=combine_method,
        normalization=normalization,
        alpha=alpha,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    kept_indices = node_mask.nonzero(as_tuple=True)[0]

    if filter_act_density:
        model_id = metadata.get("scan", "")
        info = metadata.get("info", {})
        source_set = info.get("neuronpedia_source_set") or (
            info.get("source_urls", [""])[0].split("/")[-1] if info.get("source_urls") else ""
        )
        for i in kept_indices.tolist():
            node = nodes[i]
            nid = node.node_id
            if node.feature_type == "embedding":
                ptoks = metadata.get("prompt_tokens", [])
                cidx = node.ctx_idx
                try:
                    cidx = int(cidx)
                except (TypeError, ValueError):
                    cidx = 0
                if cidx < len(ptoks):
                    nodes[i] = replace(node, clerp=f"Emb: {ptoks[cidx]}")
                continue
            if node.feature_type != "cross layer transcoder":
                continue

            layer, index = nid.split("_")[:2]
            index = int(index)
            layer = layer + "-" + source_set
            status, data = get_feature(modelId=model_id, layer=layer, index=index)
            if status != 200:
                logger.warning(
                    "Failed node=%s modelId=%s layer=%s status=%s",
                    nid,
                    model_id,
                    layer,
                    status,
                )
                continue

            json_data = json.loads(data)
            explanations = json_data.get("explanations", [])
            clerp = ""
            if explanations:
                clerp = explanations[0].get("description", "")
            act_density = json_data.get("frac_nonzero", 0)
            cur = nodes[i]
            if cur.clerp == "":
                nodes[i] = replace(cur, clerp=clerp)
            if act_density > act_density_ub or act_density < act_density_lb:
                idx_local = id_index[nid]
                node_mask[idx_local] = False
                edge_mask[idx_local, :] = False
                edge_mask[:, idx_local] = False

        idx2 = _build_index_sets(nodes)
        feature_idx = torch.tensor(idx2["feature"], dtype=torch.long, device=adj.device)
        non_boundary = torch.tensor(idx2["feature"] + idx2["error"], dtype=torch.long, device=adj.device)
        node_mask = remove_dangling_nodes(node_mask, edge_mask, feature_idx, non_boundary)

        kept_indices = node_mask.nonzero(as_tuple=True)[0]

    pruned_adj = adj[kept_indices][:, kept_indices].clone()
    kept_edge_mask = edge_mask[kept_indices][:, kept_indices]
    pruned_adj[~kept_edge_mask] = 0.0
    kept_node_inf = node_inf[kept_indices]
    kept_node_rel = node_rel[kept_indices]
    kept_edge_inf = edge_inf[kept_indices][:, kept_indices]
    kept_edge_rel = edge_rel[kept_indices][:, kept_indices]
    kept_edge_inf[~kept_edge_mask] = 0.0
    kept_edge_rel[~kept_edge_mask] = 0.0

    kept_nodes = [replace(nodes[int(j)], node_idx=i) for i, j in enumerate(kept_indices.tolist())]
    logger.info("Pruned graph: %d nodes, %d edges", len(kept_nodes), int((pruned_adj != 0).sum().item()))

    return PruneGraph(
        kept_nodes,
        pruned_adj,
        metadata,
        kept_node_inf,
        kept_node_rel,
        kept_edge_inf,
        kept_edge_rel,
    )


def prune_graph_pipeline(
    json_path: str,
    logit_weights: LogitWeightMode,
    token_weights: list[float] | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    combine_method: CombineMethod = "geometric",
    normalization: NormalizationMethod = "rank",
    alpha: float = 0.5,
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
        combine_method=combine_method,
        normalization=normalization,
        alpha=alpha,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
        filter_act_density=filter_act_density,
        act_density_lb=act_density_lb,
        act_density_ub=act_density_ub,
    )


def prune_masks_from_attr_graph(
    attr_graph: AttrGraph,
    *,
    token_weights: torch.Tensor | None = None,
    logit_weights: torch.Tensor | None = None,
    logit_weights_mode: LogitWeightMode | None = "probs",
    token_weights_list: list[float] | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    combine_method: CombineMethod = "geometric",
    normalization: NormalizationMethod = "rank",
    alpha: float = 0.5,
    keep_all_tokens_and_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared pruning step returning masks and score tensors (used by ``circuit_tracer.graph.prune_graph``)."""
    adj = attr_graph.adj
    nodes = attr_graph.nodes

    logits_seed = logit_weights
    emb_seed = token_weights
    lw_mode = None if logits_seed is not None else logit_weights_mode
    tw_list = None if emb_seed is not None else token_weights_list

    _validate_inputs(adj, nodes, lw_mode, tw_list, logits_seed, emb_seed)

    return prune_combined(
        adj,
        nodes,
        logit_weights=lw_mode,
        token_weights=tw_list,
        logits_seed=logits_seed,
        emb_weights_seed=emb_seed,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        combine_method=combine_method,
        normalization=normalization,
        alpha=alpha,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )


if __name__ == "__main__":
    prune_graph = prune_graph_pipeline(
        json_path="demos/temp_graph_files/austin_clt.json",
        logit_weights="target",
        token_weights=[0, 0, 0, 0, 1 / 3, 0, 0, 1 / 3, 0, 1 / 3, 0],
        node_threshold=1,
        edge_threshold=1,
        keep_all_tokens_and_logits=False,
    )

    print(prune_graph.num_nodes)
    print(prune_graph.num_edges)
    print(prune_graph.node_influence)
    print(prune_graph.node_relevance)
    print(prune_graph.edge_influence)
    print(prune_graph.edge_relevance)
