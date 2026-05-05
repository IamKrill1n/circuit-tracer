"""First-class supernode and summarization (cluster) graph types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SupernodeType = Literal["emb", "features", "logit"]


@dataclass
class Node:
    """Summarization-node view aligned with frontend node fields + relevance."""

    node_id: str
    feature: int
    layer: str
    ctx_idx: int
    feature_type: str
    token_prob: float = 0.0
    is_target_logit: bool = False
    run_idx: int = 0
    reverse_ctx_idx: int = 0
    jsNodeId: str = ""
    clerp: str = ""
    influence: float | None = None
    activation: float | None = None
    relevance: float | None = None


def _tensor_value_at(values: Any, idx: int) -> float | None:
    if values is None:
        return None
    try:
        raw = values[idx]
    except (IndexError, TypeError, KeyError):
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().item()
    return float(raw)


def node_from_prune_graph(
    prune_graph: Any,
    node_id: str,
    id_to_idx: dict[str, int] | None = None,
) -> Node:
    """Build a typed summarization node from `PruneGraph.attr` plus score tensors."""
    attr = prune_graph.attr.get(node_id, {})
    if id_to_idx is None:
        id_to_idx = {nid: i for i, nid in enumerate(prune_graph.kept_ids)}
    idx = id_to_idx.get(node_id)
    influence = _tensor_value_at(prune_graph.node_influence, idx) if idx is not None else None
    relevance = _tensor_value_at(prune_graph.node_relevance, idx) if idx is not None else None

    layer = attr.get("layer", "")
    feature = attr.get("feature", 0)
    ctx_idx = attr.get("ctx_idx", 0)
    run_idx = attr.get("run_idx", 0)
    reverse_ctx_idx = attr.get("reverse_ctx_idx", 0)
    token_prob = attr.get("token_prob", 0.0)
    is_target_logit = bool(attr.get("is_target_logit", False))
    feature_type = str(attr.get("feature_type", ""))
    js_node_id = str(attr.get("jsNodeId") or node_id)
    clerp = str(attr.get("clerp", ""))
    activation_raw = attr.get("activation")
    activation = float(activation_raw) if activation_raw is not None else None

    return Node(
        node_id=node_id,
        feature=int(feature),
        layer=str(layer),
        ctx_idx=int(ctx_idx),
        feature_type=feature_type,
        token_prob=float(token_prob),
        is_target_logit=is_target_logit,
        run_idx=int(run_idx),
        reverse_ctx_idx=int(reverse_ctx_idx),
        jsNodeId=js_node_id,
        clerp=clerp,
        influence=influence,
        activation=activation,
        relevance=relevance,
    )


@dataclass
class Supernode:
    """One grouped supernode: display name, typed members, role, and layer span."""

    name: str
    features: list[Node]
    type: SupernodeType
    layer_min: int
    layer_max: int

    def member_node_ids(self) -> list[str]:
        return [node.node_id for node in self.features]


@dataclass
class SummarizationGraph:
    """
    Supernode-level graph aligned with `sn_adj` / `sn_inf` row order (`nodes`).
    """

    nodes: list[Supernode]
    sn_adj: np.ndarray
    sn_inf: np.ndarray
    F_sn: np.ndarray
    sn_reach: np.ndarray
    sn_act_norm: np.ndarray
    orig_reach_total: float
    surr_reach_total: float
    dominant_paths: list[dict[str, Any]]
    bottleneck_sns: list[dict[str, Any]]

    @property
    def sn_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def to_mapping(self) -> dict[str, list[str]]:
        return {n.name: n.member_node_ids() for n in self.nodes}

    def to_legacy_dict(self) -> dict[str, Any]:
        """Same structure as the historical `build_supernode_graph` return dict."""
        return {
            "sn_names": self.sn_names,
            "sn_adj": self.sn_adj,
            "F_sn": self.F_sn,
            "sn_reach": self.sn_reach,
            "sn_act_norm": self.sn_act_norm,
            "sn_inf": self.sn_inf,
            "orig_reach_total": self.orig_reach_total,
            "surr_reach_total": self.surr_reach_total,
            "dominant_paths": list(self.dominant_paths),
            "bottleneck_sns": list(self.bottleneck_sns),
        }

    def node_by_name(self) -> dict[str, Supernode]:
        return {n.name: n for n in self.nodes}


def cluster_kind_to_supernode_type(kind: Literal["emb", "logit", "middle"]) -> SupernodeType:
    if kind == "middle":
        return "features"
    return kind
