"""First-class supernode and summarization (cluster) graph types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from summarization.prune import PruneGraph

SupernodeType = Literal["emb", "feature", "features", "logit"]


def cluster_kind_to_supernode_type(kind: str) -> SupernodeType:
    """Map ``cluster_graph`` role strings to ``Supernode.type``."""
    if kind == "emb":
        return "emb"
    if kind == "logit":
        return "logit"
    return "features"


@dataclass
class Node:
    """Summarization-node view aligned with frontend node fields + relevance."""

    node_id: str
    node_idx: int
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

    def set_clerp(self, clerp: str) -> None:
        self.clerp = clerp

    def set_node_idx(self, node_idx: int) -> None:
        self.node_idx = node_idx

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
    prune_graph: "PruneGraph",
    node_id: str,
    id_to_idx: dict[str, int] | None = None,
) -> Node:
    """Build a typed summarization node from ``PruneGraph.nodes`` plus score tensors."""
    nodes: list[Node] = prune_graph.nodes
    lookup = {n.node_id: n for n in nodes}
    base = lookup.get(node_id)
    if base is None:
        raise KeyError(f"unknown node_id for PruneGraph: {node_id!r}")

    if id_to_idx is None:
        id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    idx = id_to_idx[node_id]

    ti = _tensor_value_at(prune_graph.node_influence, idx)
    tr = _tensor_value_at(prune_graph.node_relevance, idx)

    return Node(
        node_id=base.node_id,
        node_idx=idx,
        feature=base.feature,
        layer=str(base.layer),
        ctx_idx=int(base.ctx_idx),
        feature_type=base.feature_type,
        token_prob=float(base.token_prob),
        is_target_logit=base.is_target_logit,
        run_idx=int(base.run_idx),
        reverse_ctx_idx=int(base.reverse_ctx_idx),
        jsNodeId=str(base.jsNodeId or base.node_id),
        clerp=str(base.clerp),
        influence=ti if ti is not None else base.influence,
        activation=base.activation,
        relevance=tr if tr is not None else base.relevance,
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
    Supernode-level graph
    """

    supernodes: list[Supernode]
    pruned_adj: torch.Tensor


    @property
    def sn_names(self) -> list[str]:
        return [n.name for n in self.supernodes]

    def to_mapping(self) -> dict[str, list[str]]:
        return {n.name: n.member_node_ids() for n in self.supernodes}


    @property
    def sn_adj(self) -> np.ndarray:
        # recalculate sn_adj as the sum of the pruned_adj between the supernode features
        sn_adj = np.zeros((len(self.supernodes), len(self.supernodes)), dtype=np.float64)
        for i, u in enumerate(self.supernodes):
            u_idx = [n.node_idx for n in u.features if n.node_idx >= 0]
            if not u_idx:
                continue
            for j, v in enumerate(self.supernodes):
                if i == j:
                    continue
                v_idx = [n.node_idx for n in v.features if n.node_idx >= 0]
                if not v_idx:
                    continue
                block = self.pruned_adj[np.ix_(u_idx, v_idx)].detach().cpu().numpy()
                sn_adj[i, j] = float(block.sum())
        return sn_adj

    def node_by_name(self) -> dict[str, Supernode]:
        return {n.name: n for n in self.supernodes}