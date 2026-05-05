"""First-class supernode and summarization (cluster) graph types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SupernodeType = Literal["emb", "features", "logit"]


@dataclass
class Supernode:
    """One grouped supernode: display name, member node ids, and role."""

    name: str
    features: list[str]
    type: SupernodeType


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
        return {n.name: list(n.features) for n in self.nodes}

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
