"""Canonical attribution graph for pruning and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from summarization.supernode_graph import Node
from summarization.utils import get_data_from_json


@dataclass
class AttrGraph:
    """
    Node-level graph aligned with frontend JSON / PruneGraph conventions.

    Adjacency layout: ``adj[row_i, col_j]`` is edge weight from source node ``j``
    to target node ``i`` (same as ``get_data_from_json`` and ``Graph.adjacency_matrix``).
    """

    nodes: list[Node]
    adj: torch.Tensor
    metadata: dict[str, Any]

    @classmethod
    def from_graph_file(cls, path: str) -> AttrGraph:
        adj, nodes, metadata = get_data_from_json(path)
        return cls(nodes=nodes, adj=adj, metadata=metadata)

    @classmethod
    def from_graph(cls, graph: Any) -> AttrGraph:
        """Convert a ``circuit_tracer.graph.Graph`` into attribute-oriented nodes and adjacency."""
        from circuit_tracer.graph import Graph as GraphCls

        if not isinstance(graph, GraphCls):
            raise TypeError(f"expected Graph, got {type(graph)!r}")

        n_feat = len(graph.selected_features)
        n_pos = graph.n_pos
        n_layers = int(graph.cfg.n_layers)
        n_err = n_pos * n_layers
        n_tok = len(graph.input_tokens)
        n_log = len(graph.logit_tokens)
        device = graph.adjacency_matrix.device
        dtype = graph.adjacency_matrix.dtype

        nodes: list[Node] = []

        tokenizer = None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
        except Exception:
            pass

        prompt_tokens: list[str] = []
        if tokenizer is not None:
            for t in graph.input_tokens:
                tid = int(t.item()) if hasattr(t, "item") else int(t)
                prompt_tokens.append(tokenizer.decode([tid]))
        else:
            prompt_tokens = [str(int(t)) for t in graph.input_tokens]

        def cantor_pairing(x: int, y: int) -> int:
            return (x + y) * (x + y + 1) // 2 + y

        # --- Feature nodes (same ids as frontend Node.feature_node) ---
        sel = graph.selected_features.long()
        for j in range(n_feat):
            af_row = int(sel[j].item())
            layer, pos, feat_idx = graph.active_features[af_row].tolist()
            layer_i, pos_i, feat_i = int(layer), int(pos), int(feat_idx)

            reverse_ctx_idx = 0
            nid = f"{layer_i}_{feat_i}_{pos_i}"
            nodes.append(
                Node(
                    node_id=nid,
                    node_idx=len(nodes),
                    feature=cantor_pairing(layer_i, feat_i),
                    layer=str(layer_i),
                    ctx_idx=pos_i,
                    feature_type="cross layer transcoder",
                    token_prob=0.0,
                    is_target_logit=False,
                    run_idx=0,
                    reverse_ctx_idx=reverse_ctx_idx,
                    jsNodeId=f"{layer_i}_{feat_i}-{reverse_ctx_idx}",
                    clerp="",
                    influence=None,
                    activation=float(graph.activation_values[af_row].item()),
                    relevance=None,
                )
            )

        # --- Error nodes ---
        for node_idx in range(n_feat, n_feat + n_err):
            layer, pos = divmod(node_idx - n_feat, n_pos)
            reverse_ctx_idx = 0
            nid = f"0_{layer}_{pos}"
            nodes.append(
                Node(
                    node_id=nid,
                    node_idx=len(nodes),
                    feature=-1,
                    layer=str(int(layer)),
                    ctx_idx=int(pos),
                    feature_type="mlp reconstruction error",
                    token_prob=0.0,
                    is_target_logit=False,
                    run_idx=0,
                    reverse_ctx_idx=reverse_ctx_idx,
                    jsNodeId=f"{layer}_{pos}-{reverse_ctx_idx}",
                    clerp="",
                    influence=None,
                    activation=None,
                    relevance=None,
                )
            )

        # --- Embedding / token nodes ---
        error_end_idx = n_feat + n_err
        for pos in range(n_tok):
            vocab_idx = graph.input_tokens[pos]
            vid = int(vocab_idx.item()) if hasattr(vocab_idx, "item") else int(vocab_idx)
            nid = f"E_{vid}_{pos}"
            nodes.append(
                Node(
                    node_id=nid,
                    node_idx=len(nodes),
                    feature=pos,
                    layer="E",
                    ctx_idx=pos,
                    feature_type="embedding",
                    token_prob=0.0,
                    is_target_logit=False,
                    run_idx=0,
                    reverse_ctx_idx=0,
                    jsNodeId=f"E_{vid}-{pos}",
                    clerp="",
                    influence=None,
                    activation=None,
                    relevance=None,
                )
            )

        # --- Logit nodes ---
        num_layers_cfg = n_layers
        for pos in range(n_log):
            vocab_idx = graph.logit_tokens[pos]
            vid = int(vocab_idx.item()) if hasattr(vocab_idx, "item") else int(vocab_idx)
            tok_str = tokenizer.decode([vid]) if tokenizer is not None else str(vid)
            layer_logit = str(num_layers_cfg + 1)
            token_prob = float(graph.logit_probabilities[pos].item())
            target = pos == 0
            nid = f"{layer_logit}_{vid}_{pos}"
            nodes.append(
                Node(
                    node_id=nid,
                    node_idx=len(nodes),
                    feature=vid,
                    layer=layer_logit,
                    ctx_idx=pos,
                    feature_type="logit",
                    token_prob=token_prob,
                    is_target_logit=target,
                    run_idx=0,
                    reverse_ctx_idx=0,
                    jsNodeId=f"L_{vid}-{pos}",
                    clerp=f'Output "{tok_str}" (p={token_prob:.3f})',
                    influence=None,
                    activation=None,
                    relevance=None,
                )
            )

        expected_n = n_feat + n_err + n_tok + n_log
        if graph.adjacency_matrix.shape[0] != expected_n:
            raise ValueError(
                f"Graph size mismatch: adjacency has {graph.adjacency_matrix.shape[0]} nodes, "
                f"expected {expected_n} from feature/error/token/logit counts."
            )

        adj = graph.adjacency_matrix.detach().to(dtype=dtype, device=device)
        scan_val = graph.scan
        if isinstance(scan_val, list):
            scan_meta = "-".join(scan_val)
        else:
            scan_meta = scan_val

        metadata: dict[str, Any] = {
            "scan": scan_meta,
            "prompt": graph.input_string,
            "prompt_tokens": prompt_tokens,
            "info": {},
        }

        return cls(nodes=nodes, adj=adj, metadata=metadata)
