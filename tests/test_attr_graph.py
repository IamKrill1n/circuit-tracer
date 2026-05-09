from __future__ import annotations

import json

import torch

from circuit_tracer.graph import Graph
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig
from summarization.attr_graph import AttrGraph
from summarization.prune import prune_attr_graph, prune_masks_from_attr_graph


def _tiny_config() -> UnifiedConfig:
    return UnifiedConfig(
        n_layers=2,
        d_model=8,
        d_head=4,
        n_heads=2,
        d_mlp=16,
        d_vocab=256,
        tokenizer_name="gpt2",
        model_name="stub-model",
        original_architecture="LlamaForCausalLM",
    )


def test_attr_graph_from_graph_matches_expected_order_and_adjacency() -> None:
    cfg = _tiny_config()
    n_pos = 2
    n_feat = 2
    n_err = n_pos * cfg.n_layers
    n_log = 2
    n_total = n_feat + n_err + n_pos + n_log

    active_features = torch.tensor(
        [[0, 0, 10], [1, 1, 11]],
        dtype=torch.long,
    )
    selected_features = torch.tensor([0, 1], dtype=torch.long)
    activation_values = torch.tensor([0.5, 0.25], dtype=torch.float32)
    input_tokens = torch.tensor([101, 102], dtype=torch.long)
    logit_tokens = torch.tensor([201, 202], dtype=torch.long)
    logit_probabilities = torch.tensor([0.7, 0.3], dtype=torch.float32)
    adjacency_matrix = torch.randn(n_total, n_total)

    graph = Graph(
        input_string="ab",
        input_tokens=input_tokens,
        active_features=active_features,
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_tokens=logit_tokens,
        logit_probabilities=logit_probabilities,
        selected_features=selected_features,
        activation_values=activation_values,
        scan="test-scan",
    )

    ag = AttrGraph.from_graph(graph)
    assert len(ag.nodes) == n_total
    assert ag.adj.shape == (n_total, n_total)
    assert torch.allclose(ag.adj, adjacency_matrix)
    assert ag.metadata.get("scan") == "test-scan"
    by_id = {n.node_id: n for n in ag.nodes}
    assert by_id["E_101_0"].feature_type == "embedding"
    layer_log = str(cfg.n_layers + 1)
    assert by_id[f"{layer_log}_201_0"].is_target_logit is True


def test_prune_masks_from_attr_graph_accepts_native_seeds(tmp_path) -> None:
    del tmp_path
    cfg = _tiny_config()
    ag = AttrGraph.from_graph(
        Graph(
            input_string="ab",
            input_tokens=torch.tensor([1, 2], dtype=torch.long),
            active_features=torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
            adjacency_matrix=torch.zeros(10, 10),
            cfg=cfg,
            logit_tokens=torch.tensor([3, 4], dtype=torch.long),
            logit_probabilities=torch.tensor([1.0, 0.0], dtype=torch.float32),
            selected_features=torch.tensor([0, 1], dtype=torch.long),
            activation_values=torch.tensor([1.0, 1.0], dtype=torch.float32),
            scan=None,
        )
    )
    device = ag.adj.device
    num_nodes = ag.adj.shape[0]
    logits_seed = torch.zeros(num_nodes, device=device)
    logits_seed[-2:] = torch.tensor([1.0, 0.0], device=device)
    emb = torch.zeros(num_nodes, device=device)
    emb[-4:-2] = 0.5

    node_mask, edge_mask, *_rest = prune_masks_from_attr_graph(
        ag,
        token_weights=emb,
        logit_weights=logits_seed,
        node_threshold=1.0,
        edge_threshold=1.0,
        keep_all_tokens_and_logits=True,
    )
    assert node_mask.shape == (num_nodes,)
    assert edge_mask.shape == (num_nodes, num_nodes)


def test_attr_graph_from_graph_file_roundtrip(tmp_path) -> None:
    payload = {
        "metadata": {"scan": "m", "prompt_tokens": ["a"], "info": {}},
        "nodes": [
            {
                "node_id": "E_1_0",
                "feature": 0,
                "layer": "E",
                "ctx_idx": 0,
                "feature_type": "embedding",
                "token_prob": 0.0,
                "is_target_logit": False,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": "E_1-0",
                "clerp": "",
            },
            {
                "node_id": "3_9_0",
                "feature": 9,
                "layer": "3",
                "ctx_idx": 0,
                "feature_type": "cross layer transcoder",
                "token_prob": 0.0,
                "is_target_logit": False,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": "x",
                "clerp": "",
            },
            {
                "node_id": "3_99_0",
                "feature": 99,
                "layer": "3",
                "ctx_idx": 1,
                "feature_type": "logit",
                "token_prob": 1.0,
                "is_target_logit": True,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": "y",
                "clerp": "",
            },
        ],
        "links": [{"source": "E_1_0", "target": "3_9_0", "weight": 0.5}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    ag = AttrGraph.from_graph_file(str(path))
    assert [n.node_id for n in ag.nodes] == ["E_1_0", "3_9_0", "3_99_0"]
    assert ag.adj.shape == (3, 3)
    assert abs(float(ag.adj[1, 0].item()) - 0.5) < 1e-6


def test_prune_combined_runs_on_attr_graph_json_fixture(tmp_path) -> None:
    payload = {
        "metadata": {"scan": "m", "prompt_tokens": ["a"], "info": {}},
        "nodes": [
            {
                "node_id": "E_1_0",
                "feature": 0,
                "layer": "E",
                "ctx_idx": 0,
                "feature_type": "embedding",
                "token_prob": 0.0,
                "is_target_logit": False,
                "jsNodeId": "E_1-0",
            },
            {
                "node_id": "3_9_0",
                "feature": 9,
                "layer": "3",
                "ctx_idx": 0,
                "feature_type": "cross layer transcoder",
                "jsNodeId": "x",
            },
            {
                "node_id": "3_99_0",
                "feature": 99,
                "layer": "3",
                "ctx_idx": 1,
                "feature_type": "logit",
                "token_prob": 1.0,
                "is_target_logit": True,
                "jsNodeId": "y",
            },
        ],
        "links": [
            {"source": "E_1_0", "target": "3_9_0", "weight": 1.0},
            {"source": "3_9_0", "target": "3_99_0", "weight": 1.0},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ag = AttrGraph.from_graph_file(str(path))
    pg = prune_attr_graph(ag, logit_weights="target", node_threshold=1.0, edge_threshold=1.0)
    assert len(pg.nodes) >= 1
    assert pg.pruned_adj.ndim == 2
