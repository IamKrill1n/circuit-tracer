import pytest
import torch

from summarization.prune import (
    PruneGraph,
    _validate_threshold,
    _validate_inputs,
    prune_combined,
    prune_graph_pipeline,
)
from summarization.utils import _build_index_sets, _node_from_json_dict


@pytest.fixture
def tiny_graph():
    # adj[target, source]
    node_ids = ["E_0_0", "0_10_0", "L_1", "L_2"]
    attr = {
        "E_0_0": {"feature_type": "embedding", "ctx_idx": 0},
        "0_10_0": {"feature_type": "cross layer transcoder", "layer": 0, "ctx_idx": 0},
        "L_1": {"feature_type": "logit", "is_target_logit": True, "token_prob": 0.9},
        "L_2": {"feature_type": "logit", "is_target_logit": False, "token_prob": 0.1},
    }
    adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return adj, node_ids, attr


def test_validate_threshold_valid_values():
    _validate_threshold("t", 0.0)
    _validate_threshold("t", 0.5)
    _validate_threshold("t", 1.0)


@pytest.mark.parametrize("bad", [-1e-6, 1.000001])
def test_validate_threshold_invalid_values(bad):
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        _validate_threshold("t", bad)


def test_validate_inputs_square_check(tiny_graph):
    _, node_ids, attr = tiny_graph
    bad_adj = torch.zeros((3, 4))
    with pytest.raises(ValueError, match="square 2D"):
        _validate_inputs(bad_adj, node_ids, attr, "target", None, None, None)


def test_validate_inputs_len_check(tiny_graph):
    adj, node_ids, attr = tiny_graph
    with pytest.raises(ValueError, match="mismatch"):
        _validate_inputs(adj, node_ids[:-1], attr, "target", None, None, None)


def test_validate_inputs_missing_attr_entry(tiny_graph):
    adj, node_ids, attr = tiny_graph
    bad_attr = dict(attr)
    bad_attr.pop("L_2")
    with pytest.raises(ValueError, match="attr missing entries"):
        _validate_inputs(adj, node_ids, bad_attr, "target", None, None, None)


def test_validate_inputs_bad_logit_mode(tiny_graph):
    adj, node_ids, attr = tiny_graph
    with pytest.raises(ValueError, match="logit_weights must be"):
        _validate_inputs(adj, node_ids, attr, "invalid", None, None, None)  # type: ignore[arg-type]


def test_validate_inputs_bad_token_weight_type(tiny_graph):
    adj, node_ids, attr = tiny_graph
    with pytest.raises(ValueError, match="token_weights must be a list of floats"):
        _validate_inputs(adj, node_ids, attr, "target", [1.0, "x"], None, None)  # type: ignore[list-item]


def test_build_index_sets_contents(tiny_graph):
    _, node_ids, attr = tiny_graph
    idx = _build_index_sets(node_ids, attr)
    assert idx["embedding"] == [0]
    assert idx["feature"] == [1]
    assert idx["target_logit"] == [2]
    assert idx["logit"] == [2, 3]
    assert idx["error"] == []


def test_prune_combined_shapes_and_scores_subset(tiny_graph):
    adj, node_ids, attr = tiny_graph
    (
        node_mask,
        edge_mask,
        node_inf,
        node_rel,
        _edge_inf,
        _edge_rel,
        ratio,
    ) = prune_combined(
        adj=adj,
        node_ids=node_ids,
        attr=attr,
        logit_weights="target",
        token_weights=[1.0],
        node_influence_threshold=0.9,
        node_relevance_threshold=0.9,
        edge_influence_threshold=0.9,
        edge_relevance_threshold=0.9,
        keep_all_tokens_and_logits=False,
    )
    assert node_mask.dtype == torch.bool
    assert edge_mask.dtype == torch.bool
    assert node_mask.shape[0] == len(node_ids)
    assert edge_mask.shape == adj.shape
    assert node_inf.ndim == 1
    assert node_rel.ndim == 1
    assert node_inf.shape[0] == len(node_ids)
    assert isinstance(ratio, float)


def test_prune_combined_keep_all_logits(tiny_graph):
    adj, node_ids, attr = tiny_graph
    node_mask, *_ = prune_combined(
        adj=adj,
        node_ids=node_ids,
        attr=attr,
        logit_weights="probs",
        token_weights=[1.0],
        node_influence_threshold=0.99,
        node_relevance_threshold=0.99,
        edge_influence_threshold=0.99,
        edge_relevance_threshold=0.99,
        keep_all_tokens_and_logits=True,
    )
    assert bool(node_mask[0].item()) is True
    assert bool(node_mask[2].item()) is True
    assert bool(node_mask[3].item()) is True


def test_prune_combined_target_without_target_raises(tiny_graph):
    adj, node_ids, attr = tiny_graph
    bad_attr = dict(attr)
    bad_attr["L_1"] = dict(attr["L_1"])
    bad_attr["L_1"]["is_target_logit"] = False

    with pytest.raises(ValueError, match="No target logit"):
        prune_combined(
            adj,
            node_ids,
            bad_attr,
            logit_weights="target",
            token_weights=[1.0],
            node_influence_threshold=0.9,
            node_relevance_threshold=0.9,
            edge_influence_threshold=0.9,
            edge_relevance_threshold=0.9,
            keep_all_tokens_and_logits=False,
        )


def test_prune_combined_token_weights_len_mismatch_raises(tiny_graph):
    adj, node_ids, attr = tiny_graph
    with pytest.raises(ValueError, match="token_weights length"):
        prune_combined(
            adj=adj,
            node_ids=node_ids,
            attr=attr,
            logit_weights="target",
            token_weights=[0.5, 0.5],
            node_influence_threshold=0.9,
            node_relevance_threshold=0.9,
            edge_influence_threshold=0.9,
            edge_relevance_threshold=0.9,
            keep_all_tokens_and_logits=False,
        )


def test_prune_graph_pipeline_returns_prunegraph(monkeypatch, tiny_graph):
    adj, node_ids, attr = tiny_graph

    def fake_loader(_json_path):
        nodes = [_node_from_json_dict({"node_id": nid, **attr[nid]}) for nid in node_ids]
        return adj, nodes, {"prompt": "hello"}

    monkeypatch.setattr("summarization.attr_graph.get_data_from_json", fake_loader)

    out = prune_graph_pipeline(
        json_path="dummy.json",
        logit_weights="target",
        token_weights=[1.0],
        node_threshold=0.9,
        edge_threshold=0.9,
        keep_all_tokens_and_logits=False,
    )

    assert isinstance(out, PruneGraph)
    assert isinstance(out.kept_ids, list)
    assert isinstance(out.pruned_adj, torch.Tensor)
    assert out.pruned_adj.ndim == 2
    assert set(out.kept_ids) == set(out.attr.keys())
    assert out.metadata["prompt"] == "hello"
    assert out.pruned_adj.shape[0] == len(out.kept_ids)
    assert out.pruned_adj.shape[1] == len(out.kept_ids)
    assert out.num_nodes == len(out.kept_ids)
    assert out.num_edges == int((out.pruned_adj != 0).sum().item())


def test_prune_graph_pipeline_threshold_validation(monkeypatch, tiny_graph):
    adj, node_ids, attr = tiny_graph

    def fake_loader(_json_path):
        nodes = [_node_from_json_dict({"node_id": nid, **attr[nid]}) for nid in node_ids]
        return adj, nodes, {}

    monkeypatch.setattr("summarization.attr_graph.get_data_from_json", fake_loader)

    with pytest.raises(ValueError, match="node_influence_threshold"):
        prune_graph_pipeline(
            json_path="dummy.json",
            logit_weights="target",
            node_threshold=1.5,
        )
