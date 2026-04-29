from types import SimpleNamespace

import pytest
import torch

from summarization import token_attribution as ta
from summarization.prune import PruneGraph, prune_graph_pipeline


class FakeExplainer:
    def __init__(self, values):
        self.values = values
        self.prompts = None

    def __call__(self, prompts):
        self.prompts = prompts
        return SimpleNamespace(values=self.values)


@pytest.fixture(autouse=True)
def clear_token_attribution_caches():
    ta._cached_prompt_payload_from_graph.cache_clear()
    yield
    ta._cached_prompt_payload_from_graph.cache_clear()


def test_get_token_attribution_returns_normalized_1d_tensor(monkeypatch):
    explainer = FakeExplainer([10.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(ta, "_cached_logit_target_explainer", lambda *_args: explainer)

    weights = ta.get_token_attribution(
        prompt="hello world",
        prompt_tokens=["<bos>", "hello", "world", "!"],
        model_name="fake-model",
        target_token_id=123,
        normalize_method="softmax",
    )

    assert explainer.prompts == ["hello world"]
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32
    assert weights.shape == (4,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[0].item() == 0.0
    assert torch.all(weights[1:] > 0.0)


def test_token_attribution_from_graph_converts_to_prune_pipeline_token_weights(
    monkeypatch,
    tmp_path,
):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        """
{
  "metadata": {
    "prompt": "alpha beta",
    "prompt_tokens": ["alpha", "beta", "<eos>"]
  },
  "nodes": [
    {"node_id": "E_0_0", "feature_type": "embedding", "ctx_idx": 0},
    {"node_id": "E_0_1", "feature_type": "embedding", "ctx_idx": 1},
    {"node_id": "E_0_2", "feature_type": "embedding", "ctx_idx": 2},
    {"node_id": "0_10_0", "feature_type": "cross layer transcoder", "layer": 0, "ctx_idx": 0},
    {"node_id": "L_target", "feature_type": "logit", "is_target_logit": true, "feature": 42}
  ],
  "links": [
    {"source": "E_0_0", "target": "0_10_0", "weight": 0.2},
    {"source": "E_0_1", "target": "0_10_0", "weight": 0.3},
    {"source": "E_0_2", "target": "0_10_0", "weight": 0.4},
    {"source": "0_10_0", "target": "L_target", "weight": 0.8}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ta,
        "_cached_logit_target_explainer",
        lambda *_args: FakeExplainer([1.0, 2.0, 100.0]),
    )

    def fake_graph_pruning_ops():
        def node_scores(adj, _seed):
            return torch.ones(adj.shape[0], dtype=torch.float32, device=adj.device)

        def edge_scores(adj, _seed):
            return (adj != 0).to(torch.float32)

        def find_threshold(values, _threshold):
            return torch.tensor(0.0, dtype=torch.float32, device=values.device)

        return node_scores, edge_scores, node_scores, edge_scores, find_threshold

    monkeypatch.setattr("summarization.prune._graph_pruning_ops", fake_graph_pruning_ops)

    token_weights_tensor = ta.get_token_attribution_from_graph(
        graph_path=graph_path,
        model_name="fake-model",
        normalize_method="softmax",
    ).detach().cpu().to(torch.float32)
    token_weights = token_weights_tensor.tolist()

    assert token_weights == pytest.approx([0.26894143, 0.7310586, 0.0])
    assert all(isinstance(weight, float) for weight in token_weights)

    prune_graph = prune_graph_pipeline(
        json_path=str(graph_path),
        logit_weights="target",
        token_weights=token_weights,
        node_threshold=0.0,
        edge_threshold=0.0,
        keep_all_tokens_and_logits=True,
    )

    assert isinstance(prune_graph, PruneGraph)
    assert prune_graph.metadata["prompt_tokens"] == ["alpha", "beta", "<eos>"]
    assert {"E_0_0", "E_0_1", "E_0_2"}.issubset(prune_graph.kept_ids)
