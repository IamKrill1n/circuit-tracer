import json
import pytest

from circuit_tracer.subgraph.classify import (
    normalize_text,
    heuristic_classify,
    classify_features,
)


def test_normalize_text_basic():
    assert normalize_text("Hello, WORLD!!") == "hello world"
    assert normalize_text("_Dallas-123") == "dallas123"


def test_heuristic_classify_say_prefix_is_output():
    out = heuristic_classify(
        clerp="Say hello",
        top_tokens=["a", "b", "c"],
        top_next_tokens=["x", "y", "z"],
        top_logits=["l1", "l2", "l3"],
    )
    assert out == "output"


def test_heuristic_classify_input_when_clerp_matches_top_tokens():
    out = heuristic_classify(
        clerp="dog",
        top_tokens=["dog"] * 10,
        top_next_tokens=["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"],
        top_logits=["l1", "l2", "l3"],
    )
    assert out == "input"


def test_heuristic_classify_output_when_next_token_dominates():
    out = heuristic_classify(
        clerp="animal concept",
        top_tokens=["dog", "cat", "bird", "fish"],
        top_next_tokens=["the", "the", "the", "the", "the", "the", "x", "y", "z", "w", "v"],
        top_logits=["l1", "l2"],
    )
    assert out == "output"


def test_heuristic_classify_output_when_logit_repeats():
    out = heuristic_classify(
        clerp="animal concept",
        top_tokens=["dog", "cat", "bird", "fish"],
        top_next_tokens=["a", "b", "c", "d", "e", "f"],
        top_logits=["tokA", "tokA", "tokA"],
    )
    assert out == "output"


def test_heuristic_classify_abstract_fallback():
    out = heuristic_classify(
        clerp="topic about travel",
        top_tokens=["airport", "bus", "train", "road"],
        top_next_tokens=["to", "from", "at", "near"],
        top_logits=["alpha", "beta"],
    )
    assert out == "abstract"


def test_classify_features_embedding_and_logit_passthrough():
    node_ids = ["E_0_0", "L_1"]
    attr = {
        "E_0_0": {"feature_type": "embedding"},
        "L_1": {"feature_type": "logit"},
    }
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["E_0_0"] == "embedding"
    assert out["L_1"] == "logit"


def test_classify_features_trash_when_high_act_density(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        data = {
            "explanations": [{"description": "anything"}],
            "frac_nonzero": 0.2,  # > 0.1 => trash
            "activations": [],
            "pos_str": [],
        }
        return 200, json.dumps(data)

    monkeypatch.setattr("circuit_tracer.subgraph.classify.get_feature", fake_get_feature)

    node_ids = ["25_9974_1"]
    attr = {"25_9974_1": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "gemmascope-transcoder-16k"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["25_9974_1"] == "trash"


def test_classify_features_calls_heuristic_path(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        # Build activations so top_tokens all "dog" -> heuristic input
        activations = []
        for _ in range(10):
            activations.append(
                {
                    "tokens": ["x", "dog", "next"],
                    "maxValueTokenIndex": 1,
                }
            )
        data = {
            "explanations": [{"description": "dog"}],
            "frac_nonzero": 0.01,
            "activations": activations,
            "pos_str": ["a", "b", "c"],
        }
        return 200, json.dumps(data)

    monkeypatch.setattr("circuit_tracer.subgraph.classify.get_feature", fake_get_feature)

    node_ids = ["3_42_0"]
    attr = {"3_42_0": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["3_42_0"] == "input"


def test_classify_features_skip_on_non_200(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        return 500, "error"

    monkeypatch.setattr("circuit_tracer.subgraph.classify.get_feature", fake_get_feature)

    node_ids = ["1_2_3"]
    attr = {"1_2_3": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert "1_2_3" not in out
