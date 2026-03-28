import pytest

from circuit_tracer.subgraph import group


def test_format_features_for_prompt_skips_embedding_and_logit():
    node_ids = ["E_0_0", "L_1", "3_42_0"]
    attr = {
        "E_0_0": {"feature_type": "embedding", "clerp": "emb"},
        "L_1": {"feature_type": "logit", "clerp": "logit"},
        "3_42_0": {
            "feature_type": "cross layer transcoder",
            "layer": 3,
            "ctx_idx": 0,
            "activation": 1.23,
            "clerp": "city concept",
        },
    }

    text = group._format_features_for_prompt(node_ids, attr, prompt="Hello")
    assert "Prompt:" in text
    assert "3_42_0" in text
    assert "city concept" in text
    assert "E_0_0" not in text
    assert "L_1" not in text


def test_parse_response_plain_json():
    s = '[{"label":"Input: city","node_ids":["1","2"]}]'
    out = group._parse_response(s)
    assert isinstance(out, list)
    assert out[0]["label"] == "Input: city"


def test_parse_response_markdown_json_block():
    s = """```json
    [{"label":"Abstract: geography","node_ids":["a","b"]}]
    ```"""
    out = group._parse_response(s)
    assert out[0]["label"].startswith("Abstract:")


def test_parse_response_invalid_raises():
    with pytest.raises(ValueError):
        group._parse_response("not json")


def test_clusters_to_supernodes_filters_singletons():
    clusters = [
        {"label": "Input: cities", "node_ids": ["1", "2"]},
        {"label": "Trash: syntax", "node_ids": ["3"]},
    ]
    out = group.clusters_to_supernodes(clusters)
    assert out == [["Input: cities", "1", "2"]]


def test_group_features_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(group, "GENAI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        group.group_features(["1_2_3"], {"1_2_3": {"feature_type": "cross layer transcoder"}})


def test_group_features_calls_gemini_and_parses(monkeypatch):
    class DummyResponse:
        text = '[{"label":"Input: city","node_ids":["3_42_0","3_99_0"]}]'

    class DummyModels:
        def generate_content(self, model, config, contents):
            assert model == "gemini-2.5-flash"
            assert "Features:" in contents
            return DummyResponse()

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.models = DummyModels()

    monkeypatch.setattr(group, "GENAI_API_KEY", "test-key")
    monkeypatch.setattr(group.genai, "Client", DummyClient)

    node_ids = ["3_42_0", "3_99_0"]
    attr = {
        "3_42_0": {"feature_type": "cross layer transcoder", "clerp": "Dallas"},
        "3_99_0": {"feature_type": "cross layer transcoder", "clerp": "Austin"},
    }

    out = group.group_features(node_ids=node_ids, attr=attr, prompt="capital of Texas")
    assert isinstance(out, list)
    assert out[0]["label"] == "Input: city"
    assert out[0]["node_ids"] == ["3_42_0", "3_99_0"]