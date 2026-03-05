import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModel
import networkx as nx

@torch.inference_mode()
def build_distance_graph_from_clerp(
    node_ids: List[Any],
    attr: Dict[Any, Dict[str, Any]],
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    batch_size: int = 32,
    use_cls: bool = True,
    normalize: bool = True,
    max_length: int = 64,
    progress: bool = False,
) -> np.ndarray:
    """
    Build a distance matrix between graph nodes using embeddings from a pretrained BERT-style model.

    Each node's text = attr[node_id]['clerp'] (fallback to str(node_id) if missing/empty).
    Node ids may be tuples of one string; we unwrap that.

    Embedding:
      - If use_cls=True: take hidden_state[:,0]
      - Else: mean-pool token embeddings excluding padding (mask)

    Distance:
      - If normalize=True: cosine distance = 1 - cosine_similarity
      - Else: Euclidean distance

    Returns:
      distance_graph: np.ndarray shape (N, N)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts: List[str] = []
    for n in node_ids:
        t = attr.get(n, {}).get("clerp", "")
        if not t or not isinstance(t, str):
            t = str(n)
        t = t.strip()
        if t == "":
            t = "[UNK]"
        texts.append(t)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings: List[torch.Tensor] = []
    rng = range(0, len(texts), batch_size)
    if progress:
        try:
            from tqdm import tqdm
            rng = tqdm(rng, desc="Embedding nodes")
        except ImportError:
            pass

    for start in rng:
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        out = model(**enc)
        hidden = out.last_hidden_state  # (B, T, H)
        if use_cls:
            emb = hidden[:, 0]  # CLS token
        else:
            mask = enc.attention_mask.unsqueeze(-1)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
        embeddings.append(emb.detach().cpu())

    E = torch.cat(embeddings, dim=0)  # (N, H)

    if normalize:
        E = torch.nn.functional.normalize(E, p=2, dim=1)
        # cosine distance = 1 - cosine similarity
        sim = E @ E.T  # (N, N)
        sim = sim.clamp(-1, 1)
        dist = 1.0 - sim
        distance_graph = dist.numpy()
    else:
        # Euclidean distance
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq = (E ** 2).sum(dim=1, keepdim=True)  # (N,1)
        dist2 = sq + sq.T - 2 * (E @ E.T)
        dist2.clamp_(min=0)
        distance_graph = torch.sqrt(dist2).numpy()

    np.fill_diagonal(distance_graph, 0.0)
    return distance_graph

def build_distance_graph_from_decoder_vector(
    node_ids: List[Any],
    attr: Dict[Any, Dict[str, Any]],
    transcoder: Any,  # CrossLayerTranscoder
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a distance graph using decoder vectors from a CrossLayerTranscoder.

    For each feature node, we extract its decoder vector at its output layer.
    Non-feature nodes (embeddings, logits, errors) get a zero vector.
    Then we compute pairwise cosine (or Euclidean) distances.

    Node ID format: "{layer}_{feature_idx}_{ctx_idx}"
      - layer: the encoder (input) layer
      - feature_idx: index into the transcoder's d_transcoder dimension
      - ctx_idx: token position (not used for decoder lookup)

    The decoder vector is selected at the node's own layer, i.e.
      decoder_vecs = transcoder._get_decoder_vectors(enc_layer)[feature_idx]
    which has shape (n_layers - enc_layer, d_model). We pick the slice
    corresponding to the output layer. For CLT features the output layer
    is stored in attr[node]["layer"] (the layer the feature writes to).
    If not available we default to enc_layer (the first decoder slice).

    Args:
        node_ids:    list of node id strings
        attr:        per-node attribute dicts
        transcoder:  a CrossLayerTranscoder instance
        normalize:   if True, cosine distance; else Euclidean

    Returns:
        distance_graph: np.ndarray shape (N, N)
    """
    d_model = transcoder.d_model
    N = len(node_ids)
    vectors = torch.zeros(N, d_model)

    for i, nid in enumerate(node_ids):
        a = attr.get(nid, {})
        ftype = a.get("feature_type", "")

        # Only process feature nodes (skip embedding, logit, error)
        if ftype in ("embedding", "logit", "mlp reconstruction error", ""):
            continue

        # Parse node ID: "{enc_layer}_{feature_idx}_{ctx_idx}"
        parts = str(nid).split("_")
        if len(parts) < 3:
            continue

        try:
            enc_layer = int(parts[0])
            feature_idx = int(parts[1])
        except (ValueError, IndexError):
            continue

        # The output layer this feature writes to
        # attr["layer"] is the output layer for CLT features
        out_layer = a.get("layer", enc_layer)
        if out_layer is None:
            out_layer = enc_layer

        try:
            # (d_transcoder, n_output_layers, d_model) -> (n_output_layers, d_model)
            feat_ids = torch.tensor([feature_idx])
            dec = transcoder._get_decoder_vectors(enc_layer, feat_ids)  # (1, n_out, d_model)
            dec = dec.squeeze(0)  # (n_out, d_model)

            # Index into the correct output layer slice
            out_idx = int(out_layer) - enc_layer
            out_idx = max(0, min(out_idx, dec.shape[0] - 1))
            vectors[i] = dec[out_idx].detach().cpu().float()
        except (IndexError, RuntimeError, KeyError):
            # Feature not found in transcoder; leave as zero
            continue

    # Compute distance matrix
    if normalize:
        # Cosine distance = 1 - cosine_similarity
        norms = vectors.norm(p=2, dim=1, keepdim=True).clamp(min=1e-10)
        E = vectors / norms
        sim = E @ E.T
        sim = sim.clamp(-1, 1)
        dist = 1.0 - sim
        distance_graph = dist.numpy()
    else:
        # Euclidean distance
        sq = (vectors ** 2).sum(dim=1, keepdim=True)
        dist2 = sq + sq.T - 2 * (vectors @ vectors.T)
        dist2.clamp_(min=0)
        distance_graph = torch.sqrt(dist2).numpy()

    np.fill_diagonal(distance_graph, 0.0)
    return distance_graph

def build_graph_from_distance(distance_graph: np.ndarray, threshold: float = 0.0) -> nx.Graph:
    """
    Build a graph from a distance matrix, applying a threshold to create edges.

    Args:
        distance_graph: np.ndarray shape (N, N), the distance matrix
        threshold:      float, if distance < threshold, create an edge

    Returns:
        G: nx.Graph, the resulting graph
    """
    G = nx.Graph()
    N = distance_graph.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            dist = distance_graph[i, j]
            if dist < threshold:
                G.add_edge(i, j, weight=dist)
    return G


if __name__ == "__main__":
    graph_path = "demos/graph_files/factthelargestco-1755767633671_2025-09-26T13-34-02-113Z.json"
    G, attr = trim_graph(graph_path, crit="edge_weight", edge_weight_threshold=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=False))[:5])
    distance_graph = build_distance_graph_from_clerp(G, attr, progress=True, normalize=True)
    print(distance_graph)