import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from circuit_tracer.subgraph.pruning import trim_graph

@torch.inference_mode()
def build_distance_graph_from_clerp(
    graph: nx.DiGraph,
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

    def node_key(n):
        return n[0] if isinstance(n, tuple) and len(n) == 1 else n

    nodes: List[Any] = list(graph.nodes())
    if len(nodes) == 0:
        return np.zeros((0, 0), dtype=float)

    texts: List[str] = []
    for n in nodes:
        k = node_key(n)
        t = attr.get(k, {}).get("clerp", "")
        if not t or not isinstance(t, str):
            t = str(k)
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

if __name__ == "__main__":
    graph_path = "demos/graph_files/factthelargestco-1755767633671_2025-09-26T13-34-02-113Z.json"
    G, attr = trim_graph(graph_path, crit="edge_weight", edge_weight_threshold=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=False))[:5])
    distance_graph = build_distance_graph_from_clerp(G, attr, progress=True, normalize=True)
    print(distance_graph)