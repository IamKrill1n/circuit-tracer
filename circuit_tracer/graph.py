from typing import NamedTuple

import torch
from circuit_tracer.utils.tl_nnsight_mapping import (
    convert_nnsight_config_to_transformerlens,
    UnifiedConfig,
)
from circuit_tracer.utils import get_default_device


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    active_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    selected_features: torch.Tensor
    activation_values: torch.Tensor
    logit_probabilities: torch.Tensor
    cfg: UnifiedConfig
    scan: str | list[str] | None

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        active_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg,
        logit_tokens: torch.Tensor,
        logit_probabilities: torch.Tensor,
        selected_features: torch.Tensor,
        activation_values: torch.Tensor,
        scan: str | list[str] | None = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features, transcoder errors,
        tokens, or logits. They are stored in the order [active_features[0], ...,
        active_features[n-1], error[layer0][position0], error[layer0][position1], ...,
        error[layer l - 1][position t-1], tokens[0], ..., tokens[t-1], logits[top-1 logit],
        ..., logits[top-k logit]].

        Args:
            input_string (str): The input string attributed.
            input_tokens (List[str]): The input tokens attributed.
            active_features (torch.Tensor): A tensor of shape (n_active_features, 3)
                containing the indices (layer, pos, feature_idx) of the non-zero features
                of the model on the given input string.
            adjacency_matrix (torch.Tensor): The adjacency matrix. Organized as
                [active_features, error_nodes, embed_nodes, logit_nodes], where there are
                model.cfg.n_layers * len(input_tokens) error nodes, len(input_tokens) embed
                nodes, len(logit_tokens) logit nodes. The rows represent target nodes, while
                columns represent source nodes.
            cfg (HookedTransformerConfig): The cfg of the model.
            logit_tokens (List[str]): The logit tokens attributed from.
            logit_probabilities (torch.Tensor): The probabilities of each logit token, given
                the input string.
            scan (Union[str,List[str]] | None, optional): The identifier of the
                transcoders used in the graph. Without a scan, the graph cannot be uploaded
                (since we won't know what transcoders were used). Defaults to None
        """
        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = convert_nnsight_config_to_transformerlens(cfg)
        self.n_pos = len(input_tokens)
        self.active_features = active_features
        self.logit_tokens = logit_tokens
        self.logit_probabilities = logit_probabilities
        self.input_tokens = input_tokens
        if scan is None:
            print("Graph loaded without scan to identify it. Uploading will not be possible.")
        self.scan = scan
        self.selected_features = selected_features
        self.activation_values = activation_values

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)

        Args:
            device (_type_): device to send tensors
        """
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.active_features = self.active_features.to(device)
        self.logit_tokens = self.logit_tokens.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)

    def to_pt(self, path: str):
        """Saves the graph at the given path

        Args:
            path (str): The path where the graph will be saved. Should end in .pt
        """
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "active_features": self.active_features,
            "logit_tokens": self.logit_tokens,
            "logit_probabilities": self.logit_probabilities,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "activation_values": self.activation_values,
            "scan": self.scan,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """Load a graph (saved using graph.to_pt) from a .pt file at the given path.

        Args:
            path (str): The path of the Graph to load
            map_location (str, optional): the device to load the graph onto.
                Defaults to 'cpu'.

        Returns:
            Graph: the Graph saved at the specified path
        """
        d = torch.load(path, weights_only=False, map_location=map_location)
        return Graph(**d)

def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    # Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
    # and do logit_weights @ B
    # But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
    # as follows:

    # current_influence = logit_weights @ A
    current_influence = logit_weights.clone()
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    # pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores

def compute_relevance(A: torch.Tensor, emb_weights: torch.Tensor, max_iter: int = 1000):
    # current_relevance = emb_weights @ A
    current_relevance = emb_weights.clone()
    relevance = current_relevance
    iterations = 0
    while current_relevance.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Relevance computation failed to converge after {iterations} iterations"
            )
        current_relevance = current_relevance @ A
        relevance += current_relevance
        iterations += 1
    return relevance

def compute_node_relevance(adjacency_matrix: torch.Tensor, emb_weights: torch.Tensor):
    return compute_relevance(normalize_matrix(adjacency_matrix.T), emb_weights)

def compute_edge_relevance(pruned_matrix: torch.Tensor, emb_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix.T) # (n, n)
    pruned_relevance = compute_relevance(normalized_pruned, emb_weights) # (1, n)
    # pruned_relevance += emb_weights # (1, n)
    edge_scores = normalized_pruned * pruned_relevance[:, None] # (n, n) * (n, n) -> (n, n)
    return edge_scores.T # transpose back to original orientation

def find_threshold(scores: torch.Tensor, threshold: float):
    # Find score threshold that keeps the desired fraction of total influence
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index: int = int(torch.searchsorted(cumulative_score, threshold).item())
    # make sure we don't go out of bounds (only really happens at threshold=1.0)
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]

def remove_dangling_nodes(
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    n_features: int,
    n_tokens: int,
    n_logits: int,
):
    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure feature nodes have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # iteratively prune until all nodes missing incoming / outgoing edges are gone
    # (each pruning iteration potentially opens up new candidates for pruning)
    # this should not take more than n_layers + 1 iterations
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        # Ensure feature and error nodes have outgoing edges
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        # Ensure feature nodes have incoming edges
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative influence scores for each node

def normalize_scores_min_max(scores: torch.Tensor, eps: float = 1e-10):
    scores = scores.clone()
    scores = scores - scores.min()
    scores = scores / (scores.max() + eps)
    return scores

def normalize_scores_rank(scores):
    ranks = torch.argsort(torch.argsort(scores))
    return ranks.float() / (len(scores) - 1)

def combine_scores_geometric(
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
):
    # Normalize first (important!)
    if normalization == "min_max":
        I = normalize_scores_min_max(influence, eps)
        R = normalize_scores_min_max(relevance, eps)
    elif normalization == "rank":
        I = normalize_scores_rank(influence)
        R = normalize_scores_rank(relevance)
    else:
        raise ValueError(f"Invalid normalization method: {normalization}")

    # Geometric mean with alpha
    S = (I + eps) ** alpha * (R + eps) ** (1 - alpha)
    return S

def combined_scores_arithmetic(
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
):
    if normalization == "min_max":
        I = normalize_scores_min_max(influence, eps)
        R = normalize_scores_min_max(relevance, eps)
    elif normalization == "rank":
        I = normalize_scores_rank(influence)
        R = normalize_scores_rank(relevance)
    else:
        raise ValueError(f"Invalid normalization method: {normalization}")
    return I * alpha + R * (1 - alpha)

def combined_scores_harmonic(
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
):
    if normalization == "min_max":
        I = normalize_scores_min_max(influence, eps)
        R = normalize_scores_min_max(relevance, eps)
    elif normalization == "rank":
        I = normalize_scores_rank(influence)
        R = normalize_scores_rank(relevance)
    else:
        raise ValueError(f"Invalid normalization method: {normalization}")
    return 1 / ((1 / (I + eps) + alpha) + (1 / (R + eps) + alpha))

def prune_graph(
    graph: Graph,
    token_weights=None,
    logit_weights=None,
    combined_scores_method: str = "geometric",
    normalization: str = "min_max",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    alpha: float = 0.5,
    keep_all_tokens_and_logits: bool = True,
):
    # --- Setup ---
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    num_nodes = graph.adjacency_matrix.shape[0]

    device = graph.adjacency_matrix.device

    # --- Logit weights ---
    if logit_weights is None:
        logit_weights = torch.zeros(num_nodes, device=device)
        logit_weights[-n_logits:] = graph.logit_probabilities

    # --- Token weights ---
    embed_start_idx = num_nodes - n_logits - n_tokens
    embed_end_idx = num_nodes - n_logits

    if token_weights is None:
        emb_weights = torch.zeros(num_nodes, device=device)
        emb_weights[embed_start_idx:embed_end_idx] = 1 / n_tokens
    else:
        emb_weights = torch.zeros(num_nodes, device=device)
        emb_weights[embed_start_idx:embed_end_idx] = torch.tensor(
            token_weights, device=device
        )

    # =========================
    # 1. Compute RAW scores
    # =========================
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_relevance = compute_node_relevance(graph.adjacency_matrix, emb_weights)

    # =========================
    # 2. Combine → S
    # =========================
    node_scores = combine_scores_geometric(
        node_influence, node_relevance, alpha=alpha
    )

    # =========================
    # 3. Node pruning (percentile)
    # =========================
    node_mask = node_scores >= find_threshold(node_scores, node_threshold)

    if keep_all_tokens_and_logits:
        node_mask[-n_logits - n_tokens :] = True

    # =========================
    # 4. Prune matrix
    # =========================
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    # =========================
    # 5. Edge scores (combine again)
    # =========================
    edge_influence = compute_edge_influence(pruned_matrix, logit_weights)
    edge_relevance = compute_edge_relevance(pruned_matrix, emb_weights)

    edge_scores = combine_scores_geometric(
        edge_influence.flatten(),
        edge_relevance.flatten(),
        alpha=alpha,
        normalization=normalization,
    ).reshape_as(edge_influence)

    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    # =========================
    # 6. Cleanup (same as before)
    # =========================
    remove_dangling_nodes(
        node_mask=node_mask,
        edge_mask=edge_mask,
        n_features=n_features,
        n_tokens=n_tokens,
        n_logits=n_logits,
    )

    # =========================
    # 7. Cumulative scores (based on S)
    # =========================
    sorted_scores, sorted_indices = torch.sort(node_scores, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)

    final_scores = torch.zeros_like(node_scores)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)

def compute_graph_scores(graph: Graph) -> tuple[float, float]:
    """Compute metrics for evaluating how well the graph captures the model's computation.
    This function calculates two complementary scores that measure how much of the model's
    computation flows through interpretable feature nodes versus reconstruction error nodes:
    1. Replacement Score: Measures the fraction of end-to-end influence from input tokens
       to output logits that flows through feature nodes rather than error nodes. This is
       a strict metric that rewards complete explanations where tokens influence logits
       entirely through features.
    2. Completeness Score: Measures the fraction of incoming edges to all nodes (weighted
       by each node's influence on the output) that originate from feature or token nodes
       rather than error nodes. This metric gives partial credit for nodes that are mostly
       explained by features, even if some error influence remains.
    Args:
        graph: The computation graph containing nodes for features, errors, tokens, and logits,
               along with their connections and influence weights.
    Returns:
        tuple[float, float]: A tuple containing:
            - replacement_score: Fraction of token-to-logit influence through features (0-1)
            - completeness_score: Weighted fraction of non-error inputs across all nodes (0-1)
    Note:
        Higher scores indicate better model interpretability, with 1.0 representing perfect
        reconstruction where all computation flows through interpretable features. Lower
        scores indicate more reliance on error nodes, suggesting incomplete feature coverage.
    """
    n_logits = len(graph.logit_tokens)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers  # type: ignore
    token_end = error_end + n_tokens

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    non_error_fractions = 1 - normalized_matrix[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence # + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()


def _combined_node_scores(
    node_influence: torch.Tensor,
    node_relevance: torch.Tensor,
    combined_scores_method: str = "geometric",
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
    if combined_scores_method == "geometric":
        return combine_scores_geometric(
            node_influence,
            node_relevance,
            normalization=normalization,
            alpha=alpha,
            eps=eps,
        )
    if combined_scores_method == "arithmetic":
        return combined_scores_arithmetic(
            node_influence,
            node_relevance,
            normalization=normalization,
            alpha=alpha,
            eps=eps,
        )
    if combined_scores_method == "harmonic":
        return combined_scores_harmonic(
            node_influence,
            node_relevance,
            normalization=normalization,
            alpha=alpha,
            eps=eps,
        )
    raise ValueError(
        "combined_scores_method must be one of "
        "'geometric', 'arithmetic', or 'harmonic'"
    )


def compute_combined_prune_scores(
    graph: Graph,
    node_mask: torch.Tensor,
    combined_scores_method: str = "geometric",
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
) -> tuple[float, float]:
    """Evaluate a pruned graph using influence+relevance combined metrics.

    Args:
        graph: Full graph before pruning.
        node_mask: Boolean tensor over all graph nodes. True indicates a kept node.
        combined_scores_method: How to combine influence/relevance.
        normalization: Score normalization mode passed to combine functions.
        alpha: Influence/relevance tradeoff weight for combined scores.
        eps: Numerical stability constant.

    Returns:
        tuple[float, float]:
            - combined_retention: Fraction of total combined node score retained by node_mask.
            - combined_completeness_score: Within retained nodes, weighted average of the
              non-error incoming-edge fraction, weighted by combined node score.
    """
    num_nodes = graph.adjacency_matrix.shape[0]
    if node_mask.ndim != 1 or node_mask.shape[0] != num_nodes:
        raise ValueError(
            f"node_mask must have shape ({num_nodes},), got {tuple(node_mask.shape)}"
        )
    node_mask = node_mask.to(device=graph.adjacency_matrix.device, dtype=torch.bool)

    n_logits = len(graph.logit_tokens)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers  # type: ignore
    embed_start_idx = num_nodes - n_logits - n_tokens
    embed_end_idx = num_nodes - n_logits

    logit_weights = torch.zeros(num_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    emb_weights = torch.zeros(num_nodes, device=graph.adjacency_matrix.device)
    emb_weights[embed_start_idx:embed_end_idx] = 1 / max(n_tokens, 1)

    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_relevance = compute_node_relevance(graph.adjacency_matrix, emb_weights)
    combined_scores = _combined_node_scores(
        node_influence=node_influence,
        node_relevance=node_relevance,
        combined_scores_method=combined_scores_method,
        normalization=normalization,
        alpha=alpha,
        eps=eps,
    )

    total_combined = combined_scores.sum().clamp(min=eps)
    kept_combined = combined_scores[node_mask].sum()
    combined_retention = kept_combined / total_combined

    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0
    normalized_pruned = normalize_matrix(pruned_matrix)
    non_error_fractions = 1 - normalized_pruned[:, error_start:error_end].sum(dim=-1)

    kept_non_error = non_error_fractions[node_mask]
    combined_weights = combined_scores[node_mask]
    combined_completeness_score = (
        (kept_non_error * combined_weights).sum()
        / combined_weights.sum().clamp(min=eps)
    )

    return combined_retention.item(), combined_completeness_score.item()


def compute_partial_influences(
    edge_matrix: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    max_iter: int = 128,
    device=None,
):
    """Compute partial influences using power iteration method."""
    device = device or get_default_device()

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences
