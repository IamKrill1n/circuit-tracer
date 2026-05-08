"""Pure-math graph scoring utilities — self-contained copy for the summarization package."""

import torch


def normalize_scores_min_max(scores: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    scores = scores.clone()
    scores = scores - scores.min()
    scores = scores / (scores.max() + eps)
    return scores


def normalize_scores_rank(scores: torch.Tensor) -> torch.Tensor:
    ranks = torch.argsort(torch.argsort(scores))
    return ranks.float() / (len(scores) - 1)


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
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


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor) -> torch.Tensor:
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor) -> torch.Tensor:
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def compute_relevance(A: torch.Tensor, emb_weights: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
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


def compute_node_relevance(adjacency_matrix: torch.Tensor, emb_weights: torch.Tensor) -> torch.Tensor:
    return compute_relevance(normalize_matrix(adjacency_matrix.T), emb_weights)


def compute_edge_relevance(pruned_matrix: torch.Tensor, emb_weights: torch.Tensor) -> torch.Tensor:
    normalized_pruned = normalize_matrix(pruned_matrix.T)  # (n, n)
    pruned_relevance = compute_relevance(normalized_pruned, emb_weights)  # (1, n)
    edge_scores = normalized_pruned * pruned_relevance[:, None]  # (n, n)
    return edge_scores.T


def find_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index: int = int(torch.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


def combine_scores_geometric(
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
    if normalization == "min_max":
        I = normalize_scores_min_max(influence, eps)
        R = normalize_scores_min_max(relevance, eps)
    elif normalization == "rank":
        I = normalize_scores_rank(influence)
        R = normalize_scores_rank(relevance)
    else:
        raise ValueError(f"Invalid normalization method: {normalization}")
    return (I + eps) ** alpha * (R + eps) ** (1 - alpha)


def combined_scores_arithmetic(
    influence: torch.Tensor,
    relevance: torch.Tensor,
    normalization: str = "min_max",
    alpha: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
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
) -> torch.Tensor:
    if normalization == "min_max":
        I = normalize_scores_min_max(influence, eps)
        R = normalize_scores_min_max(relevance, eps)
    elif normalization == "rank":
        I = normalize_scores_rank(influence)
        R = normalize_scores_rank(relevance)
    else:
        raise ValueError(f"Invalid normalization method: {normalization}")
    return 1 / ((1 / (I + eps) + alpha) + (1 / (R + eps) + alpha))
