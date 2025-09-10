import torch

from circuit_tracer.graph import Graph, PruneResult, prune_graph, normalize_matrix, compute_influence

def prune_graph_topk(graph, top_k=3):
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + graph.n_pos * layers
    total_nodes = graph.adjacency_matrix.size(0)
    
    highest_logit_node = total_nodes - n_logits


    node_mask, edge_mask, scores = prune_graph(graph, node_threshold = 0.8, edge_threshold = 0.98)
    new_node_mask = torch.zeros(total_nodes, dtype=torch.bool, device=graph.adjacency_matrix.device)
    new_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool, device=graph.adjacency_matrix.device)
    def dfs(node_idx):
        new_node_mask[node_idx] = True
        # Get the row corresponding to incoming effects for this target node.
        row = graph.adjacency_matrix[node_idx]
        filtered_row = row.clone() * edge_mask[node_idx]
        
        if torch.sum(filtered_row.abs()) == 0:
            return
        
        nonzero_idx = (filtered_row > 0).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) == 0:
            return
        
        cur_top_k = min(top_k, len(nonzero_idx))
        # Use absolute values to choose the strongest connections.
        top_vals, top_indices = torch.topk(filtered_row, cur_top_k)

        for src in top_indices.tolist():
            if (src not in range(n_features, error_end_idx)):
                new_edge_mask[node_idx, src] = True
                if not new_node_mask[src]:
                    dfs(src)


    dfs(highest_logit_node)

    node_type = {}

    for nodes in torch.where(new_node_mask)[0].tolist():
        if nodes in range(n_features):
            node_type[nodes] = "Feature"
        elif nodes in range(n_features, error_end_idx):
            node_type[nodes] = "Error"
        elif nodes in range(error_end_idx, error_end_idx + n_tokens):
            node_type[nodes] = "Token"
        else:
            node_type[nodes] = "Logit"

    logit_weights = torch.zeros(total_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities
    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / sorted_scores.sum()
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores
    return (new_node_mask, new_edge_mask, final_scores, node_type)

def prune_graph_edge_weights(graph, edge_weight_threshold = 3):
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + graph.n_pos * layers
    total_nodes = graph.adjacency_matrix.size(0)
    
    highest_logit_node = total_nodes - n_logits

    node_mask, edge_mask, scores = prune_graph(graph, node_threshold = 0.8, edge_threshold = 0.98)
    new_node_mask = torch.zeros(total_nodes, dtype=torch.bool, device=graph.adjacency_matrix.device)
    new_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool, device=graph.adjacency_matrix.device)

    def dfs(node_idx):
        new_node_mask[node_idx] = True
        # Get the row corresponding to incoming effects for this target node.
        row = graph.adjacency_matrix[node_idx]
        filtered_row = row.clone() * edge_mask[node_idx]
        
        if torch.sum(filtered_row) == 0:
            return
        
        nonzero_idx = (filtered_row > 0).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) == 0:
            return
        
        for src in nonzero_idx.tolist():
            if (src not in range(n_features, error_end_idx)):
                if filtered_row[src] >= edge_weight_threshold:
                    new_edge_mask[node_idx, src] = True
                    if not new_node_mask[src]:
                        dfs(src)

    dfs(highest_logit_node)

    node_type = {}

    for nodes in torch.where(new_node_mask)[0].tolist():
        if nodes in range(n_features):
            node_type[nodes] = "Feature"
        elif nodes in range(n_features, error_end_idx):
            node_type[nodes] = "Error"
        elif nodes in range(error_end_idx, error_end_idx + n_tokens):
            node_type[nodes] = "Token"
        else:
            node_type[nodes] = "Logit"

    logit_weights = torch.zeros(total_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities
    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / sorted_scores.sum()
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return (new_node_mask, new_edge_mask, final_scores, node_type)

if __name__ == "__main__":
    from circuit_tracer.graph import Graph
    graph_path = "demos/graphs/example_graph.pt"
    graph = Graph.from_pt(graph_path)
    node_mask, edge_mask, cumulative_scores, node_type = prune_graph_topk(graph, top_k = 3)
    print(edge_mask.sum(), node_mask.sum())
    node_mask, edge_mask, cumulative_scores, node_type = prune_graph_edge_weights(graph, edge_weight_threshold=3)
    print(edge_mask.sum(), node_mask.sum())