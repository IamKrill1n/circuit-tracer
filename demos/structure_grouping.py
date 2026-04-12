"""
structure_grouping.py
─────────────────────────────────────────────────────────────────────────────
Groups raw circuit nodes into supernodes by computing pairwise node similarity
from shared in/out neighbors, then clustering.

Supernode graph quantities (grounded definitions):
  SN.act        = max( activation[n] for n in SN )
                  "how strongly does this concept fire on this input"
  SN.inf        = sum( adj[n, logit_idx] for n in SN )
                  "total direct edge weight from this concept to the logit"
                  additive and exact — partitions the original logit-directed flow
  sn_adj[A, B]  = sum( adj[i,j] for i in A, j in B, layer[i] <= layer[j] )
                  "total forward attribution weight flowing from concept A
                   into concept B" — direct sum, no propagation

Validation (exact by construction):
  sum(SN.inf)      == sum(adj[i, logit_idx] for all i)   [partition check]
  sum(sn_adj[A,B]) == sum(adj[i,j] for all cross-SN fwd edges)  [edge conservation]

FIX HISTORY:
  - inf_to_logit now sourced from adj[:, logit_idx] (actual edge weights)
    instead of attr['influence'] (global scalar proxy). This fixes the
    flow residual inflation at late-layer supernodes.
  - max_sn budget fix is in enforce_dag: applies to middle SNs only.
  - Forward mask uses <= (not <) so same-layer cross-SN edges are included.
  - compute_similarity now applies a mediation penalty to S[i,j] when a
    node k exists at an intermediate layer with adj[i,k]>0 and adj[k,j]>0.
    This prevents spectral clustering from grouping non-adjacent nodes like
    A(L1)+C(L3) when B(L2) mediates A→B→C, which would create a cycle in
    the supernode graph (sn_adj[AC,B]>0 AND sn_adj[B,AC]>0).

Usage:
  python structure_grouping.py --file subgraph/austin_plt.pt --target-k 7
  python structure_grouping.py --synthetic --target-k 7
  python structure_grouping.py --file subgraph/austin_plt.pt --threshold 0.45
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


# ─────────────────────────────────────────────────────────────────────────────
# 0.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_layer(nid: str) -> int:
    if nid.startswith('E'):   return 0
    if nid.startswith('27'):  return 27
    return int(nid.split('_')[0])


def parse_activation(attr_node: dict) -> float:
    a = attr_node.get('activation')
    if a is not None:
        return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100


def parse_influence(attr_node: dict) -> float:
    inf = attr_node.get('influence')
    return float(inf) if inf is not None else 0.0


def load_snapshot(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PREPARE GRAPH DATA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_graph_data(raw: dict) -> dict:
    """
    Returns:
        kept_ids     : list[str]
        adj          : torch.Tensor (N,N)  sender-indexed after .T
                       adj[i,j] = "node i sends attribution weight to node j"
        act_values   : dict[str, float]    activation per node
        inf_values   : dict[str, float]    influence score per node (kept for
                       similarity weighting only — NOT used for flow)
        inf_to_logit : dict[str, float]    direct edge weight to logit node
                       SOURCE: adj[i, logit_idx]  (not attr['influence'])
        clerp        : dict[str, str]
        layers       : list[int]
        node_inf     : torch.Tensor (N,)   normalised influence (for similarity)
        logit_idx    : int
    """
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float().T   # receiver→sender flip
    attr      = raw['attr']
    logit_idx = len(kept_ids) - 1

    layers     = [parse_layer(n) for n in kept_ids]
    act_values = {n: parse_activation(attr[n]) for n in kept_ids}
    inf_values = {n: parse_influence(attr[n])  for n in kept_ids}
    clerp      = {n: attr[n].get('clerp', '')  for n in kept_ids}

    # FIX: use actual edge weights to logit, not attr['influence']
    inf_to_logit = {
        nid: float(adj[i, logit_idx])
        for i, nid in enumerate(kept_ids)
    }

    if 'node_inf' in raw:
        node_inf = raw['node_inf'].float()
    else:
        ni = torch.tensor([inf_values[n] for n in kept_ids], dtype=torch.float32)
        node_inf = ni / (ni.max() + 1e-8)

    return dict(
        kept_ids     = kept_ids,
        adj          = adj,
        act_values   = act_values,
        inf_values   = inf_values,
        inf_to_logit = inf_to_logit,
        clerp        = clerp,
        layers       = layers,
        node_inf     = node_inf,
        logit_idx    = logit_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMILARITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_norm(M: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.diag(M).clamp(min=1e-8))
    return M / diag.unsqueeze(1) / diag.unsqueeze(0)


def _compute_mediation_penalty(
    adj: torch.Tensor,
    layers: list,
    mediation_penalty: float = 0.1,
) -> torch.Tensor:
    """
    Compute a penalty matrix P where P[i,j] < 1 if grouping nodes i and j
    would create a cycle in the supernode graph.

    A cycle arises when:
      - layers[i] < layers[k] < layers[j]  (k is strictly between i and j)
      - adj[i,k] > 0  (i sends flow to k)
      - adj[k,j] > 0  (k sends flow to j)

    In that case, grouping i+j into SN_ij produces:
      sn_adj[SN_ij, SN_k] > 0  (via i→k, since layer[i] <= layer[k])
      sn_adj[SN_k, SN_ij] > 0  (via k→j, since layer[k] <= layer[j])
      → CYCLE

    Implementation (vectorized, O(N²) memory, O(N³) ops but via matmul):
      For each pair (i,j), we need: exists k such that
        layer[i] < layer[k] < layer[j] AND adj[i,k]>0 AND adj[k,j]>0

      For a fixed pair (i,j) this is (adj[i,:] > 0) @ (adj[:,j] > 0)
      restricted to k where layer[i] < layer[k] < layer[j].

      We can't vectorize over all three indices simultaneously without
      O(N³) memory, so we loop over unique layer values of k (typically
      ~10-25 distinct layers), which is fast in practice.

    Returns:
        P : torch.Tensor (N, N), values in {mediation_penalty, 1.0}
            P[i,j] = mediation_penalty if a mediating path exists, else 1.0
    """
    N = adj.shape[0]
    layer_t = torch.tensor(layers, dtype=torch.float32)

    # Binarize adjacency (we only care about existence, not weight)
    A = (adj > 0).float()   # (N, N)

    # P starts at 1.0; we'll write mediation_penalty where cycles would form
    P = torch.ones(N, N)

    unique_layers = sorted(set(layers))

    for lk in unique_layers:
        # Mediator mask: nodes at layer lk
        k_mask = (layer_t == lk)          # (N,)
        if not k_mask.any():
            continue

        # A_ik : (N, n_k)  — which sources send to mediators at lk
        A_ik = A[:, k_mask]               # shape (N, n_k)
        # A_kj : (n_k, N)  — which mediators at lk send to targets
        A_kj = A[k_mask, :]               # shape (n_k, N)

        # mediated[i,j] > 0  iff ∃ k at layer lk with adj[i,k]>0 AND adj[k,j]>0
        mediated = (A_ik @ A_kj).clamp(0, 1)   # (N, N), binary

        # Only penalize pairs where lk is STRICTLY between layer[i] and layer[j]
        # i.e. layer[i] < lk < layer[j]  OR  layer[j] < lk < layer[i]
        # (handle both orderings so P is symmetric)
        li = layer_t.unsqueeze(1)   # (N,1)
        lj = layer_t.unsqueeze(0)   # (1,N)

        strictly_between = (
            ((li < lk) & (lj > lk)) |
            ((lj < lk) & (li > lk))
        ).float()   # (N, N)

        # Mark pairs that have at least one mediating path at this layer level
        has_mediator = (mediated * strictly_between) > 0

        P[has_mediator] = mediation_penalty

    # Never penalize self-similarity
    P.fill_diagonal_(1.0)

    return P


def compute_similarity(
    data: dict,
    alpha: float = 0.5,
    beta:  float = 0.5,
    mediation_penalty: float = 0.1,
) -> torch.Tensor:
    """
    S[i,j] = 0.5 * cosine(shared out-neighbors) + 0.5 * cosine(shared in-neighbors)
    Weighted by W = diag(alpha*act_norm + beta*inf_norm).

    Then apply mediation penalty: if a node k exists strictly between layers[i]
    and layers[j] with adj[i,k]>0 and adj[k,j]>0, multiply S[i,j] by
    mediation_penalty (default 0.1). This prevents spectral clustering from
    grouping non-adjacent nodes that have a mediator between them, which would
    otherwise create a cycle A→B→C where A and C are in the same supernode.

    Set mediation_penalty=1.0 to disable (backward compatible).
    """
    kept_ids   = data['kept_ids']
    adj        = data['adj']
    act_values = data['act_values']
    inf_values = data['inf_values']
    layers     = data['layers']

    act_t = torch.tensor([act_values[n] for n in kept_ids], dtype=torch.float32)
    inf_t = torch.tensor([inf_values[n] for n in kept_ids], dtype=torch.float32)

    act_norm = act_t / (act_t.max() + 1e-8)
    inf_norm = inf_t / (inf_t.max() + 1e-8)
    W = torch.diag(alpha * act_norm + beta * inf_norm)

    S_out_cos = _cosine_norm(adj @ W @ adj.T)
    S_in_cos  = _cosine_norm(adj.T @ W @ adj)

    S = (0.5 * S_out_cos + 0.5 * S_in_cos).clamp(0.0, 1.0)

    if mediation_penalty < 1.0:
        P = _compute_mediation_penalty(adj, layers, mediation_penalty)
        S = (S * P).clamp(0.0, 1.0)

        n_penalized = int((P < 1.0).sum().item()) // 2   # symmetric
        print(f'  Mediation penalty applied to {n_penalized} node pairs '
              f'(penalty={mediation_penalty})')

    return S


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def _is_fixed(nid: str) -> bool:
    return nid.startswith('E') or nid.startswith('27')


def cluster_middle_nodes(data: dict,
                         S: torch.Tensor,
                         threshold: float = 0.50,
                         linkage_method: str = 'average') -> dict:
    kept_ids   = data['kept_ids']
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]

    S_mid = S[middle_idx][:, middle_idx].numpy()
    D_mid = (1.0 - S_mid).clip(0.0, 1.0)
    Z     = linkage(squareform(D_mid, checks=False), method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion='distance')

    raw_clusters: dict[int, list[str]] = defaultdict(list)
    for nid, lbl in zip(middle_ids, labels):
        raw_clusters[int(lbl)].append(nid)

    return dict(raw_clusters=dict(raw_clusters), Z=Z, middle_ids=middle_ids, D_mid=D_mid)


def cluster_with_target_k(data: dict,
                          S: torch.Tensor,
                          target_k: int = 7,
                          max_layer_span: int = 4,
                          max_sn: int = None) -> dict:
    from sklearn.cluster import SpectralClustering

    kept_ids   = data['kept_ids']
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    M          = len(middle_ids)

    if target_k >= M:
        raise ValueError(f'target_k={target_k} >= middle nodes={M}')

    S_mid = ((S[middle_idx][:, middle_idx].numpy() +
              S[middle_idx][:, middle_idx].numpy().T) / 2).clip(0, 1)

    upper       = S_mid[np.triu_indices(M, k=1)]
    global_mean = float(upper.mean())
    print(f'  Similarity  mean={global_mean:.4f}  median={np.median(upper):.4f}'
          f'  p75={np.percentile(upper,75):.4f}  max={upper.max():.4f}')

    outlier_mask = S_mid.max(axis=1) < global_mean
    core_local   = [i for i in range(M) if not outlier_mask[i]]
    outlier_ids  = [middle_ids[i] for i in range(M) if outlier_mask[i]]
    core_ids     = [middle_ids[i] for i in core_local]

    if outlier_ids:
        print(f'  Outliers ({len(outlier_ids)}): {outlier_ids}')

    S_core      = S_mid[np.ix_(core_local, core_local)]
    effective_k = min(target_k, len(core_ids) - 1)

    if effective_k < 2:
        raw_clusters = {0: core_ids}
    else:
        labels = SpectralClustering(
            n_clusters=effective_k, affinity='precomputed',
            assign_labels='kmeans', random_state=42, n_init=20,
        ).fit_predict(S_core)
        raw_clusters: dict[int, list[str]] = defaultdict(list)
        for nid, lbl in zip(core_ids, labels):
            raw_clusters[int(lbl)].append(nid)

    layers = {nid: parse_layer(nid) for nid in middle_ids}
    if outlier_ids:
        cluster_mean_layer = {
            lbl: np.mean([layers[n] for n in members])
            for lbl, members in raw_clusters.items()
        }
        for nid in outlier_ids:
            nearest = min(cluster_mean_layer,
                          key=lambda l: abs(cluster_mean_layer[l] - layers[nid]))
            raw_clusters[nearest].append(nid)

    print(f'  Spectral clusters (before DAG enforcement): {len(raw_clusters)}')
    return enforce_dag(dict(raw_clusters), data, max_layer_span, max_sn)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DAG ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def merge_to_budget(final: list, layers: dict, max_sn: int) -> list:
    """
    Greedily merge layer-adjacent cluster pairs until len(final) <= max_sn.
    """
    while len(final) > max_sn:
        best_i, best_j, best_gap = 0, 1, float('inf')
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                ci = np.mean([layers[n] for n in final[i]])
                cj = np.mean([layers[n] for n in final[j]])
                if abs(ci - cj) < best_gap:
                    best_i, best_j, best_gap = i, j, abs(ci - cj)
        final[best_i] = final[best_i] + final[best_j]
        final.pop(best_j)
        final.sort(key=lambda m: min(layers[n] for n in m))
    return final


def enforce_dag(raw_clusters: dict,
                data: dict,
                max_layer_span: int = 4,
                max_sn: int = None) -> dict:
    """
    Enforce DAG structure (no interleaving layer ranges) and apply max_sn budget.

    max_sn applies to MIDDLE supernodes only. Fixed EMB/LOGIT nodes each get
    their own supernode and are counted separately.
    """
    layers = {nid: parse_layer(nid)
              for members in raw_clusters.values() for nid in members}

    queue = list(raw_clusters.values())
    final: list[list[str]] = []

    while queue:
        members = queue.pop()
        lvals   = sorted(set(layers[n] for n in members))
        if lvals[-1] - lvals[0] <= max_layer_span or len(members) == 1:
            final.append(members)
        else:
            mid = (lvals[0] + lvals[-1]) / 2
            lower = [n for n in members if layers[n] <= mid]
            upper = [n for n in members if layers[n] >  mid]
            if not lower or not upper:
                half = len(members) // 2
                lower, upper = members[:half], members[half:]
            queue.extend([lower, upper])

    final.sort(key=lambda m: min(layers[n] for n in m))

    # Resolve interleaving and containment cycles.
    #
    # Interleaving: lo_i < lo_j < hi_i < hi_j  (ranges overlap but neither contains)
    #   → split the wider one at the other's lo boundary.
    #
    # Containment: lo_i < lo_j AND hi_j < hi_i  (SN_i wraps around SN_j)
    #   → guarantees a cycle in sn_adj: members of SN_i below lo_j send flow
    #     into SN_j (layer i_low <= layer j), and SN_j sends flow into members
    #     of SN_i above hi_j (layer j <= layer i_high).
    #   → split SN_i at lo_j so the two parts sit cleanly on either side of SN_j.
    changed = True
    while changed:
        changed = False
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                lo_i = min(layers[n] for n in final[i])
                hi_i = max(layers[n] for n in final[i])
                lo_j = min(layers[n] for n in final[j])
                hi_j = max(layers[n] for n in final[j])

                if (lo_i < lo_j < hi_i) or (lo_j < lo_i < hi_j):
                    # Interleaving: split the wider cluster
                    victim = i if (hi_i - lo_i) >= (hi_j - lo_j) else j
                    other  = j if victim == i else i
                    split  = min(layers[n] for n in final[other])
                    lo_part = [n for n in final[victim] if layers[n] <  split]
                    hi_part = [n for n in final[victim] if layers[n] >= split]
                    if lo_part and hi_part:
                        final[victim] = lo_part
                        final.append(hi_part)
                        final.sort(key=lambda m: min(layers[n] for n in m))
                        changed = True
                        break

                elif (lo_i < lo_j and hi_j < hi_i):
                    # SN_i contains SN_j — split SN_i at lo_j
                    lo_part = [n for n in final[i] if layers[n] <  lo_j]
                    hi_part = [n for n in final[i] if layers[n] >= lo_j]
                    if lo_part and hi_part:
                        final[i] = lo_part
                        final.append(hi_part)
                        final.sort(key=lambda m: min(layers[n] for n in m))
                        changed = True
                        break

                elif (lo_j < lo_i and hi_i < hi_j):
                    # SN_j contains SN_i — split SN_j at lo_i
                    lo_part = [n for n in final[j] if layers[n] <  lo_i]
                    hi_part = [n for n in final[j] if layers[n] >= lo_i]
                    if lo_part and hi_part:
                        final[j] = lo_part
                        final.append(hi_part)
                        final.sort(key=lambda m: min(layers[n] for n in m))
                        changed = True
                        break

            if changed:
                break

    if max_sn is not None:
        n_emb   = len([n for n in data['kept_ids'] if n.startswith('E')])
        n_logit = len([n for n in data['kept_ids'] if n.startswith('27')])
        n_fixed = n_emb + n_logit
        effective_max_middle = max_sn - n_fixed
        if effective_max_middle > 0 and len(final) > effective_max_middle:
            final = merge_to_budget(final, layers, effective_max_middle)
        elif effective_max_middle <= 0:
            if len(final) > 1:
                final = merge_to_budget(final, layers, 1)

    # Name middle supernodes
    final_supernodes = {}
    for idx, members in enumerate(final):
        lo = min(layers[n] for n in members)
        hi = max(layers[n] for n in members)
        name = f'SN_{idx:02d}_L{lo}' if lo == hi else f'SN_{idx:02d}_L{lo}_{hi}'
        final_supernodes[name] = members

    for nid in [n for n in data['kept_ids'] if n.startswith('E')]:
        final_supernodes[f'SN_EMB_{nid}'] = [nid]
    for nid in [n for n in data['kept_ids'] if n.startswith('27')]:
        final_supernodes[f'SN_LOGIT_{nid}'] = [nid]

    return final_supernodes


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DAG CYCLE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_dag_safety(final_supernodes: dict) -> list:
    """
    Check for interleaving layer ranges between supernode pairs.
    Also detects containment cycles: SN_A fully contains SN_B's layer range,
    which would cause both sn_adj[A,B]>0 and sn_adj[B,A]>0 if A has members
    at layers both above and below B.

    Returns list of (sn_a, sn_b) warning pairs.
    """
    def layer_range(members):
        lvals = [parse_layer(n) for n in members]
        return min(lvals), max(lvals)

    ranges   = {sn: layer_range(m) for sn, m in final_supernodes.items()}
    sn_list  = list(ranges.keys())
    warnings = []
    for i, sn_a in enumerate(sn_list):
        for sn_b in sn_list[i+1:]:
            lo_a, hi_a = ranges[sn_a]
            lo_b, hi_b = ranges[sn_b]
            # Interleaving
            if (lo_a < lo_b < hi_a) or (lo_b < lo_a < hi_b):
                warnings.append((sn_a, sn_b))
            # Containment: one SN's range fully contains the other's
            # (lo_a < lo_b AND hi_b < hi_a) means A wraps around B —
            # members of A at layers below lo_b will send to B,
            # and B will send to members of A at layers above hi_b → cycle
            elif (lo_a < lo_b and hi_b < hi_a) or (lo_b < lo_a and hi_a < hi_b):
                warnings.append((sn_a, sn_b))
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_grouping(final_supernodes: dict,
                      data: dict,
                      S: torch.Tensor) -> dict:
    kept_ids   = data['kept_ids']
    act_values = data['act_values']
    clerp      = data['clerp']
    id2idx     = {nid: i for i, nid in enumerate(kept_ids)}

    stats = {}
    for sn, members in final_supernodes.items():
        idx   = [id2idx[n] for n in members if n in id2idx]
        lvals = [parse_layer(n) for n in members]
        pairs = [(i, j) for ii, i in enumerate(idx) for j in idx[ii+1:]]
        if pairs:
            sims = [S[i, j].item() for i, j in pairs]
            intra_mean, intra_min = float(np.mean(sims)), float(np.min(sims))
        else:
            intra_mean = intra_min = 1.0

        stats[sn] = dict(
            n              = len(members),
            layer_lo       = min(lvals),
            layer_hi       = max(lvals),
            layer_span     = max(lvals) - min(lvals),
            act_max        = max(act_values.get(n, 0) for n in members),
            intra_sim_mean = intra_mean,
            intra_sim_min  = intra_min,
            members        = members,
            clerps         = [clerp.get(n, '') for n in members],
        )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SUPERNODE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_supernode_graph(final_supernodes: dict, data: dict) -> dict:
    """
    Construct the supernode graph with three grounded quantities:

    sn_act[SN]   = max( act_values[n] for n in SN )
    sn_inf[SN]   = sum( adj[i, logit_idx] for i in SN )
    sn_adj[A][B] = sum( adj[i,j] for i in A, j in B, layer[i] <= layer[j] )
    """
    kept_ids     = data['kept_ids']
    adj          = data['adj']
    act_values   = data['act_values']
    inf_to_logit = data['inf_to_logit']
    layers       = data['layers']
    logit_idx    = data['logit_idx']

    id2idx   = {nid: i for i, nid in enumerate(kept_ids)}
    sn_names = list(final_supernodes.keys())
    K        = len(sn_names)

    sn_act = {
        sn: max((act_values.get(n, 0.0) for n in members), default=0.0)
        for sn, members in final_supernodes.items()
    }
    act_max_global = max(sn_act.values()) or 1.0
    sn_act_norm = {sn: v / act_max_global for sn, v in sn_act.items()}

    sn_inf = {
        sn: sum(inf_to_logit.get(n, 0.0) for n in members)
        for sn, members in final_supernodes.items()
    }

    sn_adj_mat = np.zeros((K, K), dtype=np.float64)

    for i, sn_a in enumerate(sn_names):
        members_a = final_supernodes[sn_a]
        idx_a = [id2idx[n] for n in members_a if n in id2idx]
        if not idx_a:
            continue
        for j, sn_b in enumerate(sn_names):
            if i == j:
                continue
            members_b = final_supernodes[sn_b]
            idx_b = [id2idx[n] for n in members_b if n in id2idx]
            if not idx_b:
                continue
            src_t = torch.tensor(idx_a, dtype=torch.long)
            tgt_t = torch.tensor(idx_b, dtype=torch.long)
            block = adj[src_t][:, tgt_t]
            src_layers = torch.tensor([layers[s] for s in idx_a])
            tgt_layers = torch.tensor([layers[t] for t in idx_b])
            fwd_mask   = (src_layers.unsqueeze(1) <= tgt_layers.unsqueeze(0)).float()
            sn_adj_mat[i, j] = (block * fwd_mask).sum().item()

    # Validation
    total_inf_orig = float(adj[:, logit_idx].sum())
    total_inf_sn   = sum(sn_inf.values())
    inf_conservation = total_inf_sn / (total_inf_orig + 1e-12)

    node2sn = {nid: sn
               for sn, members in final_supernodes.items()
               for nid in members}
    total_fwd_orig = sum(
        adj[i, j].item()
        for i in range(len(kept_ids))
        for j in range(len(kept_ids))
        if i != j
        and layers[i] <= layers[j]
        and adj[i, j].item() != 0.0
        and node2sn.get(kept_ids[i]) != node2sn.get(kept_ids[j])
    )
    total_fwd_sn      = float(sn_adj_mat.sum())
    edge_conservation = total_fwd_sn / (total_fwd_orig + 1e-12)

    # Cycle check on the resulting sn_adj (belt-and-suspenders)
    cycles_in_sn_adj = [
        (sn_names[i], sn_names[j])
        for i in range(K) for j in range(i+1, K)
        if sn_adj_mat[i, j] > 0 and sn_adj_mat[j, i] > 0
    ]
    if cycles_in_sn_adj:
        print(f'  [WARN] {len(cycles_in_sn_adj)} cycle(s) detected in sn_adj '
              f'despite mediation penalty:')
        for a, b in cycles_in_sn_adj[:5]:
            print(f'    {a} ↔ {b}')

    edges = []
    for i, src in enumerate(sn_names):
        for j, tgt in enumerate(sn_names):
            if i != j and sn_adj_mat[i, j] > 0:
                edges.append(dict(src=src, tgt=tgt, weight=float(sn_adj_mat[i, j])))
    edges.sort(key=lambda e: e['weight'], reverse=True)
    dominant_paths = edges[:5]

    excit_in = np.array([
        sum(v for v in sn_adj_mat[:, j] if v > 0)
        for j in range(K)
    ])
    total_inf_val = sum(sn_inf.values()) or 1.0
    pos_in        = excit_in[excit_in > 0]
    med_excit_in  = float(np.median(pos_in)) if len(pos_in) > 0 else 0.0

    bottleneck_sns = [
        sn_names[i] for i in range(K)
        if (excit_in[i] > med_excit_in
            and sn_inf.get(sn_names[i], 0) / total_inf_val > 0.05
            and 'LOGIT' not in sn_names[i]
            and 'EMB' not in sn_names[i])
    ]

    sn_inf_arr      = np.array([sn_inf[sn]      for sn in sn_names])
    sn_act_arr      = np.array([sn_act[sn]      for sn in sn_names])
    sn_act_norm_arr = np.array([sn_act_norm[sn] for sn in sn_names])

    return dict(
        sn_names          = sn_names,
        sn_act            = sn_act_arr,
        sn_act_norm       = sn_act_norm_arr,
        sn_inf            = sn_inf_arr,
        sn_adj            = sn_adj_mat,
        F_sn              = sn_adj_mat,
        sn_reach          = sn_inf_arr,
        inf_conservation  = inf_conservation,
        edge_conservation = edge_conservation,
        total_inf_orig    = total_inf_orig,
        total_inf_sn      = total_inf_sn,
        total_fwd_orig    = total_fwd_orig,
        total_fwd_sn      = total_fwd_sn,
        dominant_paths    = dominant_paths,
        bottleneck_sns    = bottleneck_sns,
        preservation      = inf_conservation,
        orig_reach_total  = total_inf_orig,
        surr_reach_total  = total_inf_sn,
        cycles_in_sn_adj  = cycles_in_sn_adj,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(final_supernodes: dict,
                 stats: dict,
                 sng: dict,
                 dag_warnings: list) -> None:
    SEP = '─' * 72
    sn_names = sng['sn_names']

    print(f'\n{"═"*72}')
    print('  SUPERNODE GRAPH REPORT')
    print(f'{"═"*72}')

    print(f'\n{SEP}')
    print('  PARTITION VALIDATION  (should both be ~1.000 — exact by construction)')
    print(SEP)
    ic = sng['inf_conservation']
    ec = sng['edge_conservation']
    print(f'  Influence conservation : {ic:.6f}'
          f'  ({sng["total_inf_sn"]:.4f} / {sng["total_inf_orig"]:.4f})'
          f'  {"[PASS]" if abs(ic-1)<0.001 else "[WARN]"}')
    print(f'  Edge conservation      : {ec:.6f}'
          f'  ({sng["total_fwd_sn"]:.4f} / {sng["total_fwd_orig"]:.4f})'
          f'  {"[PASS]" if abs(ec-1)<0.001 else "[WARN]"}')

    cycles = sng.get('cycles_in_sn_adj', [])
    print(f'  SN-adj cycles          : {len(cycles)}'
          f'  {"[PASS]" if not cycles else "[WARN] — see below"}')

    print(f'\n{SEP}')
    print('  SUPERNODE SUMMARY')
    print(SEP)
    print(f"  {'Name':<22} {'n':>3}  {'layers':>10}  {'act_max':>8}  "
          f"{'inf_sum':>9}  {'inf%':>6}  {'intra_sim':>10}")
    print(f'  {"-"*72}')

    total_inf = sng['total_inf_sn'] or 1.0
    for sn, st in stats.items():
        lstr = (f"L{st['layer_lo']}" if st['layer_span'] == 0
                else f"L{st['layer_lo']}–{st['layer_hi']}")
        i   = sn_names.index(sn) if sn in sn_names else -1
        inf = float(sng['sn_inf'][i]) if i >= 0 else 0.0
        pct = inf / total_inf * 100
        print(f"  {sn:<22} {st['n']:>3}  {lstr:>10}  "
              f"{st['act_max']:>8.2f}  {inf:>9.4f}  {pct:>5.1f}%  "
              f"{st['intra_sim_mean']:>10.4f}")

    print(f'\n{SEP}')
    print('  ATTRIBUTION RANKING  (sum adj[i,logit] per supernode)')
    print(SEP)
    for sn in sn_names:
        if 'LOGIT' in sn:
            continue
        i   = sn_names.index(sn)
        inf = float(sng['sn_inf'][i])
        pct = inf / total_inf * 100
        bar = '█' * max(0, int(pct / 2))
        print(f'  {sn:<22}  {inf:>8.4f}  ({pct:>5.1f}%)  {bar}')

    print(f'\n{SEP}')
    print('  DOMINANT SN→SN EDGES  (top 5 by forward attribution weight)')
    print(SEP)
    print(f'  {"Source":<22}  {"Target":<22}  {"sn_adj weight":>14}')
    for e in sng['dominant_paths']:
        print(f'  {e["src"]:<22}  {e["tgt"]:<22}  {e["weight"]:>14.6f}')

    print(f'\n{SEP}')
    print('  SN→SN ADJACENCY  (non-zero forward edges only)')
    print(SEP)
    sn_adj = sng['sn_adj']
    K      = len(sn_names)
    printed = 0
    for i in range(K):
        parts = [f'{sn_names[j]}:{sn_adj[i,j]:.3f}'
                 for j in range(K) if i != j and sn_adj[i, j] > 1e-6]
        if parts:
            print(f'  {sn_names[i]:<22} → {", ".join(parts)}')
            printed += 1
    if printed == 0:
        print('  [INFO] No inter-supernode edges found.')

    print(f'\n{SEP}')
    print('  BOTTLENECK SUPERNODES  (high excitatory in-flow, attribution share > 5%)')
    print(SEP)
    if sng['bottleneck_sns']:
        for sn in sng['bottleneck_sns']:
            i         = sn_names.index(sn)
            outgoing  = sn_adj[i, :]
            incoming  = sn_adj[:, i]
            excit_in  = float(sum(v for v in incoming if v > 0))
            suppr_in  = float(sum(v for v in incoming if v < 0))
            excit_out = float(sum(v for v in outgoing if v > 0))
            suppr_out = float(sum(v for v in outgoing if v < 0))
            inf_sn    = float(sng['sn_inf'][i])
            print(f'  {sn:<22}'
                  f'  in=(+{excit_in:.3f} / {suppr_in:.3f})'
                  f'  out=(+{excit_out:.3f} / {suppr_out:.3f})'
                  f'  inf={inf_sn:.4f}')
    else:
        print('  [PASS] No bottlenecks detected.')

    print(f'\n{SEP}')
    print('  MEMBER DETAILS')
    print(SEP)
    for sn, st in stats.items():
        print(f'\n  ┌── {sn}  '
              f'(n={st["n"]}, layers L{st["layer_lo"]}–{st["layer_hi"]}, '
              f'intra_sim_min={st["intra_sim_min"]:.3f})')
        for nid, c in zip(st['members'], st['clerps']):
            print(f'  │   {nid:<22}  L{parse_layer(nid):<3}  {c}')
        print(f'  └{"─"*60}')

    print(f'\n{SEP}')
    print('  DAG SAFETY CHECK')
    print(SEP)
    if dag_warnings:
        for sn_a, sn_b in dag_warnings:
            print(f'  [WARN] Interleaving/containment layer ranges: {sn_a} ↔ {sn_b}')
    else:
        print('  [PASS] No interleaving or containment layer ranges.')

    if cycles:
        print(f'\n  SN-ADJ CYCLE CHECK')
        for a, b in cycles:
            print(f'  [WARN] Cycle in sn_adj: {a} ↔ {b}')

    print(f'\n{"═"*72}\n')


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DENDROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z, middle_ids: list, threshold: float,
                    out_path: str = 'dendrogram.png') -> None:
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [INFO] matplotlib not available — skipping dendrogram.')
        return
    fig, ax = plt.subplots(figsize=(max(14, len(middle_ids) * 0.22), 5))
    dendrogram(Z, labels=middle_ids, leaf_rotation=90, leaf_font_size=6,
               ax=ax, color_threshold=threshold)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=0.8)
    ax.set_title('Hierarchical Clustering — Middle Nodes', fontsize=11)
    ax.set_ylabel('Distance  (1 − similarity)', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Dendrogram saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SYNTHETIC SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_snapshot() -> dict:
    kept_ids = [
        '16_25_9', '16_12678_9', '16_4298_10', '16_13497_10',
        '17_7178_10',
        '18_1026_10', '18_1437_10', '18_3852_10',
        '18_5495_10', '18_6101_10', '18_8959_10', '18_16041_10',
        '19_7477_9', '19_37_10', '19_1445_10', '19_2439_10',
        '19_2695_10', '19_7477_10',
        '20_15589_9', '20_114_10', '20_5916_10', '20_6026_10',
        '20_7507_10', '20_15276_10', '20_15589_10',
        '21_5943_10', '21_6316_10', '21_6795_10', '21_14975_10',
        '22_31_10', '22_3064_10', '22_3551_10', '22_4999_10', '22_11718_10',
        '23_2288_10', '23_6617_10', '23_11444_10', '23_12237_10',
        '23_12918_10', '23_13193_10', '23_13541_10', '23_13841_10', '23_15366_10',
        '24_709_10', '24_6044_10', '24_6394_10', '24_13277_10',
        '24_15013_10', '24_15627_10', '24_15694_10', '24_16258_10',
        '25_553_10', '25_583_10', '25_762_10', '25_2687_10',
        '25_4259_10', '25_4679_10', '25_4717_10', '25_4886_10',
        '25_13300_10', '25_16302_10',
        'E_6037_4', 'E_2329_7', 'E_26865_9',
        '27_22605_10',
    ]
    act_v = {
        '16_25_9':28.16,'16_12678_9':32.06,'16_4298_10':19.93,'16_13497_10':10.26,
        '17_7178_10':27.50,'18_1026_10':15.99,'18_1437_10':9.26,'18_3852_10':8.45,
        '18_5495_10':12.45,'18_6101_10':20.99,'18_8959_10':39.96,'18_16041_10':9.47,
        '19_7477_9':55.78,'19_37_10':9.70,'19_1445_10':35.45,'19_2439_10':17.40,
        '19_2695_10':17.39,'19_7477_10':26.97,'20_15589_9':45.66,'20_114_10':17.25,
        '20_5916_10':53.43,'20_6026_10':17.02,'20_7507_10':8.21,'20_15276_10':9.98,
        '20_15589_10':49.56,'21_5943_10':56.37,'21_6316_10':10.86,'21_6795_10':16.33,
        '21_14975_10':12.31,'22_31_10':24.70,'22_3064_10':58.50,'22_3551_10':48.36,
        '22_4999_10':37.47,'22_11718_10':34.40,'23_2288_10':14.86,'23_6617_10':9.10,
        '23_11444_10':31.55,'23_12237_10':54.44,'23_12918_10':18.48,'23_13193_10':26.53,
        '23_13541_10':13.38,'23_13841_10':23.95,'23_15366_10':23.05,'24_709_10':10.41,
        '24_6044_10':55.07,'24_6394_10':25.96,'24_13277_10':116.57,'24_15013_10':29.57,
        '24_15627_10':26.43,'24_15694_10':15.49,'24_16258_10':22.16,'25_553_10':51.21,
        '25_583_10':25.03,'25_762_10':11.22,'25_2687_10':10.40,'25_4259_10':29.21,
        '25_4679_10':16.61,'25_4717_10':31.33,'25_4886_10':27.92,'25_13300_10':22.51,
        '25_16302_10':39.76,'E_6037_4':4.15,'E_2329_7':22.77,'E_26865_9':17.14,
        '27_22605_10':45.04,
    }
    inf_v = {
        '16_25_9':0.3085,'16_12678_9':0.4794,'16_4298_10':0.4371,'16_13497_10':0.5397,
        '17_7178_10':0.4327,'18_1026_10':0.4726,'18_1437_10':0.3878,'18_3852_10':0.5543,
        '18_5495_10':0.3778,'18_6101_10':0.3535,'18_8959_10':0.3443,'18_16041_10':0.5711,
        '19_7477_9':0.4462,'19_37_10':0.4213,'19_1445_10':0.3703,'19_2439_10':0.4290,
        '19_2695_10':0.3482,'19_7477_10':0.3802,'20_15589_9':0.3150,'20_114_10':0.5014,
        '20_5916_10':0.4005,'20_6026_10':0.5161,'20_7507_10':0.4665,'20_15276_10':0.4912,
        '20_15589_10':0.2618,'21_5943_10':0.2895,'21_6316_10':0.5390,'21_6795_10':0.3928,
        '21_14975_10':0.4778,'22_31_10':0.4495,'22_3064_10':0.3551,'22_3551_10':0.3996,
        '22_4999_10':0.3403,'22_11718_10':0.3581,'23_2288_10':0.4864,'23_6617_10':0.4076,
        '23_11444_10':0.3977,'23_12237_10':0.2938,'23_12918_10':0.4414,'23_13193_10':0.3919,
        '23_13541_10':0.4482,'23_13841_10':0.4110,'23_15366_10':0.4101,'24_709_10':0.5278,
        '24_6044_10':0.2694,'24_6394_10':0.3181,'24_13277_10':0.3207,'24_15013_10':0.3967,
        '24_15627_10':0.5335,'24_15694_10':0.3857,'24_16258_10':0.4341,'25_553_10':0.3909,
        '25_583_10':0.4400,'25_762_10':0.5573,'25_2687_10':0.5372,'25_4259_10':0.3813,
        '25_4679_10':0.4768,'25_4717_10':0.5019,'25_4886_10':0.4710,'25_13300_10':0.3118,
        '25_16302_10':0.4198,'E_6037_4':0.0415,'E_2329_7':0.2277,'E_26865_9':0.1714,
        '27_22605_10':None,
    }
    clerp = {
        '16_25_9':'Texas legal documents','16_12678_9':'cities',
        '16_4298_10':'capital','16_13497_10':'Numbers and parameters',
        '17_7178_10':'government buildings','18_1026_10':'country names',
        '18_1437_10':'Legal documents from Texas','18_3852_10':'Locations',
        '18_5495_10':'locations','18_6101_10':'capital cities',
        '18_8959_10':'government/state','18_16041_10':'capital',
        '19_7477_9':'Dallas','19_37_10':'Places','19_1445_10':'Downtowns of cities',
        '19_2439_10':'Politics and government','19_2695_10':'cities',
        '19_7477_10':'Dallas','20_15589_9':'Texas','20_114_10':'Oklahoma locations',
        '20_5916_10':'capital','20_6026_10':'political titles',
        '20_7507_10':'countries','20_15276_10':'Dallas sports',
        '20_15589_10':'Texas','21_5943_10':'cities','21_6316_10':'special',
        '21_6795_10':'geographic place names near Texas',
        '21_14975_10':'state/states','22_31_10':'government and policy',
        '22_3064_10':'Texas/Dallas','22_3551_10':'Place names and legal cases',
        '22_4999_10':'Locations','22_11718_10':'Texas locations',
        '23_2288_10':'Texas','23_6617_10':'Locations',
        '23_11444_10':'cities and places','23_12237_10':'Cities and states names',
        '23_12918_10':'Texas','23_13193_10':'Legal and Southern place names',
        '23_13541_10':'News articles','23_13841_10':'towns and cities',
        '23_15366_10':'Code snippets','24_709_10':'patent identifiers',
        '24_6044_10':'in','24_6394_10':'locations',
        '24_13277_10':'Romance languages','24_15013_10':'in',
        '24_15627_10':'Locations','24_15694_10':'US states',
        '24_16258_10':'Detects place names','25_553_10':'general English text',
        '25_583_10':'city names','25_762_10':'international locations',
        '25_2687_10':'locations','25_4259_10':'place names',
        '25_4679_10':'locations','25_4717_10':'unusual/fantastical narratives',
        '25_4886_10':'last names','25_13300_10':'Texas',
        '25_16302_10':'Legal/court cases','E_6037_4':'Emb: " capital"',
        'E_2329_7':'Emb: " state"','E_26865_9':'Emb: " Dallas"',
        '27_22605_10':'Output " Austin" (p=0.450)',
    }
    N         = len(kept_ids)
    adj       = torch.zeros(N, N)
    logit_idx = N - 1
    layers    = [parse_layer(n) for n in kept_ids]
    torch.manual_seed(42)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if j == logit_idx:
                if layers[i] >= 22:
                    w = (act_v[kept_ids[i]] / 300.0) * torch.rand(1).item()
                    if w > 0.03:
                        adj[i, j] = round(w, 4)
            elif layers[i] < layers[j]:
                w = (act_v[kept_ids[i]] / 200.0) * torch.rand(1).item()
                if w > 0.05:
                    adj[i, j] = round(w, 4)
    attr = {}
    for nid in kept_ids:
        is_emb = nid.startswith('E')
        is_log = nid.startswith('27')
        attr[nid] = dict(
            activation      = act_v[nid] if not (is_emb or is_log) else None,
            influence       = inf_v.get(nid),
            clerp           = clerp[nid],
            is_target_logit = is_log,
            token_prob      = 0.4504 if is_log else None,
            ctx_idx         = int(nid.split('_')[2]) if '_' in nid and not is_emb else 0,
        )
    return dict(kept_ids=kept_ids, pruned_adj=adj, attr=attr)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Supernode grouping for circuit graphs')
    parser.add_argument('--file',              type=str,   default='subgraph/austin_plt.pt')
    parser.add_argument('--synthetic',         action='store_true')
    parser.add_argument('--target-k',          type=int,   default=None)
    parser.add_argument('--threshold',         type=float, default=0.50)
    parser.add_argument('--linkage',           type=str,   default='average',
                        choices=['average', 'complete', 'single'])
    parser.add_argument('--max-layer-span',    type=int,   default=4)
    parser.add_argument('--alpha',             type=float, default=0.5)
    parser.add_argument('--beta',              type=float, default=0.5)
    parser.add_argument('--max-sn',            type=int,   default=None)
    parser.add_argument('--mediation-penalty', type=float, default=0.1,
                        help='Similarity penalty for node pairs with a mediating '
                             'path between them (0=full block, 1=disable). Default: 0.1')
    parser.add_argument('--dendrogram',        type=str,   default='dendrogram.png')
    parser.add_argument('--out-json',          type=str,   default='supernode_map.json')
    args = parser.parse_args()

    if args.synthetic:
        print('Using synthetic snapshot...')
        raw = build_synthetic_snapshot()
    else:
        print(f'Loading {args.file}...')
        raw = load_snapshot(args.file)

    data = prepare_graph_data(raw)
    print(f'  Nodes: {len(data["kept_ids"])}  |  adj shape: {data["adj"].shape}')
    print(f'  logit_idx: {data["logit_idx"]}')
    print(f'  total adj[:,logit] = {data["adj"][:, data["logit_idx"]].sum():.4f}')

    print('\nComputing similarity matrix...')
    S = compute_similarity(data, alpha=args.alpha, beta=args.beta,
                           mediation_penalty=args.mediation_penalty)
    print(f'  S range: [{S.min():.4f}, {S.max():.4f}]')

    if args.target_k is not None:
        print(f'\nSpectral clustering (k={args.target_k}, max_span={args.max_layer_span})...')
        final_supernodes = cluster_with_target_k(
            data, S, target_k=args.target_k,
            max_layer_span=args.max_layer_span, max_sn=args.max_sn)
    else:
        print(f'\nHierarchical clustering (threshold={args.threshold})...')
        cluster_result = cluster_middle_nodes(
            data, S, threshold=args.threshold, linkage_method=args.linkage)
        plot_dendrogram(
            cluster_result['Z'], cluster_result['middle_ids'],
            threshold=args.threshold, out_path=args.dendrogram)
        final_supernodes = enforce_dag(
            cluster_result['raw_clusters'], data,
            max_layer_span=args.max_layer_span, max_sn=args.max_sn)

    n_middle = sum(1 for sn in final_supernodes if 'EMB' not in sn and 'LOGIT' not in sn)
    n_total  = len(final_supernodes)
    print(f'  Final supernodes: {n_total} total ({n_middle} middle + {n_total - n_middle} fixed)')

    dag_warnings = check_dag_safety(final_supernodes)
    stats        = evaluate_grouping(final_supernodes, data, S)

    print('\nBuilding supernode graph...')
    sng = build_supernode_graph(final_supernodes, data)
    print(f'  inf_conservation  = {sng["inf_conservation"]:.6f}')
    print(f'  edge_conservation = {sng["edge_conservation"]:.6f}')
    print(f'  sn_adj cycles     = {len(sng.get("cycles_in_sn_adj", []))}')

    print_report(final_supernodes, stats, sng, dag_warnings)

    with open(args.out_json, 'w') as f:
        json.dump(final_supernodes, f, indent=2)
    print(f'Supernode map saved → {args.out_json}')

    sn_flow_out = args.out_json.replace('.json', '_sn_flow.json')
    with open(sn_flow_out, 'w') as f:
        json.dump({
            'sn_names'         : sng['sn_names'],
            'sn_adj'           : sng['sn_adj'].tolist(),
            'F_sn'             : sng['F_sn'].tolist(),
            'sn_reach'         : sng['sn_reach'].tolist(),
            'sn_act_norm'      : sng['sn_act_norm'].tolist(),
            'sn_inf'           : sng['sn_inf'].tolist(),
            'preservation'     : sng['preservation'],
            'orig_reach_total' : sng['orig_reach_total'],
            'surr_reach_total' : sng['surr_reach_total'],
            'inf_conservation' : sng['inf_conservation'],
            'edge_conservation': sng['edge_conservation'],
            'dominant_paths'   : sng['dominant_paths'],
            'bottleneck_sns'   : sng['bottleneck_sns'],
        }, f, indent=2)
    print(f'Supernode flow saved → {sn_flow_out}')


if __name__ == '__main__':
    main()