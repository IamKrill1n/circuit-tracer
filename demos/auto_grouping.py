"""
auto_grouping.py
─────────────────────────────────────────────────────────────────────────────
Automatic selection of the optimal target-k for structure_grouping.py.

Combines two strategies:
  1. Eigengap heuristic    → narrows the search range
  2. Attribution-aware objective  → picks the best k within that range

Score components (all normalised to [0, 1]):
  intra_sim    : mean intra-cluster cosine similarity (coherence)
  dag_safety   : 1 - fraction of SN pairs with cycle warnings
  attr_balance : entropy of sn_inf distribution / log(n_sn)
  size_score   : proximity of n_supernodes to sqrt(N_middle_nodes)

FIX HISTORY:
  - middle_sn filter uses 'EMB' not in sn and 'LOGIT' not in sn
    (was checking sn not in ('SN_EMB', 'SN_LOGIT') which broke when
    EMB/LOGIT nodes each got per-node names like SN_EMB_E_26865_9).
  - attr_balance uses sn_inf = sum(adj[i, logit_idx]) — actual logit
    edge weights, not attr['influence'] (fixed upstream in
    structure_grouping.prepare_graph_data).

Usage (standalone):
  python auto_grouping.py --file subgraph/austin_plt.pt --max-layer-span 4
  python auto_grouping.py --file subgraph/austin_plt.pt --run-best

Usage (as library):
  from auto_grouping import find_best_k, eigengap_analysis, score_k
"""

import argparse
import json

import numpy as np
import torch
from scipy.linalg import eigvalsh

from structure_grouping import (
    load_snapshot,
    prepare_graph_data,
    compute_similarity,
    cluster_with_target_k,
    check_dag_safety,
    evaluate_grouping,
    build_supernode_graph,
    print_report,
    parse_layer,
    _is_fixed,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EIGENGAP HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

def eigengap_analysis(S: torch.Tensor,
                      kept_ids: list,
                      max_k: int = 20) -> dict:
    """
    Normalised graph Laplacian eigengap on the middle-node similarity submatrix.
    The largest gap between consecutive eigenvalues λ_k, λ_{k+1} suggests
    k natural clusters.

    Returns:
        eigengap_k   : int          suggested k
        eigenvalues  : np.ndarray   first max_k eigenvalues (ascending)
        gaps         : np.ndarray   gaps[i] = λ_{i+1} − λ_i
        search_range : (int, int)   recommended sweep range
    """
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    M = len(middle_idx)

    if M < 3:
        return dict(eigengap_k=2, eigenvalues=np.array([0.0, 1.0]),
                    gaps=np.array([1.0]), search_range=(2, 2))

    S_mid = S[middle_idx][:, middle_idx].numpy()
    S_mid = ((S_mid + S_mid.T) / 2).clip(0.0, 1.0)

    deg        = S_mid.sum(axis=1)
    deg_safe   = np.where(deg > 1e-8, deg, 1e-8)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_norm     = D_inv_sqrt @ (np.diag(deg) - S_mid) @ D_inv_sqrt

    n_eig       = min(max_k + 1, M)
    eigenvalues = np.sort(eigvalsh(L_norm))[:n_eig]
    gaps        = np.diff(eigenvalues)

    search_end = min(len(gaps), max_k)
    if search_end < 2:
        eigengap_k = 2
    else:
        gap_slice  = gaps[1:search_end]
        eigengap_k = int(np.argmax(gap_slice)) + 2

    k_min = max(2, eigengap_k - 2)
    k_max = min(M - 1, eigengap_k + 2)
    if k_max - k_min < 2:
        k_max = min(M - 1, k_min + 4)

    return dict(
        eigengap_k   = eigengap_k,
        eigenvalues  = eigenvalues,
        gaps         = gaps,
        search_range = (k_min, k_max),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ATTRIBUTION-AWARE COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_k(final_supernodes: dict,
            data: dict,
            S: torch.Tensor,
            target_n_middle: int,
            w_intra: float = 0.30,
            w_dag:   float = 0.25,
            w_attr:  float = 0.25,
            w_size:  float = 0.20) -> dict:
    """
    Evaluate a supernode grouping with a composite score.

    Components:
      intra_sim    : weighted mean intra-cluster similarity
      dag_safety   : 1 - (n_cycle_warnings / n_SN_pairs)
      attr_balance : entropy of sn_inf distribution
                     sn_inf = sum(adj[i, logit_idx]) — actual logit flow
      size_score   : 1 - |n_middle_SN - sqrt(N_middle)| / N_middle
    """
    stats        = evaluate_grouping(final_supernodes, data, S)
    dag_warnings = check_dag_safety(final_supernodes)
    sng          = build_supernode_graph(final_supernodes, data)

    # FIX: substring check for per-node EMB/LOGIT names
    # (e.g. SN_EMB_E_26865_9, SN_LOGIT_27_22605_10)
    middle_sn = {sn: st for sn, st in stats.items()
                 if 'EMB' not in sn and 'LOGIT' not in sn}
    n_middle = len(middle_sn)

    if n_middle == 0:
        return dict(total=0.0, intra_sim=0.0, dag_safety=0.0,
                    attr_balance=0.0, size_score=0.0, n_middle=0,
                    n_warnings=0, inf_conservation=0.0,
                    edge_conservation=0.0, details={})

    # ── 1. Intra-cluster similarity ───────────────────────────────────────────
    sizes   = [st['n'] for st in middle_sn.values()]
    total_n = sum(sizes) or 1
    intra_raw = sum(
        st['intra_sim_mean'] * st['n'] / total_n
        for st in middle_sn.values()
    )
    middle_idx  = [i for i, nid in enumerate(data['kept_ids']) if not _is_fixed(nid)]
    S_mid_upper = S[middle_idx][:, middle_idx].numpy()
    upper_vals  = S_mid_upper[np.triu_indices(len(middle_idx), k=1)]
    global_mean = float(upper_vals.mean()) + 1e-8
    intra_sim   = min(1.0, intra_raw / (global_mean * 2))

    # Cohesion penalty: clusters with very low intra_sim_min are bad
    weak      = sum(1 for st in middle_sn.values() if st['intra_sim_min'] < 0.3)
    intra_sim = intra_sim * (1.0 - 0.5 * weak / max(n_middle, 1))

    # ── 2. DAG safety ─────────────────────────────────────────────────────────
    n_pairs    = n_middle * (n_middle - 1) / 2
    dag_safety = 1.0 - len(dag_warnings) / max(n_pairs, 1)

    # ── 3. Attribution balance ────────────────────────────────────────────────
    # sn_inf = sum(adj[i, logit_idx]) — actual direct logit-edge flow per SN.
    # Entropy of this distribution: high entropy = attribution spread evenly.
    sn_names = sng['sn_names']
    sn_inf   = sng['sn_inf']   # np.ndarray aligned with sn_names

    attr_vals = np.array([
        max(float(sn_inf[sn_names.index(sn)]), 0.0)
        for sn in middle_sn
        if sn in sn_names
    ])
    attr_total = attr_vals.sum() + 1e-8
    probs      = attr_vals / attr_total
    probs      = probs[probs > 1e-10]

    if len(probs) > 1:
        entropy      = -np.sum(probs * np.log(probs + 1e-10))
        attr_balance = entropy / np.log(len(probs))
    else:
        attr_balance = 0.0

    # ── 4. Size score ─────────────────────────────────────────────────────────
    ideal_k    = max(2, int(np.sqrt(target_n_middle)))
    deviation  = abs(n_middle - ideal_k) / max(target_n_middle, 1)
    size_score = max(0.0, 1.0 - deviation)

    # ── Composite ─────────────────────────────────────────────────────────────
    total = (w_intra * intra_sim
             + w_dag  * dag_safety
             + w_attr * attr_balance
             + w_size * size_score)

    return dict(
        total             = total,
        intra_sim         = intra_sim,
        dag_safety        = dag_safety,
        flow_balance      = attr_balance,   # backward compat key
        attr_balance      = attr_balance,
        size_score        = size_score,
        n_middle          = n_middle,
        n_warnings        = len(dag_warnings),
        inf_conservation  = float(sng['inf_conservation']),
        edge_conservation = float(sng['edge_conservation']),
        details           = {
            sn: dict(intra_sim=st['intra_sim_mean'],
                     n=st['n'],
                     layers=f"L{st['layer_lo']}-{st['layer_hi']}")
            for sn, st in middle_sn.items()
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_best_k(data: dict,
                S: torch.Tensor,
                max_layer_span: int = 4,
                k_min_override: int = None,
                k_max_override: int = None,
                weights: dict = None,
                max_sn: int = None) -> tuple:
    """
    Full auto-k pipeline:
      1. Eigengap heuristic → search range
      2. Sweep k in range, score each with composite objective
      3. Return best k and all results

    Returns:
        best_k  : int
        results : dict  k → score_dict (includes 'final_supernodes')
    """
    kept_ids   = data['kept_ids']
    middle_ids = [nid for nid in kept_ids if not _is_fixed(nid)]
    N_middle   = len(middle_ids)

    if N_middle < 3:
        print('  Too few middle nodes for auto-k. Defaulting to k=2.')
        return 2, {}

    # ── Step 1: Eigengap ──────────────────────────────────────────────────────
    print('\n── Step 1: Eigengap analysis ──')
    eg = eigengap_analysis(S, kept_ids, max_k=min(20, N_middle - 1))
    print(f'  Eigengap suggests k = {eg["eigengap_k"]}')
    print(f'  Eigenvalue gaps (first 10): '
          f'{", ".join(f"{g:.4f}" for g in eg["gaps"][:10])}')
    print(f'  Initial search range: [{eg["search_range"][0]}, {eg["search_range"][1]}]')

    k_min = k_min_override if k_min_override is not None else eg['search_range'][0]
    k_max = k_max_override if k_max_override is not None else eg['search_range'][1]
    k_min = max(2, k_min)
    k_max = min(N_middle - 1, k_max)
    if k_min > k_max:
        k_min = k_max
    print(f'  Final search range: [{k_min}, {k_max}]')

    w = weights or {}

    # ── Step 2: Sweep ─────────────────────────────────────────────────────────
    print(f'\n── Step 2: Scoring k = {k_min}..{k_max} ──')
    print(f'  {"k":>3}  {"n_mid":>5}  {"intra":>6}  {"dag":>5}  '
          f'{"attr_bal":>8}  {"size":>5}  {"TOTAL":>6}  {"warns":>5}  '
          f'{"inf_con":>7}  {"edg_con":>7}')
    print(f'  {"─"*68}')

    results = {}
    for k in range(k_min, k_max + 1):
        try:
            final_sn = cluster_with_target_k(
                data, S, target_k=k,
                max_layer_span=max_layer_span, max_sn=max_sn)
        except Exception as e:
            print(f'  k={k} failed: {e}')
            continue

        sc = score_k(
            final_sn, data, S,
            target_n_middle = N_middle,
            w_intra = w.get('w_intra', 0.30),
            w_dag   = w.get('w_dag',   0.25),
            w_attr  = w.get('w_flow',  0.25),
            w_size  = w.get('w_size',  0.20),
        )
        sc['final_supernodes'] = final_sn
        results[k] = sc

        n_total  = len(final_sn)
        n_middle = sc['n_middle']
        print(f'  {k:>3}  {n_middle:>2}+{n_total-n_middle:<2}  {sc["intra_sim"]:>6.4f}  '
              f'{sc["dag_safety"]:>5.4f}  {sc["attr_balance"]:>8.4f}  '
              f'{sc["size_score"]:>5.4f}  {sc["total"]:>6.4f}  '
              f'{sc["n_warnings"]:>5}  '
              f'{sc["inf_conservation"]:>7.4f}  {sc["edge_conservation"]:>7.4f}')

    if not results:
        print('  All k values failed. Falling back to eigengap suggestion.')
        return eg['eigengap_k'], {}

    # ── Step 3: Best k ────────────────────────────────────────────────────────
    best_k = max(results, key=lambda k: results[k]['total'])
    best   = results[best_k]

    n_total  = len(best['final_supernodes'])
    n_middle = best['n_middle']

    print(f'\n── Result ──')
    print(f'  Best k = {best_k}  (total score = {best["total"]:.4f})')
    print(f'    Supernodes: {n_middle} middle + {n_total - n_middle} fixed = {n_total} total')
    print(f'    intra_sim    = {best["intra_sim"]:.4f}')
    print(f'    dag_safety   = {best["dag_safety"]:.4f}')
    print(f'    attr_balance = {best["attr_balance"]:.4f}')
    print(f'    size_score   = {best["size_score"]:.4f}')
    print(f'    n_warnings   = {best["n_warnings"]}')
    print(f'    inf_conservation  = {best["inf_conservation"]:.6f}')
    print(f'    edge_conservation = {best["edge_conservation"]:.6f}')

    return best_k, results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, best_k: int,
                 out_path: str = 'auto_k_results.json') -> None:
    out = {'best_k': best_k, 'sweep': {}}
    for k, sc in results.items():
        out['sweep'][str(k)] = {
            key: val for key, val in sc.items()
            if key != 'final_supernodes'
        }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSweep results saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_auto_k(eigengap_result: dict, results: dict, best_k: int,
                out_path: str = 'auto_k_plot.png') -> None:
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print('  [INFO] matplotlib not available — skipping plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: eigenvalue gaps ─────────────────────────────────────────────────
    ax    = axes[0]
    gaps  = eigengap_result['gaps']
    x_gap = list(range(2, 2 + len(gaps)))
    ax.bar(x_gap, gaps, color='#3d8eff', alpha=0.7, edgecolor='#1a2540')
    eg_k = eigengap_result['eigengap_k']
    if eg_k in x_gap:
        ax.bar(eg_k, gaps[x_gap.index(eg_k)], color='#f5a623', alpha=0.9,
               edgecolor='#1a2540')
    ax.axvline(x=eg_k, color='#f5a623', linestyle='--', alpha=0.6,
               label=f'eigengap k={eg_k}')
    ax.set_xlabel('k', fontsize=9)
    ax.set_ylabel('Eigenvalue gap (λ_{k+1} − λ_k)', fontsize=9)
    ax.set_title('Eigengap Heuristic', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # ── Right: composite scores ───────────────────────────────────────────────
    ax     = axes[1]
    ks     = sorted(results.keys())
    colors = ['#f5a623' if k == best_k else '#3d8eff' for k in ks]
    ax.bar(ks, [results[k]['total'] for k in ks],
           color=colors, alpha=0.8, edgecolor='#1a2540')

    components   = ['intra_sim', 'dag_safety', 'attr_balance', 'size_score']
    comp_colors  = ['#10b981', '#8b5cf6', '#f43f5e', '#06b6d4']
    comp_weights = [0.30,       0.25,      0.25,       0.20]
    comp_labels  = ['intra_sim (0.30)', 'dag_safety (0.25)',
                    'attr_balance (0.25)', 'size_score (0.20)']
    bottom = np.zeros(len(ks))
    for comp, cc, w in zip(components, comp_colors, comp_weights):
        vals = [results[k].get(comp, 0.0) * w for k in ks]
        ax.bar(ks, vals, bottom=bottom, color=cc, alpha=0.3, width=0.6)
        bottom += np.array(vals)

    ax.axvline(x=best_k, color='#f5a623', linestyle='--', alpha=0.6,
               label=f'best k={best_k}')
    ax.legend(handles=[
        Patch(facecolor=c, alpha=0.4, label=l)
        for c, l in zip(comp_colors, comp_labels)
    ] + [Patch(facecolor='#f5a623', alpha=0.9, label=f'best k={best_k}')],
              fontsize=7, loc='upper right')
    ax.set_xlabel('k', fontsize=9)
    ax.set_ylabel('Composite score', fontsize=9)
    ax.set_title('Attribution-Aware Objective', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Auto-k plot saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automatically determine optimal target-k for structure_grouping.py')
    parser.add_argument('--file',           type=str, required=True)
    parser.add_argument('--max-layer-span', type=int,   default=4)
    parser.add_argument('--alpha',          type=float, default=0.5)
    parser.add_argument('--beta',           type=float, default=0.5)
    parser.add_argument('--k-min',          type=int,   default=None)
    parser.add_argument('--k-max',          type=int,   default=None)
    parser.add_argument('--w-intra',        type=float, default=0.30)
    parser.add_argument('--w-dag',          type=float, default=0.25)
    parser.add_argument('--w-flow',         type=float, default=0.25)
    parser.add_argument('--w-size',         type=float, default=0.20)
    parser.add_argument('--max-sn',         type=int,   default=None)
    parser.add_argument('--out-json',       type=str,   default='auto_k_results.json')
    parser.add_argument('--out-plot',       type=str,   default='auto_k_plot.png')
    parser.add_argument('--run-best',       action='store_true',
                        help='Run structure_grouping with best k and save all outputs')
    args = parser.parse_args()

    print(f'Loading {args.file}...')
    raw  = load_snapshot(args.file)
    data = prepare_graph_data(raw)
    print(f'  Nodes: {len(data["kept_ids"])}')
    print(f'  logit_idx: {data["logit_idx"]}')
    print(f'  total adj[:,logit] = {data["adj"][:, data["logit_idx"]].sum():.4f}')

    print('Computing similarity matrix...')
    S = compute_similarity(data, alpha=args.alpha, beta=args.beta)
    print(f'  S shape: {S.shape}')

    eg = eigengap_analysis(S, data['kept_ids'])

    weights = dict(w_intra=args.w_intra, w_dag=args.w_dag,
                   w_flow=args.w_flow,   w_size=args.w_size)

    best_k, results = find_best_k(
        data, S,
        max_layer_span = args.max_layer_span,
        k_min_override = args.k_min,
        k_max_override = args.k_max,
        weights        = weights,
        max_sn         = args.max_sn,
    )

    if results:
        save_results(results, best_k, args.out_json)
        plot_auto_k(eg, results, best_k, args.out_plot)

    if args.run_best and best_k in results:
        print(f'\n── Running structure_grouping with k={best_k} ──')
        final_sn = results[best_k]['final_supernodes']

        with open('supernode_map.json', 'w') as f:
            json.dump(final_sn, f, indent=2)
        print('Supernode map saved → supernode_map.json')

        dag_warnings = check_dag_safety(final_sn)
        stats        = evaluate_grouping(final_sn, data, S)
        sng          = build_supernode_graph(final_sn, data)
        print_report(final_sn, stats, sng, dag_warnings)

        sn_flow_out = 'supernode_map_sn_flow.json'
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

    print(f'\n  ➜  Use:  python structure_grouping.py --file {args.file} '
          f'--target-k {best_k} --max-layer-span {args.max_layer_span}')


if __name__ == '__main__':
    main()