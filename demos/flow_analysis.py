"""
flow_analysis.py
─────────────────────────────────────────────────────────────────────────────
Flow-theoretic additions to the supernode abstraction pipeline.

Three analyses that upgrade conservation checks (necessary) into
flow faithfulness checks (closer to sufficient):

  1. Path Attribution Decomposition
  2. Local Flow Residual
  3. Shortcut Ratio

FIX HISTORY:
  - sn_inf is now sourced from adj[:, logit_idx] in structure_grouping.py.
  - max_sn budget fix is in structure_grouping.enforce_dag.
  - R_phi is now split into two independent metrics:
      R_phi_balance    : mean residual over non-suppressive middle SNs only.
                         Reflects genuine grouping quality (flow imbalance).
      R_phi_suppressive: mean |inf_exit| / (|in_flow| + ε) over suppressive
                         SNs only. Reflects suppression strength, reported
                         separately and does NOT penalize residual_score.
    This prevents suppressive nodes (inf_exit < 0) from inflating R_phi
    beyond 1.0, which masked real imbalance issues and distorted F(phi).
"""

import argparse
import json
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from structure_grouping import (
    load_snapshot,
    prepare_graph_data,
    compute_similarity,
    cluster_with_target_k,
    check_dag_safety,
    evaluate_grouping,
    build_supernode_graph,
    build_synthetic_snapshot,
    parse_layer,
    _is_fixed,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _classify_sn(sn_name: str) -> str:
    if 'EMB' in sn_name:
        return 'emb'
    elif 'LOGIT' in sn_name:
        return 'logit'
    else:
        return 'middle'


def _build_sn_dag_order(sn_names: list, final_supernodes: dict) -> list:
    def min_layer(sn):
        members = final_supernodes[sn]
        return min(parse_layer(n) for n in members)
    return sorted(sn_names, key=min_layer)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATH ATTRIBUTION DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def path_attribution_decomposition(
    sng: dict,
    final_supernodes: dict,
    top_k: int = 10,
    min_flow_frac: float = 1e-4,
) -> dict:
    sn_names = sng['sn_names']
    sn_adj   = sng['sn_adj']
    sn_inf   = sng['sn_inf']
    K        = len(sn_names)

    name2idx = {sn: i for i, sn in enumerate(sn_names)}

    emb_sns   = [sn for sn in sn_names if _classify_sn(sn) == 'emb']
    logit_sns = [sn for sn in sn_names if _classify_sn(sn) == 'logit']

    topo_order = _build_sn_dag_order(sn_names, final_supernodes)

    path_flows: dict[tuple, float] = {}

    for sn_e in emb_sns:
        i = name2idx[sn_e]
        out_total = sum(max(0.0, sn_adj[i, j]) for j in range(K) if j != i)
        if out_total > 0:
            path_flows[(sn_e,)] = out_total

    for sn_e in emb_sns:
        i = name2idx[sn_e]
        direct_inf = float(sn_inf[i])
        if direct_inf > 0:
            for sn_l in logit_sns:
                path_key = (sn_e, sn_l)
                path_flows[path_key] = path_flows.get(path_key, 0.0) + direct_inf

    total_emb_out = sum(path_flows.get((sn_e,), 0.0) for sn_e in emb_sns)
    prune_threshold = total_emb_out * min_flow_frac if total_emb_out > 0 else 0.0

    completed_paths: dict[tuple, float] = {}

    for sn in topo_order:
        i    = name2idx[sn]
        kind = _classify_sn(sn)

        if kind == 'logit':
            for path, flow in list(path_flows.items()):
                if path[-1] == sn:
                    completed_paths[path] = completed_paths.get(path, 0.0) + flow
            continue

        if kind == 'emb':
            prefixes_here = [(p, f) for p, f in path_flows.items()
                             if p == (sn,)]
        else:
            prefixes_here = [(p, f) for p, f in path_flows.items()
                             if p[-1] == sn and _classify_sn(p[-1]) != 'logit']

        if not prefixes_here:
            continue

        out_weights = {}
        out_total   = 0.0
        for j in range(K):
            if j == i:
                continue
            w = float(sn_adj[i, j])
            if w > 0:
                out_weights[sn_names[j]] = w
                out_total += w

        direct_inf = float(sn_inf[i]) if kind == 'middle' else 0.0
        exit_total = out_total + max(0.0, direct_inf)

        if exit_total <= 0:
            continue

        for prefix, flow in prefixes_here:
            if prefix in path_flows:
                del path_flows[prefix]

            for sn_next, w in out_weights.items():
                distributed = flow * (w / exit_total)
                if distributed < prune_threshold:
                    continue
                new_path = prefix + (sn_next,)
                path_flows[new_path] = path_flows.get(new_path, 0.0) + distributed

            if direct_inf > 0:
                inf_flow = flow * (direct_inf / exit_total)
                if inf_flow >= prune_threshold:
                    for sn_l in logit_sns:
                        exit_path = prefix + (sn_l,)
                        completed_paths[exit_path] = (
                            completed_paths.get(exit_path, 0.0) + inf_flow
                        )

    for path, flow in path_flows.items():
        if _classify_sn(path[-1]) == 'logit':
            completed_paths[path] = completed_paths.get(path, 0.0) + flow

    sorted_paths = sorted(completed_paths.items(), key=lambda x: -x[1])
    total_flow   = sum(f for _, f in sorted_paths) if sorted_paths else 0.0

    concentration = {}
    cumulative    = 0.0
    for rank, (path, flow) in enumerate(sorted_paths, 1):
        cumulative += flow
        frac = cumulative / (total_flow + 1e-12)
        concentration[rank] = frac
        if rank >= top_k and frac > 0.99:
            break

    top_k_frac = concentration.get(min(top_k, len(sorted_paths)), 0.0)
    D_phi      = 1.0 - top_k_frac

    paths_out = [
        {
            'path':   list(path),
            'flow':   flow,
            'frac':   flow / (total_flow + 1e-12),
            'length': len(path),
        }
        for path, flow in sorted_paths[:top_k]
    ]

    return dict(
        paths         = paths_out,
        total_flow    = total_flow,
        n_paths       = len(sorted_paths),
        concentration = {str(k): v for k, v in concentration.items()},
        D_phi         = D_phi,
        top_k         = top_k,
        top_k_frac    = top_k_frac,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOCAL FLOW RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

def local_flow_residuals(
    sng: dict,
    final_supernodes: dict,
) -> dict:
    """
    At each middle supernode S, compute signed in/out flow and residual.

    Nodes are split into two groups before aggregation:

      Non-suppressive (inf_exit >= 0):
        residual_rel = |in_net - total_out| / (|in_net| + ε)
        Aggregated into R_phi_balance — the primary metric for grouping quality.
        A high R_phi_balance means flow enters a concept but doesn't leave,
        indicating the grouping is too coarse or misaligned.

      Suppressive (inf_exit < 0):
        suppression_rel = |inf_exit| / (|in_net| + ε)
        Aggregated into R_phi_suppressive — reported separately.
        These nodes genuinely inhibit the logit; their negative exit is not
        a grouping artifact, so they must not inflate R_phi_balance.

    Returns:
        per_sn           : dict with per-supernode stats
        R_phi_balance    : float, mean residual over non-suppressive SNs
        R_phi_suppressive: float, mean suppression ratio over suppressive SNs
        R_phi_max        : float, worst-case residual (non-suppressive only)
        n_middle         : int, total middle SNs
        n_suppressive    : int, count of suppressive SNs
    """
    sn_names = sng['sn_names']
    sn_adj   = sng['sn_adj']
    sn_inf   = sng['sn_inf']
    K        = len(sn_names)

    name2idx = {sn: i for i, sn in enumerate(sn_names)}

    per_sn = {}
    balance_residuals     = []   # non-suppressive nodes
    suppressive_ratios    = []   # suppressive nodes

    for sn in sn_names:
        kind = _classify_sn(sn)
        if kind != 'middle':
            continue

        i = name2idx[sn]

        in_flow_pos  = sum(max(0.0, float(sn_adj[j, i])) for j in range(K) if j != i)
        in_flow_neg  = sum(min(0.0, float(sn_adj[j, i])) for j in range(K) if j != i)
        out_flow_pos = sum(max(0.0, float(sn_adj[i, j])) for j in range(K) if j != i)
        out_flow_neg = sum(min(0.0, float(sn_adj[i, j])) for j in range(K) if j != i)

        inf_exit     = float(sn_inf[i])
        in_flow_net  = in_flow_pos + in_flow_neg
        total_out    = out_flow_pos + out_flow_neg + inf_exit

        residual_abs = abs(in_flow_net - total_out)
        residual_rel = residual_abs / (abs(in_flow_net) + 1e-12)

        is_suppressive = inf_exit < 0

        if is_suppressive:
            # Suppression ratio: how much of in-flow is consumed by inhibition
            suppression_rel = abs(inf_exit) / (abs(in_flow_net) + 1e-12)
            suppressive_ratios.append(suppression_rel)
        else:
            balance_residuals.append(residual_rel)

        per_sn[sn] = dict(
            in_flow_pos      = in_flow_pos,
            in_flow_neg      = in_flow_neg,
            in_flow_net      = in_flow_net,
            out_flow_pos     = out_flow_pos,
            out_flow_neg     = out_flow_neg,
            inf_exit         = inf_exit,
            total_out        = total_out,
            residual_abs     = residual_abs,
            residual_rel     = residual_rel,
            balance          = in_flow_net - total_out,
            is_suppressive   = is_suppressive,
            suppression_rel  = abs(inf_exit) / (abs(in_flow_net) + 1e-12)
                               if is_suppressive else None,
        )

    R_phi_balance     = float(np.mean(balance_residuals))     if balance_residuals     else 0.0
    R_phi_suppressive = float(np.mean(suppressive_ratios))    if suppressive_ratios    else 0.0
    R_phi_max         = float(np.max(balance_residuals))      if balance_residuals     else 0.0

    return dict(
        per_sn            = per_sn,
        R_phi_balance     = R_phi_balance,
        R_phi_suppressive = R_phi_suppressive,
        R_phi_max         = R_phi_max,
        # backward-compat alias so flow_faithfulness_score doesn't break
        R_phi             = R_phi_balance,
        n_middle          = len(per_sn),
        n_suppressive     = len(suppressive_ratios),
        n_balanced        = len(balance_residuals),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SHORTCUT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def shortcut_analysis(
    sng: dict,
    final_supernodes: dict,
    min_edge_weight: float = 1e-6,
) -> dict:
    sn_names = sng['sn_names']
    sn_adj   = sng['sn_adj']
    K        = len(sn_names)

    edges = []
    total_weight    = 0.0
    shortcut_weight = 0.0

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            w_direct = float(sn_adj[i, j])
            if w_direct < min_edge_weight:
                continue

            best_mediation = 0.0
            best_mediator  = None

            for b in range(K):
                if b == i or b == j:
                    continue
                w_ab = float(sn_adj[i, b])
                w_bc = float(sn_adj[b, j])
                if w_ab > 0 and w_bc > 0:
                    mediation = min(w_ab, w_bc)
                    if mediation > best_mediation:
                        best_mediation = mediation
                        best_mediator  = sn_names[b]

            ratio = w_direct / (w_direct + best_mediation + 1e-12)

            is_shortcut = ratio < 0.5
            total_weight += w_direct
            if is_shortcut:
                shortcut_weight += w_direct

            edges.append(dict(
                src                = sn_names[i],
                tgt                = sn_names[j],
                weight             = w_direct,
                shortcut_ratio     = ratio,
                is_shortcut        = is_shortcut,
                best_mediator      = best_mediator,
                mediation_strength = best_mediation,
            ))

    edges.sort(key=lambda e: -e['weight'])

    n_shortcuts = sum(1 for e in edges if e['is_shortcut'])
    n_direct    = sum(1 for e in edges if not e['is_shortcut'])

    return dict(
        edges                = edges,
        n_shortcuts          = n_shortcuts,
        n_direct             = n_direct,
        n_total              = len(edges),
        global_shortcut_frac = shortcut_weight / (total_weight + 1e-12),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMBINED FLOW FAITHFULNESS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def flow_faithfulness_score(
    path_result: dict,
    residual_result: dict,
    shortcut_result: dict,
    w_concentration: float = 0.40,
    w_residual:      float = 0.30,
    w_shortcut:      float = 0.30,
) -> dict:
    """
    Combined flow faithfulness score F(φ) ∈ [0, 1].

    residual_score uses R_phi_balance only (non-suppressive nodes).
    R_phi_suppressive is reported but does not affect F(phi).
    """
    path_score     = 1.0 - path_result['D_phi']
    residual_score = 1.0 - min(1.0, residual_result['R_phi_balance'])
    shortcut_score = 1.0 - shortcut_result['global_shortcut_frac']

    F_phi = (w_concentration * path_score
             + w_residual    * residual_score
             + w_shortcut    * shortcut_score)

    return dict(
        F_phi             = F_phi,
        path_score        = path_score,
        residual_score    = residual_score,
        shortcut_score    = shortcut_score,
        D_phi             = path_result['D_phi'],
        R_phi             = residual_result['R_phi_balance'],      # compat alias
        R_phi_balance     = residual_result['R_phi_balance'],
        R_phi_suppressive = residual_result['R_phi_suppressive'],
        R_phi_max         = residual_result['R_phi_max'],
        shortcut_frac     = shortcut_result['global_shortcut_frac'],
        n_paths           = path_result['n_paths'],
        top_k_frac        = path_result['top_k_frac'],
        n_shortcuts       = shortcut_result['n_shortcuts'],
        n_direct          = shortcut_result['n_direct'],
        n_suppressive     = residual_result['n_suppressive'],
        n_balanced        = residual_result['n_balanced'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def flow_faithfulness_report(
    sng: dict,
    final_supernodes: dict,
    top_k: int = 10,
) -> dict:
    path_result     = path_attribution_decomposition(sng, final_supernodes, top_k=top_k)
    residual_result = local_flow_residuals(sng, final_supernodes)
    shortcut_result = shortcut_analysis(sng, final_supernodes)
    combined        = flow_faithfulness_score(path_result, residual_result, shortcut_result)

    return dict(
        path_decomposition = path_result,
        local_residuals    = residual_result,
        shortcut_analysis  = shortcut_result,
        combined           = combined,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_flow_report(report: dict) -> None:
    SEP = '─' * 72

    print(f'\n{"═"*72}')
    print('  FLOW FAITHFULNESS REPORT')
    print(f'{"═"*72}')

    combined = report['combined']
    print(f'\n{SEP}')
    print('  COMBINED FLOW FAITHFULNESS SCORE')
    print(SEP)
    print(f'  F(φ) = {combined["F_phi"]:.4f}')
    print(f'    path_score     = {combined["path_score"]:.4f}'
          f'  (1 - D_phi, top-{report["path_decomposition"]["top_k"]} concentration)')
    print(f'    residual_score = {combined["residual_score"]:.4f}'
          f'  (1 - R_phi_balance, non-suppressive nodes only)')
    print(f'    shortcut_score = {combined["shortcut_score"]:.4f}'
          f'  (1 - shortcut_frac, genuine direct links)')
    print()
    F = combined['F_phi']
    if F > 0.8:
        verdict = '[EXCELLENT] Supernode graph is a faithful flow abstraction.'
    elif F > 0.6:
        verdict = '[GOOD] Supernode graph captures main flow structure with some leakage.'
    elif F > 0.4:
        verdict = '[FAIR] Flow structure partially preserved — consider adjusting k or layer span.'
    else:
        verdict = '[POOR] Flow structure not well preserved — grouping may be too coarse.'
    print(f'  {verdict}')

    # ── Path decomposition ────────────────────────────────────────────────────
    pd = report['path_decomposition']
    print(f'\n{SEP}')
    print('  PATH ATTRIBUTION DECOMPOSITION')
    print(SEP)
    print(f'  Total paths found: {pd["n_paths"]}')
    print(f'  Total flow: {pd["total_flow"]:.6f}')
    print(f'  D(φ) = {pd["D_phi"]:.4f}'
          f'  (flow distortion: 1 - top-{pd["top_k"]} concentration)')
    print(f'  Top-{pd["top_k"]} paths capture {pd["top_k_frac"]*100:.1f}% of total flow')
    print()
    print(f'  {"Rank":>4}  {"Flow":>10}  {"Frac":>7}  {"Len":>3}  Path')
    print(f'  {"-"*68}')
    for rank, p in enumerate(pd['paths'], 1):
        path_str = ' → '.join(p['path'])
        print(f'  {rank:>4}  {p["flow"]:>10.6f}  {p["frac"]*100:>6.1f}%  '
              f'{p["length"]:>3}  {path_str}')

    print(f'\n  Concentration curve:')
    for k_str, frac in sorted(pd['concentration'].items(), key=lambda x: int(x[0])):
        k = int(k_str)
        bar = '█' * int(frac * 40)
        print(f'    top-{k:>2}: {frac*100:>6.1f}%  {bar}')
        if k >= 15:
            break

    # ── Local flow residuals ──────────────────────────────────────────────────
    lr = report['local_residuals']
    print(f'\n{SEP}')
    print('  LOCAL FLOW RESIDUALS')
    print(SEP)
    print(f'  R_phi_balance     = {lr["R_phi_balance"]:.4f}'
          f'  (mean residual, {lr["n_balanced"]} non-suppressive SNs)'
          f'  ← used in F(φ)')
    print(f'  R_phi_suppressive = {lr["R_phi_suppressive"]:.4f}'
          f'  (mean suppression ratio, {lr["n_suppressive"]} suppressive SNs)'
          f'  ← reported only')
    print(f'  R_phi_max         = {lr["R_phi_max"]:.4f}'
          f'  (worst-case, non-suppressive only)')
    print(f'  Middle supernodes : {lr["n_middle"]} total'
          f'  ({lr["n_balanced"]} balanced, {lr["n_suppressive"]} suppressive)')

    # Non-suppressive nodes
    balanced_sn = {sn: st for sn, st in lr['per_sn'].items()
                   if not st['is_suppressive']}
    if balanced_sn:
        print(f'\n  Non-suppressive nodes (contribute to R_phi_balance):')
        print(f'  {"Supernode":<22}  {"in_net":>9}  {"out_net":>9}  {"inf_exit":>9}'
              f'  {"residual":>9}  {"balance":>9}  Status')
        print(f'  {"-"*86}')
        for sn, st in sorted(balanced_sn.items(), key=lambda x: -x[1]['residual_rel']):
            status = ('[OK]'   if st['residual_rel'] < 0.3 else
                      '[WARN]' if st['residual_rel'] < 0.6 else '[HIGH]')
            print(f'  {sn:<22}  {st["in_flow_net"]:>+9.4f}  {st["total_out"]:>+9.4f}'
                  f'  {st["inf_exit"]:>+9.4f}  {st["residual_rel"]:>9.4f}'
                  f'  {st["balance"]:>+9.4f}  {status}')

    # Suppressive nodes
    suppressive_sn = {sn: st for sn, st in lr['per_sn'].items()
                      if st['is_suppressive']}
    if suppressive_sn:
        print(f'\n  Suppressive nodes (inf_exit < 0, reported separately):')
        print(f'  {"Supernode":<22}  {"in_net":>9}  {"out_net":>9}  {"inf_exit":>9}'
              f'  {"suppr_ratio":>11}  {"balance":>9}')
        print(f'  {"-"*86}')
        for sn, st in sorted(suppressive_sn.items(),
                              key=lambda x: x[1]['suppression_rel'], reverse=True):
            print(f'  {sn:<22}  {st["in_flow_net"]:>+9.4f}  {st["total_out"]:>+9.4f}'
                  f'  {st["inf_exit"]:>+9.4f}  {st["suppression_rel"]:>11.4f}'
                  f'  {st["balance"]:>+9.4f}')

    # ── Shortcut analysis ─────────────────────────────────────────────────────
    sa = report['shortcut_analysis']
    print(f'\n{SEP}')
    print('  SHORTCUT ANALYSIS')
    print(SEP)
    print(f'  Total edges: {sa["n_total"]}')
    print(f'  Direct (ratio ≥ 0.5): {sa["n_direct"]}')
    print(f'  Shortcuts (ratio < 0.5): {sa["n_shortcuts"]}')
    print(f'  Global shortcut fraction: {sa["global_shortcut_frac"]*100:.1f}%'
          f' of total edge weight')
    print()

    top_edges = sa['edges'][:15]
    print(f'  {"Source":<20}  {"Target":<20}  {"Weight":>9}  '
          f'{"Ratio":>7}  {"Type":>10}  Mediator')
    print(f'  {"-"*90}')
    for e in top_edges:
        etype = 'DIRECT' if not e['is_shortcut'] else 'SHORTCUT'
        med   = e['best_mediator'] or '—'
        print(f'  {e["src"]:<20}  {e["tgt"]:<20}  {e["weight"]:>9.4f}'
              f'  {e["shortcut_ratio"]:>7.3f}  {etype:>10}  {med}')

    print(f'\n{"═"*72}\n')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INTEGRATION WITH AUTO_GROUPING: ENHANCED SCORE
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_score_k(
    final_supernodes: dict,
    data: dict,
    S: torch.Tensor,
    target_n_middle: int,
    w_intra: float = 0.20,
    w_dag:   float = 0.15,
    w_attr:  float = 0.15,
    w_size:  float = 0.10,
    w_flow:  float = 0.40,
) -> dict:
    from auto_grouping import score_k as base_score_k

    base = base_score_k(final_supernodes, data, S, target_n_middle,
                        w_intra=1.0, w_dag=1.0, w_attr=1.0, w_size=1.0)

    sng    = build_supernode_graph(final_supernodes, data)
    report = flow_faithfulness_report(sng, final_supernodes, top_k=10)

    F_phi = report['combined']['F_phi']

    total = (w_intra * base['intra_sim']
             + w_dag  * base['dag_safety']
             + w_attr * base['attr_balance']
             + w_size * base['size_score']
             + w_flow * F_phi)

    return dict(
        total             = total,
        intra_sim         = base['intra_sim'],
        dag_safety        = base['dag_safety'],
        attr_balance      = base['attr_balance'],
        size_score        = base['size_score'],
        F_phi             = F_phi,
        D_phi             = report['combined']['D_phi'],
        R_phi             = report['combined']['R_phi_balance'],
        R_phi_balance     = report['combined']['R_phi_balance'],
        R_phi_suppressive = report['combined']['R_phi_suppressive'],
        shortcut_frac     = report['combined']['shortcut_frac'],
        path_score        = report['combined']['path_score'],
        residual_score    = report['combined']['residual_score'],
        shortcut_score    = report['combined']['shortcut_score'],
        n_middle          = base['n_middle'],
        n_warnings        = base['n_warnings'],
        inf_conservation  = base.get('inf_conservation', 0.0),
        edge_conservation = base.get('edge_conservation', 0.0),
        n_paths           = report['path_decomposition']['n_paths'],
        top_k_frac        = report['path_decomposition']['top_k_frac'],
        n_suppressive     = report['combined']['n_suppressive'],
        flow_report       = report,
        final_supernodes  = final_supernodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENHANCED AUTO-K SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_best_k_with_flow(
    data: dict,
    S: torch.Tensor,
    max_layer_span: int = 4,
    k_min: int = None,
    k_max: int = None,
    max_sn: int = None,
    weights: dict = None,
) -> tuple:
    from auto_grouping import eigengap_analysis

    kept_ids   = data['kept_ids']
    middle_ids = [nid for nid in kept_ids if not _is_fixed(nid)]
    N_middle   = len(middle_ids)

    if N_middle < 3:
        print('  Too few middle nodes for auto-k.')
        return 2, {}

    eg = eigengap_analysis(S, kept_ids, max_k=min(20, N_middle - 1))
    print(f'  Eigengap suggests k = {eg["eigengap_k"]}')

    k_lo = k_min if k_min is not None else eg['search_range'][0]
    k_hi = k_max if k_max is not None else eg['search_range'][1]
    k_lo = max(2, k_lo)
    k_hi = min(N_middle - 1, k_hi)
    if k_lo > k_hi:
        k_lo = k_hi

    w = weights or {}
    print(f'\n  Sweeping k = {k_lo}..{k_hi} with flow-enhanced scoring')
    print(f'  {"k":>3}  {"n_sn":>4}  {"intra":>6}  {"dag":>5}  {"attr":>5}'
          f'  {"size":>5}  {"F_phi":>6}  {"D_phi":>6}  {"R_bal":>6}'
          f'  {"R_sup":>6}  {"short":>6}  {"TOTAL":>6}')
    print(f'  {"─"*84}')

    results = {}
    for k in range(k_lo, k_hi + 1):
        try:
            final_sn = cluster_with_target_k(
                data, S, target_k=k,
                max_layer_span=max_layer_span, max_sn=max_sn)
        except Exception as e:
            print(f'  k={k} failed: {e}')
            continue

        sc = enhanced_score_k(
            final_sn, data, S, N_middle,
            w_intra = w.get('w_intra', 0.20),
            w_dag   = w.get('w_dag',   0.15),
            w_attr  = w.get('w_attr',  0.15),
            w_size  = w.get('w_size',  0.10),
            w_flow  = w.get('w_flow',  0.40),
        )
        results[k] = sc

        n_total  = len(final_sn)
        n_middle = sc['n_middle']
        print(f'  {k:>3}  {n_middle:>2}+{n_total-n_middle:<2}  {sc["intra_sim"]:>6.4f}'
              f'  {sc["dag_safety"]:>5.3f}  {sc["attr_balance"]:>5.3f}'
              f'  {sc["size_score"]:>5.3f}  {sc["F_phi"]:>6.4f}'
              f'  {sc["D_phi"]:>6.4f}  {sc["R_phi_balance"]:>6.4f}'
              f'  {sc["R_phi_suppressive"]:>6.4f}  {sc["shortcut_frac"]:>6.3f}'
              f'  {sc["total"]:>6.4f}')

    if not results:
        return eg['eigengap_k'], {}

    best_k = max(results, key=lambda k: results[k]['total'])
    best   = results[best_k]

    n_total  = len(results[best_k]['final_supernodes'])
    n_middle = best['n_middle']
    print(f'\n  Best k = {best_k}  (total = {best["total"]:.4f})')
    print(f'    Supernodes: {n_middle} middle + {n_total - n_middle} fixed = {n_total} total')
    print(f'    F(φ) = {best["F_phi"]:.4f}  '
          f'(path={best["path_score"]:.3f}, '
          f'residual={best["residual_score"]:.3f}, '
          f'shortcut={best["shortcut_score"]:.3f})')
    print(f'    R_phi_balance     = {best["R_phi_balance"]:.4f}'
          f'  ({best["n_middle"] - best["n_suppressive"]} non-suppressive SNs)')
    print(f'    R_phi_suppressive = {best["R_phi_suppressive"]:.4f}'
          f'  ({best["n_suppressive"]} suppressive SNs)')
    print(f'    inf_conservation  = {best["inf_conservation"]:.6f}')
    print(f'    edge_conservation = {best["edge_conservation"]:.6f}')

    return best_k, results


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_flow_report(report: dict, out_path: str = 'flow_analysis.json') -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(report), f, indent=2)
    print(f'Flow report saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Flow faithfulness analysis for supernode abstractions')
    parser.add_argument('--file',           type=str,   default=None)
    parser.add_argument('--synthetic',      action='store_true')
    parser.add_argument('--target-k',       type=int,   default=7)
    parser.add_argument('--max-layer-span', type=int,   default=4)
    parser.add_argument('--alpha',          type=float, default=0.5)
    parser.add_argument('--beta',           type=float, default=0.5)
    parser.add_argument('--top-k-paths',    type=int,   default=10)
    parser.add_argument('--out-json',       type=str,   default='flow_analysis.json')
    parser.add_argument('--auto-k',         action='store_true')
    parser.add_argument('--k-min',          type=int,   default=None)
    parser.add_argument('--k-max',          type=int,   default=None)
    parser.add_argument('--max-sn',         type=int,   default=None)
    parser.add_argument('--mediation-penalty', type=float, default=0.1)
    args = parser.parse_args()

    if args.synthetic:
        print('Using synthetic snapshot...')
        raw = build_synthetic_snapshot()
    elif args.file:
        print(f'Loading {args.file}...')
        raw = load_snapshot(args.file)
    else:
        parser.error('Provide --file or --synthetic')

    data = prepare_graph_data(raw)
    print(f'  Nodes: {len(data["kept_ids"])}')
    print(f'  logit_idx: {data["logit_idx"]}')
    print(f'  total adj[:,logit] = {data["adj"][:, data["logit_idx"]].sum():.4f}')

    print('Computing similarity matrix...')
    S = compute_similarity(data, alpha=args.alpha, beta=args.beta,
                           mediation_penalty=args.mediation_penalty)

    if args.auto_k:
        best_k, results = find_best_k_with_flow(
            data, S,
            max_layer_span=args.max_layer_span,
            k_min=args.k_min, k_max=args.k_max,
            max_sn=args.max_sn,
        )
        if best_k in results and 'flow_report' in results[best_k]:
            print_flow_report(results[best_k]['flow_report'])
            save_flow_report(results[best_k]['flow_report'], args.out_json)
            best_sn = results[best_k]['final_supernodes']
            sng = build_supernode_graph(best_sn, data)
            sn_flow_path = args.out_json.replace('.json', '_sn_flow.json')
            with open(sn_flow_path, 'w') as f:
                import json as _json
                _json.dump({
                    'sn_names': sng['sn_names'],
                    'sn_adj': sng['sn_adj'].tolist(),
                    'F_sn': sng['F_sn'].tolist(),
                    'sn_reach': sng['sn_reach'].tolist(),
                    'sn_act_norm': sng['sn_act_norm'].tolist(),
                    'sn_inf': sng['sn_inf'].tolist(),
                    'preservation': sng['preservation'],
                    'orig_reach_total': sng['orig_reach_total'],
                    'surr_reach_total': sng['surr_reach_total'],
                    'inf_conservation': sng['inf_conservation'],
                    'edge_conservation': sng['edge_conservation'],
                    'dominant_paths': sng['dominant_paths'],
                    'bottleneck_sns': sng['bottleneck_sns'],
                }, f, indent=2)
            print(f'SN flow JSON saved → {sn_flow_path}')
            sn_map_path = args.out_json.replace('.json', '_supernode_map.json')
            with open(sn_map_path, 'w') as f:
                import json as _json
                _json.dump(best_sn, f, indent=2)
            n_mid = sum(1 for s in best_sn if 'EMB' not in s and 'LOGIT' not in s)
            n_fix = len(best_sn) - n_mid
            print(f'Supernode map saved → {sn_map_path}  ({n_mid} middle + {n_fix} fixed)')
        return

    print(f'\nClustering with target-k={args.target_k}...')
    final_sn = cluster_with_target_k(
        data, S, target_k=args.target_k,
        max_layer_span=args.max_layer_span, max_sn=args.max_sn)

    n_middle = sum(1 for sn in final_sn if 'EMB' not in sn and 'LOGIT' not in sn)
    n_total  = len(final_sn)
    print(f'  Supernodes: {n_middle} middle + {n_total - n_middle} fixed = {n_total} total')

    print('\nBuilding supernode graph...')
    sng = build_supernode_graph(final_sn, data)
    print(f'  inf_conservation  = {sng["inf_conservation"]:.6f}')
    print(f'  edge_conservation = {sng["edge_conservation"]:.6f}')

    print('\nRunning flow faithfulness analysis...')
    report = flow_faithfulness_report(sng, final_sn, top_k=args.top_k_paths)

    print_flow_report(report)
    save_flow_report(report, args.out_json)


if __name__ == '__main__':
    main()