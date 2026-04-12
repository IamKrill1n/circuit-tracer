"""
visualize_circuit_sp_rep.py
─────────────────────────────────────────────────────────────────────────────
Enhanced circuit graph visualization with supernode overlay + repositioning
+ supernode flow surrogate view.

Loads:
  - subgraph/austin_plt.pt      (raw circuit graph)
  - supernode_map.json          (output of structure_grouping.py)
  - supernode_map_sn_flow.json  (output of structure_grouping.py --sn-flow)
                                 Optional: enables SN Flow surrogate view.

Modes (three-way toggle):
  RAW        Original node graph, colored by node_inf gradient.
  SUPERNODE  Raw nodes colored by supernode membership, convex-hull regions,
             sidebar membership list, reposition-by-cluster button.
  SN FLOW    Surrogate graph: each supernode = one large node.
             Edges = SN→SN flow (F_sn matrix). Node size ∝ reach-to-logit.
             Shows preservation ratio, dominant paths, bottlenecks.

Usage:
  python visualize_circuit_sp_rep.py \\
      --file subgraph/austin_plt.pt \\
      --supernode supernode_map.json \\
      --sn-flow supernode_map_sn_flow.json \\
      --out circuit_sp_rep.html
"""

import json
import argparse
import torch


def parse_layer(nid):
    if nid.startswith('E'): return 0
    if nid.startswith('27'): return 27
    return int(nid.split('_')[0])


def parse_activation(attr_node):
    a = attr_node['activation']
    if a is not None: return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100


def load_snapshot(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def prepare_graph_data(raw):
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float().T
    attr      = raw['attr']
    node_inf  = raw['node_inf'].float()
    node_rel  = raw['node_rel'].float()
    logit_idx = len(kept_ids) - 1

    layers = [parse_layer(n) for n in kept_ids]
    acts   = [parse_activation(attr[n]) for n in kept_ids]

    nodes = []
    for i, nid in enumerate(kept_ids):
        a = attr[nid]
        layer = layers[i]
        is_emb = nid.startswith('E')
        is_log = a.get('is_target_logit', False)
        if is_emb:   ntype = 'embedding'
        elif is_log: ntype = 'logit'
        else:        ntype = 'transcoder'
        nodes.append({
            'id':           nid,
            'idx':          i,
            'layer':        layer,
            'clerp':        a['clerp'],
            'activation':   acts[i],
            'influence':    float(a.get('influence') or 0),
            'inf_to_logit': float(a.get('influence') or 0),
            'node_inf':     float(node_inf[i]),
            'node_rel':     float(node_rel[i]),
            'type':         ntype,
            'ctx_idx':      a.get('ctx_idx', 0),
        })

    act_tensor = torch.tensor(acts, dtype=torch.float32)
    flow_mat   = act_tensor.unsqueeze(1) * adj
    N          = len(kept_ids)
    logit_id   = kept_ids[logit_idx]

    edges = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            w    = adj[i, j].item()
            flow = flow_mat[i, j].item()
            if w == 0.0: continue
            edges.append({
                'source':   kept_ids[i],
                'target':   kept_ids[j],
                'weight':   w,
                'flow':     flow,
                'abs_flow': abs(flow),
                'to_logit': kept_ids[j] == logit_id,
            })

    print(f"  Total edges:    {len(edges)}")
    print(f"  Edges to logit (real circuit only): {sum(1 for e in edges if e['to_logit'])}")
    return {'nodes': nodes, 'edges': edges}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Circuit Graph — Austin · Supernode Reposition</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:        #060910;
  --bg2:       #0b1020;
  --border:    #1a2540;
  --text:      #c8d4e8;
  --dim:       #3a4a60;
  --accent:    #3d8eff;
  --gold:      #f5a623;
  --red:       #f43f5e;
  --green:     #10b981;
  --purple:    #8b5cf6;
}

* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:'JetBrains Mono',monospace; overflow:hidden; }
#app { display:flex; height:100vh; width:100vw; }

/* ── Sidebar ── */
#sidebar {
  width:300px; min-width:300px;
  background:var(--bg2);
  border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow:hidden;
}
#sidebar-header { padding:18px 16px 14px; border-bottom:1px solid var(--border); }
#sidebar-header h2 {
  font-family:'Syne',sans-serif; font-weight:800; font-size:13px;
  letter-spacing:.08em; color:var(--accent); margin-bottom:3px;
}
#sidebar-header p { font-size:9px; color:var(--dim); letter-spacing:.06em; }

/* ── Mode buttons ── */
#btn-wrap {
  padding:12px 16px; border-bottom:1px solid var(--border);
  display:flex; flex-direction:column; gap:6px;
}
.mode-btn {
  width:100%; padding:8px 12px;
  background:transparent; border:1px solid var(--border); border-radius:4px;
  color:var(--dim); font-family:'JetBrains Mono',monospace; font-size:10px;
  letter-spacing:.08em; cursor:pointer;
  display:flex; align-items:center; gap:8px; transition:all 0.2s;
}
.mode-btn:hover { border-color:var(--accent); color:var(--accent); }
.mode-btn.active { background:rgba(61,142,255,0.08); border-color:var(--accent); color:var(--accent); }
.mode-btn .dot {
  width:7px; height:7px; border-radius:50%; background:var(--dim);
  flex-shrink:0; transition:background 0.2s;
}
.mode-btn.active .dot { background:var(--accent); }
.mode-btn:disabled { opacity:0.3; cursor:not-allowed; }
.mode-btn:disabled:hover { border-color:var(--border); color:var(--dim); }

/* ── Controls ── */
#controls {
  padding:12px 16px; border-bottom:1px solid var(--border);
  display:flex; flex-direction:column; gap:8px;
}
.control-row { display:flex; align-items:center; gap:8px; font-size:9px; color:var(--dim); }
.control-row label { min-width:78px; letter-spacing:.04em; }
input[type=range] { flex:1; accent-color:var(--accent); height:2px; }
.val-display { min-width:28px; text-align:right; color:var(--accent); font-size:9px; }

/* ── Legend ── */
#legend { padding:10px 16px; border-bottom:1px solid var(--border); }
#legend h3 { font-size:8px; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:7px; }
.legend-item { display:flex; align-items:center; gap:7px; margin-bottom:4px; font-size:9px; color:var(--dim); }
.legend-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

/* ── Supernode legend ── */
#sn-legend {
  padding:10px 16px; border-bottom:1px solid var(--border);
  display:none; max-height:200px; overflow-y:auto;
}
#sn-legend.visible { display:block; }
#sn-legend h3 { font-size:8px; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:7px; }
.sn-legend-item {
  display:flex; align-items:center; gap:7px; margin-bottom:4px;
  font-size:9px; color:var(--dim); cursor:pointer;
  padding:2px 4px; border-radius:3px; transition:background 0.15s;
}
.sn-legend-item:hover { background:rgba(255,255,255,0.04); color:var(--text); }
.sn-legend-item .sn-dot { width:8px; height:8px; border-radius:2px; flex-shrink:0; }
.sn-legend-item .sn-count { margin-left:auto; opacity:.5; font-size:8px; }

/* ── SN Flow stats panel ── */
#sn-flow-panel {
  padding:10px 16px; border-bottom:1px solid var(--border);
  display:none; overflow-y:auto; max-height:220px;
}
#sn-flow-panel.visible { display:block; }
#sn-flow-panel h3 { font-size:8px; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:8px; }
.flow-stat-row { display:flex; justify-content:space-between; margin-bottom:4px; font-size:9px; }
.flow-stat-label { color:var(--dim); }
.flow-stat-value { color:var(--accent); }
.flow-badge {
  display:inline-block; padding:2px 8px; border-radius:3px;
  font-size:8px; letter-spacing:.04em; margin-bottom:6px;
}
.flow-pass { background:rgba(16,185,129,0.1); color:#10b981; border:1px solid rgba(16,185,129,0.3); }
.flow-warn { background:rgba(245,166,35,0.1); color:#f5a623; border:1px solid rgba(245,166,35,0.3); }
.flow-fail { background:rgba(244,63,94,0.1); color:#f43f5e; border:1px solid rgba(244,63,94,0.3); }
.bottleneck-tag {
  display:inline-block; padding:1px 5px; border-radius:2px; font-size:7px;
  background:rgba(244,63,94,0.12); color:#f43f5e;
  border:1px solid rgba(244,63,94,0.25); margin-left:4px;
}
.dominant-edge-row {
  font-size:8px; color:var(--dim); padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.03);
  display:flex; justify-content:space-between;
}
.dominant-edge-row span { color:var(--text); }

/* ── Node info panel ── */
#node-info { flex:1; overflow-y:auto; padding:12px 16px; }
#node-info h3 { font-size:8px; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:10px; }
.info-row { display:flex; justify-content:space-between; margin-bottom:4px; font-size:9px; }
.info-label { color:var(--dim); }
.info-value { color:var(--text); }
.info-clerp {
  font-size:10px; color:var(--accent); margin-bottom:10px; line-height:1.5;
  padding:8px 10px; background:rgba(61,142,255,0.06); border-radius:3px;
  border-left:2px solid var(--accent);
}
.info-sn-badge {
  display:inline-block; padding:2px 7px; border-radius:3px;
  font-size:8px; letter-spacing:.05em; margin-bottom:8px;
}
.bar-container { margin-top:4px; margin-bottom:5px; }
.bar-label { font-size:8px; color:var(--dim); margin-bottom:2px; }
.bar-track { height:3px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:2px; transition:width 0.3s; }

/* ── Canvas ── */
#canvas { flex:1; position:relative; overflow:hidden; }
svg { width:100%; height:100%; }
.node circle { cursor:pointer; transition:r 0.15s; }
.node:hover circle { filter:brightness(1.3); }
.node text { pointer-events:none; font-family:'JetBrains Mono',monospace; }
.link { fill:none; stroke-linecap:round; }
.layer-line { stroke-dasharray:2,8; }
.sn-hull {
  fill-opacity:0.06; stroke-width:1.5; stroke-opacity:0.35;
  pointer-events:none; transition:fill-opacity 0.3s;
}
.sn-label {
  pointer-events:none; font-family:'Syne',sans-serif;
  font-size:10px; font-weight:600; letter-spacing:.06em;
  opacity:0; transition:opacity 0.3s;
}
.sn-label.visible { opacity:0.55; }

/* SN Flow mode node styles */
.sn-node { cursor:pointer; }
.sn-node circle { transition:r 0.2s, filter 0.2s; }
.sn-node:hover circle { filter:brightness(1.4) drop-shadow(0 0 8px currentColor); }
.sn-node text { pointer-events:none; font-family:'Syne',sans-serif; }
.sn-flow-edge { fill:none; stroke-linecap:round; }

.clerp-tooltip {
  position:absolute; background:#0d1525; border:1px solid var(--border);
  color:var(--text); font-family:'JetBrains Mono',monospace; font-size:10px;
  padding:6px 10px; border-radius:4px; pointer-events:none;
  white-space:nowrap; z-index:100; opacity:0; transition:opacity 0.15s;
  max-width:350px; overflow:hidden; text-overflow:ellipsis;
}
.clerp-tooltip.show { opacity:1; }

#title-overlay {
  position:absolute; top:14px; left:14px; font-size:8px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--dim);
  pointer-events:none; font-family:'JetBrains Mono',monospace;
}
#mode-overlay {
  position:absolute; top:14px; right:14px; font-size:9px;
  letter-spacing:.08em; text-transform:uppercase;
  pointer-events:none; font-family:'Syne',sans-serif;
  font-weight:600; opacity:0.4;
}

::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <h2>Circuit Graph</h2>
      <p>Gemma-2-2B &middot; Austin prediction &middot; supernode flow</p>
    </div>

    <div id="btn-wrap">
      <button class="mode-btn active" id="btn-raw" onclick="setMode('raw')">
        <span class="dot"></span><span>RAW NODES</span>
      </button>
      <button class="mode-btn" id="btn-supernode" onclick="setMode('supernode')">
        <span class="dot"></span><span>SUPERNODE OVERLAY</span>
      </button>
      <button class="mode-btn" id="btn-snflow" onclick="setMode('snflow')"
              __SNFLOW_DISABLED__>
        <span class="dot"></span><span>SN FLOW SURROGATE</span>
      </button>
      <!-- Sub-button only relevant in supernode mode -->
      <button class="mode-btn" id="repo-toggle" onclick="toggleReposition()" disabled style="opacity:0.3;margin-top:2px;padding:6px 12px;font-size:9px">
        <span class="dot"></span><span id="repo-toggle-label">REPOSITION BY CLUSTER</span>
      </button>
    </div>

    <div id="controls">
      <div class="control-row">
        <label>Flow thresh</label>
        <input type="range" id="flow-thresh" min="0" max="500" value="10" step="5">
        <span class="val-display" id="flow-val">10</span>
      </div>
      <div class="control-row">
        <label>Node size</label>
        <input type="range" id="node-size" min="3" max="20" value="8" step="1">
        <span class="val-display" id="size-val">8</span>
      </div>
      <div class="control-row">
        <label>Edge opacity</label>
        <input type="range" id="edge-opacity" min="5" max="100" value="30" step="5">
        <span class="val-display" id="opacity-val">30</span>
      </div>
      <!-- SN Flow specific: edge weight threshold -->
      <div class="control-row" id="snflow-thresh-row" style="display:none">
        <label>SN edge min</label>
        <input type="range" id="sn-edge-thresh" min="0" max="100" value="0" step="1">
        <span class="val-display" id="sn-edge-val">0</span>
      </div>
    </div>

    <div id="legend">
      <h3>Legend</h3>
      <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Embedding</div>
      <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Transcoder low inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#3d8eff"></div>Transcoder high inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f43f5e"></div>Logit output</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f5a623"></div>Node size = influence</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Suppression (−)</div>
    </div>

    <div id="sn-legend">
      <h3>Supernodes</h3>
      <div id="sn-legend-items"></div>
    </div>

    <!-- SN Flow stats panel -->
    <div id="sn-flow-panel">
      <h3>Surrogate Flow Stats</h3>
      <div id="sn-flow-stats"></div>
    </div>

    <div id="node-info">
      <h3>Node inspector</h3>
      <p style="font-size:9px;color:var(--dim)">Click a node to inspect</p>
    </div>
  </div>

  <div id="canvas">
    <div id="title-overlay">prompt: "Fact: The capital of the state containing Dallas is"</div>
    <div id="mode-overlay">RAW</div>
    <div class="clerp-tooltip" id="tooltip"></div>
    <svg id="svg"></svg>
  </div>
</div>

<script>
const GRAPH_DATA    = __GRAPH_DATA__;
const SUPERNODE_MAP = __SUPERNODE_MAP__;
const SN_FLOW_DATA  = __SN_FLOW_DATA__;   // null if not provided

// ── Palette ──────────────────────────────────────────────────────────────────
const SN_PALETTE = [
  '#3d8eff','#f5a623','#10b981','#f43f5e','#8b5cf6',
  '#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6',
  '#a78bfa','#fb7185','#fbbf24','#34d399','#60a5fa',
];

const node2sn = {};
const snNames = Object.keys(SUPERNODE_MAP);
snNames.forEach(sn => SUPERNODE_MAP[sn].forEach(nid => { node2sn[nid] = sn; }));

const snColor = {};
let ci = 0;
snNames.forEach(sn => {
  if (sn === 'SN_EMB')   { snColor[sn] = '#f5a623'; return; }
  if (sn === 'SN_LOGIT') { snColor[sn] = '#f43f5e'; return; }
  snColor[sn] = SN_PALETTE[ci++ % SN_PALETTE.length];
});

// ── State ────────────────────────────────────────────────────────────────────
// mode: 'raw' | 'supernode' | 'snflow'
let mode            = 'raw';
let repositioned    = false;
let currentFlowThresh    = 10;
let currentSNEdgeThresh  = 0;
let currentNodeSize      = 8;
let currentOpacity       = 0.30;
let highlightedSN        = null;
let animating            = false;

let posLayer   = {};
let posCluster = {};

const svg     = d3.select('#svg');
const width   = () => document.getElementById('canvas').clientWidth;
const height  = () => document.getElementById('canvas').clientHeight;

const allLayers  = [...new Set(GRAPH_DATA.nodes.map(d => d.layer))].sort((a,b)=>a-b);
const layerNodes = {};
GRAPH_DATA.nodes.forEach(d => {
  if (!layerNodes[d.layer]) layerNodes[d.layer] = [];
  layerNodes[d.layer].push(d);
});
const nodeLayerMap = {};
GRAPH_DATA.nodes.forEach(d => { nodeLayerMap[d.id] = d.layer; });
const maxFlow = d3.max(GRAPH_DATA.edges.map(e => e.abs_flow)) || 1;
const tooltip = document.getElementById('tooltip');

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  ['raw','supernode','snflow'].forEach(id => {
    document.getElementById('btn-' + id).classList.toggle('active', id === m);
  });
  document.getElementById('mode-overlay').textContent =
    m === 'raw' ? 'RAW' : m === 'supernode' ? 'SUPERNODE' : 'SN FLOW SURROGATE';

  // Sidebar panels
  document.getElementById('legend').style.display       = m === 'raw' ? 'block' : 'none';
  document.getElementById('sn-legend').classList.toggle('visible', m === 'supernode');
  document.getElementById('sn-flow-panel').classList.toggle('visible', m === 'snflow');

  // Reposition sub-button
  const repoBtn = document.getElementById('repo-toggle');
  repoBtn.disabled = (m !== 'supernode');
  repoBtn.style.opacity = (m !== 'supernode') ? '0.3' : '1';

  // SN edge threshold control
  document.getElementById('snflow-thresh-row').style.display = m === 'snflow' ? 'flex' : 'none';

  // If leaving supernode mode, reset reposition
  if (m !== 'supernode' && repositioned) {
    repositioned = false;
    repoBtn.classList.remove('active');
    document.getElementById('repo-toggle-label').textContent = 'REPOSITION BY CLUSTER';
  }

  highlightedSN = null;
  render();
}

// ── Toggle reposition (supernode mode only) ──────────────────────────────────
function toggleReposition() {
  if (mode !== 'supernode' || animating) return;
  repositioned = !repositioned;
  const btn = document.getElementById('repo-toggle');
  btn.classList.toggle('active', repositioned);
  document.getElementById('repo-toggle-label').textContent =
    repositioned ? 'RESTORE LAYERS' : 'REPOSITION BY CLUSTER';
  animateTransition();
}

// ── Positions ─────────────────────────────────────────────────────────────────
function computeLayerPositions() {
  const W = width(), H = height();
  const layerX = d3.scalePoint().domain(allLayers).range([80, W-80]).padding(0.3);
  const pos = {};
  GRAPH_DATA.nodes.forEach(d => {
    const group = layerNodes[d.layer];
    const idx = group.indexOf(d), n = group.length;
    const margin = 50, step = (H - 2*margin) / Math.max(n, 1);
    pos[d.id] = { x: layerX(d.layer), y: n===1 ? H/2 : margin + idx*step + step/2 };
  });
  return pos;
}

function computeClusterPositions() {
  const W = width(), H = height();
  const pos = {};
  const snInfo = snNames.map(sn => {
    const members = SUPERNODE_MAP[sn];
    const layers  = members.map(nid => {
      const nd = GRAPH_DATA.nodes.find(n => n.id === nid);
      return nd ? nd.layer : 0;
    });
    const avgLayer = layers.reduce((a,b)=>a+b,0)/layers.length;
    return { sn, members, avgLayer };
  });
  snInfo.sort((a,b) => a.avgLayer - b.avgLayer);
  const xScale = d3.scaleLinear()
    .domain([d3.min(snInfo,d=>d.avgLayer), d3.max(snInfo,d=>d.avgLayer)])
    .range([140, W-140]);
  const rowH = H / (snInfo.length + 1);
  snInfo.forEach((info, si) => {
    const cx = xScale(info.avgLayer), cy = rowH*(si+1);
    const n = info.members.length;
    const cols = Math.ceil(Math.sqrt(n*1.8));
    const spacingX = 52, spacingY = 42;
    const startX = cx - (cols-1)*spacingX/2;
    const startY = cy - (Math.ceil(n/cols)-1)*spacingY/2;
    const sorted = [...info.members].sort((a,b)=>((nodeLayerMap[a]||0)-(nodeLayerMap[b]||0)) || a.localeCompare(b));
    sorted.forEach((nid, i) => {
      pos[nid] = { x: startX+(i%cols)*spacingX, y: startY+Math.floor(i/cols)*spacingY };
    });
  });
  return pos;
}

// ── SN Flow surrogate positions ───────────────────────────────────────────────
// Layout: supernodes arranged left→right by mean layer,
// vertically spread by average layer-within-group to reduce crossings.
function computeSNFlowPositions() {
  if (!SN_FLOW_DATA) return {};
  const W = width(), H = height();
  const sn_names = SN_FLOW_DATA.sn_names;

  // Compute mean layer per SN
  const snMeanLayer = {};
  sn_names.forEach(sn => {
    const members = SUPERNODE_MAP[sn] || [];
    const layers  = members.map(nid => nodeLayerMap[nid] || 0);
    snMeanLayer[sn] = layers.length ? layers.reduce((a,b)=>a+b,0)/layers.length : 0;
  });

  // Group by rounded mean layer band → vertical spread within band
  const layerBands = {};
  sn_names.forEach(sn => {
    const band = Math.round(snMeanLayer[sn]);
    if (!layerBands[band]) layerBands[band] = [];
    layerBands[band].push(sn);
  });
  const bands = Object.keys(layerBands).map(Number).sort((a,b)=>a-b);

  const xScale = d3.scaleLinear().domain([bands[0], bands[bands.length-1]]).range([100, W-100]);
  const pos = {};

  bands.forEach(band => {
    const members = layerBands[band];
    const x = xScale(band);
    const n = members.length;
    const step = H / (n+1);
    members.forEach((sn, i) => {
      pos[sn] = { x, y: step*(i+1) };
    });
  });

  return pos;
}

function animateTransition() {
  animating = true;
  posLayer   = computeLayerPositions();
  posCluster = computeClusterPositions();
  const fromPos = repositioned ? posLayer : posCluster;
  const toPos   = repositioned ? posCluster : posLayer;
  const duration = 800;
  d3.selectAll('.node')
    .transition().duration(duration).ease(d3.easeCubicInOut)
    .attr('transform', d => {
      const p = toPos[d.id] || fromPos[d.id] || {x:0,y:0};
      d.x = p.x; d.y = p.y;
      return `translate(${p.x},${p.y})`;
    });
  setTimeout(() => { animating = false; render(); }, duration+50);
}

function getCurrentPositions() {
  if (repositioned && mode === 'supernode') return computeClusterPositions();
  return computeLayerPositions();
}

// ── Supernode legend ──────────────────────────────────────────────────────────
function buildSNLegend() {
  const container = document.getElementById('sn-legend-items');
  container.innerHTML = '';
  snNames.forEach(sn => {
    const members = SUPERNODE_MAP[sn];
    const color = snColor[sn];
    const div = document.createElement('div');
    div.className = 'sn-legend-item';
    div.innerHTML = `<span class="sn-dot" style="background:${color}"></span>
      <span>${sn}</span><span class="sn-count">${members.length}n</span>`;
    div.addEventListener('click', () => {
      highlightedSN = (highlightedSN === sn) ? null : sn;
      render();
    });
    container.appendChild(div);
  });
}
buildSNLegend();

// ── SN Flow stats panel ───────────────────────────────────────────────────────
function buildSNFlowStats() {
  if (!SN_FLOW_DATA) {
    document.getElementById('sn-flow-stats').innerHTML =
      '<p style="font-size:9px;color:var(--dim)">No sn_flow JSON provided.<br>Run structure_grouping.py to generate.</p>';
    return;
  }
  const d   = SN_FLOW_DATA;
  const p   = d.preservation;
  const badgeClass = (p >= 0.8 && p <= 1.2) ? 'flow-pass'
    : (p > 1.2 || (p >= 0.5 && p < 0.8)) ? 'flow-warn' : 'flow-fail';
  const badgeText  = (p >= 0.8 && p <= 1.2) ? 'PASS'
    : (p > 1.2) ? `WARN amplified ${(p*100).toFixed(0)}%`
    : (p >= 0.5) ? `WARN low ${(p*100).toFixed(0)}%` : 'FAIL';

  const dominantRows = (d.dominant_paths || []).slice(0,5).map(e =>
    `<div class="dominant-edge-row">
       <span style="color:${snColor[e.src]||'#aaa'}">${e.src}</span>
       <span>→</span>
       <span style="color:${snColor[e.tgt]||'#aaa'}">${e.tgt}</span>
       <span>${e.weight.toFixed(5)}</span>
     </div>`
  ).join('');

  const bottlenecks = (d.bottleneck_sns || []);

  document.getElementById('sn-flow-stats').innerHTML = `
    <div class="flow-badge ${badgeClass}">PRESERVATION: ${(p*100).toFixed(1)}% — ${badgeText}</div>
    <div class="flow-stat-row">
      <span class="flow-stat-label">Orig reach→logit</span>
      <span class="flow-stat-value">${d.orig_reach !== undefined ? d.orig_reach.toFixed(4) : 'n/a'}</span>
    </div>
    <div class="flow-stat-row">
      <span class="flow-stat-label">Surr reach→logit</span>
      <span class="flow-stat-value">${d.surr_reach !== undefined ? d.surr_reach.toFixed(4) : 'n/a'}</span>
    </div>
    <div class="flow-stat-row">
      <span class="flow-stat-label">SN count</span>
      <span class="flow-stat-value">${d.sn_names.length}</span>
    </div>
    ${bottlenecks.length ? `<div class="flow-stat-row">
      <span class="flow-stat-label">Bottlenecks</span>
      <span class="flow-stat-value">${bottlenecks.map(b=>`<span class="bottleneck-tag">${b}</span>`).join('')}</span>
    </div>` : ''}
    <div style="margin-top:8px;margin-bottom:4px;font-size:8px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase">Top SN→SN Edges</div>
    ${dominantRows}`;
}
buildSNFlowStats();

// ── Color helpers ─────────────────────────────────────────────────────────────
const colorNodeDefault = d => {
  if (d.type==='embedding') return '#f59e0b';
  if (d.type==='logit')     return '#f43f5e';
  return d3.interpolateRgb('#10b981','#3d8eff')(d.node_inf);
};
const colorNodeSN = d => {
  const sn = node2sn[d.id];
  return sn ? snColor[sn] : '#334155';
};

// ── Convex hull ───────────────────────────────────────────────────────────────
function computeHulls(pos) {
  const hulls = {};
  snNames.forEach(sn => {
    const pts = SUPERNODE_MAP[sn].filter(nid=>pos[nid] && isFinite(pos[nid].x) && isFinite(pos[nid].y)).map(nid=>[pos[nid].x,pos[nid].y]);
    if (!pts.length) return;
    if (pts.length===1) {
      const r=currentNodeSize*2.5; const [cx,cy]=pts[0];
      hulls[sn]=`M${cx-r},${cy} A${r},${r} 0 1,0 ${cx+r},${cy} A${r},${r} 0 1,0 ${cx-r},${cy}`;
      return;
    }
    if (pts.length===2) pts.push([pts[0][0]+0.1,pts[0][1]+currentNodeSize*2.2]);
    try {
      const hull = d3.polygonHull(pts);
      if (!hull) return;
      const centroid = d3.polygonCentroid(hull);
      const pad = repositioned ? currentNodeSize*3.5 : currentNodeSize*2.8;
      const padded = hull.map(([x,y])=>{
        const dx=x-centroid[0],dy=y-centroid[1],dist=Math.sqrt(dx*dx+dy*dy)||1;
        return [x+dx/dist*pad,y+dy/dist*pad];
      });
      if (padded.some(p => !isFinite(p[0]) || !isFinite(p[1]))) return;
      hulls[sn]=`M${padded.map(p=>p.join(',')).join('L')}Z`;
    } catch(e){}
  });
  return hulls;
}

function clerpLabel(d) {
  if (d.type==='embedding') return (d.clerp||'').replace(/Emb:\s*"?\s*/,'').replace('"','').trim();
  const c = d.clerp || d.id;
  const maxLen = repositioned ? 18 : 14;
  return c.length > maxLen ? c.slice(0, maxLen-1)+'…' : c;
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: SN FLOW SURROGATE
// ─────────────────────────────────────────────────────────────────────────────
function renderSNFlow() {
  if (!SN_FLOW_DATA) {
    svg.selectAll('*').remove();
    svg.append('text').attr('x',width()/2).attr('y',height()/2)
      .attr('text-anchor','middle').attr('fill','#3a4a60')
      .attr('font-family','JetBrains Mono').attr('font-size',13)
      .text('No SN flow data — run structure_grouping.py with --out-json flag');
    return;
  }

  svg.selectAll('*').remove();
  const W = width(), H = height();
  svg.attr('viewBox',`0 0 ${W} ${H}`);

  const d         = SN_FLOW_DATA;
  const sn_names  = d.sn_names;
  const F_sn      = d.F_sn;       // K×K array
  const sn_reach  = d.sn_reach;   // K
  const sn_adj    = d.sn_adj;     // K×K raw weights
  const bottlenecks = new Set(d.bottleneck_sns || []);

  const snPos = computeSNFlowPositions();

  // Max values for scaling
  const maxReach   = Math.max(...sn_reach.map(Math.abs)) || 1;
  const allWeights = F_sn.flat().filter(v=>v>0);
  const maxWeight  = Math.max(...allWeights) || 1;
  const snEdgeThreshPct = currentSNEdgeThresh / 100;

  // ── Defs ──
  const defs = svg.append('defs');
  sn_names.forEach(sn => {
    const c = snColor[sn] || '#3d8eff';
    defs.append('marker')
      .attr('id',`snarrow-${sn.replace(/[^a-zA-Z0-9]/g,'_')}`).attr('viewBox','0 -4 8 8')
      .attr('refX',8).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4').attr('fill',c).attr('opacity',0.85);
  });

  // ── Background layer bands ──
  // Faint vertical bands for layer context
  const snMeanLayer = {};
  sn_names.forEach(sn => {
    const members = SUPERNODE_MAP[sn]||[];
    const layers  = members.map(nid=>nodeLayerMap[nid]||0);
    snMeanLayer[sn] = layers.length ? layers.reduce((a,b)=>a+b,0)/layers.length : 0;
  });
  const allLayerVals = sn_names.map(sn=>snMeanLayer[sn]);
  const lMin = Math.min(...allLayerVals), lMax = Math.max(...allLayerVals);
  const xScaleBg = d3.scaleLinear().domain([lMin,lMax]).range([100,W-100]);

  // ── Draw edges first (under nodes) ──
  const edgeG = svg.append('g').attr('id','sn-edges');

  sn_names.forEach((src, i) => {
    const srcPos = snPos[src];
    if (!srcPos) return;
    sn_names.forEach((tgt, j) => {
      if (i===j) return;
      const w = F_sn[i][j];
      if (!w || w <= maxWeight * snEdgeThreshPct) return;
      const tgtPos = snPos[tgt];
      if (!tgtPos) return;

      const normW  = w / maxWeight;
      const strokeW = Math.max(0.5, normW * 8);
      const opacity = Math.max(0.08, normW * currentOpacity * 2.2);
      const color   = snColor[src] || '#3d8eff';

      // Curved edge
      const dx=tgtPos.x-srcPos.x, dy=tgtPos.y-srcPos.y;
      const dist=Math.sqrt(dx*dx+dy*dy)||1;
      const curv = 40;
      const cx2=(srcPos.x+tgtPos.x)/2 - dy*curv/dist;
      const cy2=(srcPos.y+tgtPos.y)/2 + dx*curv/dist;

      // Offset endpoints to sit on circle edge
      const R_src = snNodeRadius(sn_reach[i], maxReach);
      const R_tgt = snNodeRadius(sn_reach[j], maxReach);
      const angle_s = Math.atan2(cy2-srcPos.y, cx2-srcPos.x);
      const angle_t = Math.atan2(cy2-tgtPos.y, cx2-tgtPos.x);
      const x1 = srcPos.x + Math.cos(angle_s)*R_src;
      const y1 = srcPos.y + Math.sin(angle_s)*R_src;
      const x2 = tgtPos.x + Math.cos(angle_t)*R_tgt;
      const y2 = tgtPos.y + Math.sin(angle_t)*R_tgt;

    edgeG.append('path')
      .attr('class','sn-flow-edge')
      .attr('data-src', src)          // ← add
      .attr('data-tgt', tgt)          // ← add
      .attr('data-base-opacity', opacity)   // ← add
      .attr('data-base-stroke', strokeW)    // ← add
      .attr('d',`M${x1},${y1} Q${cx2},${cy2} ${x2},${y2}`)
      .attr('stroke', color)
      .attr('stroke-width', strokeW)
      .attr('opacity', opacity)
      .attr('marker-end',`url(#snarrow-${src.replace(/[^a-zA-Z0-9]/g,'_')})`)
        .on('mouseenter', function(ev) {
          d3.select(this).attr('opacity', Math.min(opacity*3, 0.95));
          tooltip.textContent = `${src} → ${tgt}  |  F=${w.toFixed(5)}  adj=${(sn_adj[i]&&sn_adj[i][j]||0).toFixed(3)}`;
          tooltip.classList.add('show');
          const r=document.getElementById('canvas').getBoundingClientRect();
          tooltip.style.left=(ev.clientX-r.left+12)+'px';
          tooltip.style.top=(ev.clientY-r.top-28)+'px';
        })
        .on('mousemove', function(ev) {
          const r=document.getElementById('canvas').getBoundingClientRect();
          tooltip.style.left=(ev.clientX-r.left+12)+'px';
          tooltip.style.top=(ev.clientY-r.top-28)+'px';
        })
        .on('mouseleave', function() {
          d3.select(this).attr('opacity', opacity);
          tooltip.classList.remove('show');
        });
    });
  });

  // ── Draw SN nodes ──
  const nodeG = svg.append('g').attr('id','sn-nodes');

  sn_names.forEach((sn, i) => {
    const p = snPos[sn];
    if (!p) return;
    const color  = snColor[sn] || '#3d8eff';
    const r      = snNodeRadius(sn_reach[i], maxReach);
    const isBottle = bottlenecks.has(sn);
    const members  = SUPERNODE_MAP[sn] || [];

    const g = nodeG.append('g')
      .attr('class','sn-node')
      .attr('transform',`translate(${p.x},${p.y})`)
      .on('click', () => showSNInfo(sn, i))
         .on('mouseenter', function(ev) {
          // Highlight only edges connected to this SN (incoming + outgoing)
          edgeG.selectAll('.sn-flow-edge').each(function() {
            const el = d3.select(this);
            const edgeSrc = el.attr('data-src');
            const edgeTgt = el.attr('data-tgt');
            const isConnected = edgeSrc === sn || edgeTgt === sn;
            el.attr('opacity', isConnected ? Math.min(0.95, currentOpacity * 4) : 0.02);
            el.attr('stroke-width', isConnected ? parseFloat(el.attr('data-base-stroke')) * 2.2 : parseFloat(el.attr('data-base-stroke')) * 0.5);
          });
          showTooltipText(ev, `${sn}  |  reach=${sn_reach[i].toFixed(4)}  n=${members.length}`);
        })
        .on('mousemove', ev => moveTooltip(ev))
        .on('mouseleave', function() {
          // Restore all edges to their original opacity/width
          edgeG.selectAll('.sn-flow-edge').each(function() {
            const el = d3.select(this);
            el.attr('opacity', el.attr('data-base-opacity'));
            el.attr('stroke-width', el.attr('data-base-stroke'));
          });
          hideTooltip();
        })
      .on('mousemove', ev => moveTooltip(ev))
      .on('mouseleave', function() {
        edgeG.selectAll('.sn-flow-edge').attr('opacity', function() {
          // restore — approximate, exact opacity computed above
          return currentOpacity * 0.5;
        });
        hideTooltip();
        // full re-render to restore exact opacities
        renderSNFlow();
      });

    // Outer glow ring for bottlenecks
    if (isBottle) {
      g.append('circle')
        .attr('r', r+5).attr('fill','none')
        .attr('stroke','#f43f5e').attr('stroke-width',1.5)
        .attr('stroke-dasharray','3,3').attr('opacity',0.6);
    }

    // Main circle — size ∝ reach-to-logit, min radius 14
    g.append('circle')
      .attr('r', r)
      .attr('fill', color)
      .attr('fill-opacity', sn==='SN_LOGIT' ? 0.35 : 0.18)
      .attr('stroke', color)
      .attr('stroke-width', sn==='SN_LOGIT' ? 2.5 : 1.5)
      .attr('stroke-opacity', 0.8);

    // Inner dot proportional to activation
    const actNorm = SN_FLOW_DATA.sn_act_norm ? SN_FLOW_DATA.sn_act_norm[i] : 0.5;
    g.append('circle')
      .attr('r', Math.max(3, r*0.35*actNorm))
      .attr('fill', color)
      .attr('fill-opacity', 0.65);

    // SN name label
    const shortName = sn.replace('SN_','').replace(/_L\d+_?\d*/,'');
    g.append('text')
      .attr('text-anchor','middle').attr('dominant-baseline','middle')
      .attr('fill', color).attr('font-size', Math.max(8, Math.min(12, r*0.55)))
      .attr('font-family','Syne,sans-serif').attr('font-weight','600')
      .text(shortName);

    // Reach % label below
    const _reachTotal = sn_reach.filter(v=>v>0).reduce((a,b)=>a+b,0) || 1;
    const reachPct = (Math.max(0, sn_reach[i]) / _reachTotal * 100).toFixed(1);
    g.append('text')
      .attr('y', r+13).attr('text-anchor','middle')
      .attr('fill',color).attr('font-size',8).attr('opacity',0.65)
      .attr('font-family','JetBrains Mono,monospace')
      .text(`${reachPct}%`);

    // Member count
    g.append('text')
      .attr('y', r+23).attr('text-anchor','middle')
      .attr('fill','#3a4a60').attr('font-size',7)
      .attr('font-family','JetBrains Mono,monospace')
      .text(`n=${members.length}`);

    // Bottleneck tag
    if (isBottle) {
      g.append('text')
        .attr('y', -(r+8)).attr('text-anchor','middle')
        .attr('fill','#f43f5e').attr('font-size',7).attr('opacity',0.8)
        .text('⚠ bottleneck');
    }
  });

  // ── Layer axis at bottom ──
  const axisG = svg.append('g').attr('id','sn-axis');
  sn_names.forEach(sn => {
    const p = snPos[sn];
    if (!p) return;
    const lv = snMeanLayer[sn];
    axisG.append('line')
      .attr('x1',p.x).attr('y1',H-12).attr('x2',p.x).attr('y2',H-20)
      .attr('stroke','#1a2540').attr('stroke-width',1);
    axisG.append('text')
      .attr('x',p.x).attr('y',H-4).attr('text-anchor','middle')
      .attr('fill','#253550').attr('font-size',7).attr('font-family','JetBrains Mono')
      .text(`L${lv.toFixed(1)}`);
  });
}

function snNodeRadius(reach, maxReach) {
  const base = currentNodeSize * 1.2;
  const scale = Math.max(0.15, Math.abs(reach) / (maxReach || 1));
  return Math.max(14, base + scale * 38);
}

// ── SN info panel ─────────────────────────────────────────────────────────────
function showSNInfo(sn, i) {
  if (!SN_FLOW_DATA) return;
  const panel   = document.getElementById('node-info');
  const d       = SN_FLOW_DATA;
  const color   = snColor[sn];
  const members = SUPERNODE_MAP[sn] || [];
  const reach   = d.sn_reach[i];
  const actNorm = d.sn_act_norm ? d.sn_act_norm[i] : 0;
  const total   = d.sn_reach.reduce((a,b)=>a+b,0)||1;

  // Outgoing edges
  const outEdges = d.sn_names.map((tgt,j)=>({tgt, w:d.F_sn[i][j]}))
    .filter(e=>e.w>0 && e.tgt!==sn).sort((a,b)=>b.w-a.w).slice(0,5);

  const outRows = outEdges.map(e=>
    `<div class="info-row">
       <span class="info-label">→${e.tgt.replace('SN_','')}</span>
       <span class="info-value">${e.w.toFixed(5)}</span>
     </div>`).join('');

  const memberList = members.slice(0,6).map(nid=>{
    const nd = GRAPH_DATA.nodes.find(n=>n.id===nid);
    return `<div style="font-size:8px;color:var(--dim);padding:1px 0">${nd?nd.clerp:nid}</div>`;
  }).join('') + (members.length>6?`<div style="font-size:7px;color:#253550">+${members.length-6} more</div>`:'');

  panel.innerHTML = `
    <h3>SN Inspector</h3>
    <div class="info-sn-badge" style="background:${color}22;color:${color};border:1px solid ${color}55">${sn}</div>
    <div class="bar-container">
      <div class="bar-label">Reach→logit: ${reach.toFixed(4)} (${(reach/total*100).toFixed(1)}%)</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,reach/total*100).toFixed(1)}%;background:${color}"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">act_norm: ${actNorm.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(actNorm*100).toFixed(1)}%;background:#f59e0b"></div></div>
    </div>
    <div class="info-row"><span class="info-label">Members</span><span class="info-value">${members.length} nodes</span></div>
    ${outRows ? '<div style="margin-top:6px;font-size:8px;color:var(--dim);letter-spacing:.08em">TOP OUTFLOWS</div>'+outRows : ''}
    <div style="margin-top:8px;font-size:8px;color:var(--dim);letter-spacing:.08em;margin-bottom:4px">MEMBERS</div>
    ${memberList}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: RAW / SUPERNODE
// ─────────────────────────────────────────────────────────────────────────────
function renderNodeGraph() {
  if (animating) return;
  svg.selectAll('*').remove();
  const W = width(), H = height();
  svg.attr('viewBox',`0 0 ${W} ${H}`);

  const pos = getCurrentPositions();
  GRAPH_DATA.nodes.forEach(d => { if (pos[d.id] && isFinite(pos[d.id].x)) { d.x=pos[d.id].x; d.y=pos[d.id].y; } else if (!isFinite(d.x)) { d.x=100; d.y=100; } });

  const defs = svg.append('defs');
  [{id:'fwd',color:'#3d8eff'},{id:'bwd',color:'#f87171'},{id:'neg',color:'#8b5cf6'},{id:'logit',color:'#f5a623'}]
    .forEach(({id,color}) => {
      defs.append('marker').attr('id',`arrow-${id}`).attr('viewBox','0 -4 8 8')
        .attr('refX',8).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
        .append('path').attr('d','M0,-4L8,0L0,4').attr('fill',color).attr('opacity',0.8);
    });
  snNames.forEach(sn => {
    const c = snColor[sn];
    defs.append('marker').attr('id',`arrow-sn-${sn}`).attr('viewBox','0 -4 8 8')
      .attr('refX',8).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4').attr('fill',c).attr('opacity',0.8);
  });

  if (!repositioned) {
    const layerX = d3.scalePoint().domain(allLayers).range([80,W-80]).padding(0.3);
    allLayers.forEach(l => {
      const x=layerX(l);
      svg.append('line').attr('class','layer-line').attr('x1',x).attr('y1',22).attr('x2',x).attr('y2',H-8)
        .attr('stroke','#1a2540').attr('stroke-width',1);
      svg.append('text').attr('x',x).attr('y',14).attr('text-anchor','middle').attr('fill','#253550')
        .attr('font-size',9).attr('font-family','JetBrains Mono')
        .text(l===0?'EMB':l===27?'LOGIT':`L${l}`);
    });
  }

  const isSN = mode==='supernode';
  if (isSN) {
    const hulls=computeHulls(pos);
    const hullG=svg.append('g').attr('id','hulls');
    const labelG=svg.append('g').attr('id','hull-labels');
    snNames.forEach(sn => {
      if (!hulls[sn]) return;
      const color=snColor[sn], dim=highlightedSN&&highlightedSN!==sn;
      hullG.append('path').attr('class','sn-hull').attr('d',hulls[sn]).attr('fill',color).attr('stroke',color)
        .attr('fill-opacity',dim?0.01:(repositioned?0.09:0.07))
        .attr('stroke-opacity',dim?0.08:(repositioned?0.55:0.38));
      const members=SUPERNODE_MAP[sn].filter(nid=>pos[nid] && isFinite(pos[nid].x));
      if (!members.length) return;
      const cx=members.reduce((s,n)=>s+pos[n].x,0)/members.length;
      const minY=Math.min(...members.map(n=>pos[n].y));
      labelG.append('text').attr('class','sn-label visible')
        .attr('x',cx).attr('y',minY-(repositioned?currentNodeSize*4.5:currentNodeSize*3.5))
        .attr('text-anchor','middle').attr('fill',color)
        .attr('opacity',dim?0.1:(repositioned?0.7:0.5))
        .attr('font-size',repositioned?'11px':'8px').text(sn);
    });
  }

  const edgeG=svg.append('g').attr('id','edges-regular');
  const logitEdgeG=svg.append('g').attr('id','edges-logit');

  GRAPH_DATA.edges.forEach(e => {
    const src=pos[e.source], tgt=pos[e.target];
    if (!src||!tgt||!isFinite(src.x)||!isFinite(tgt.x)) return;
    const isToLogit=e.to_logit===true, isNeg=e.flow<0;
    const srcL=nodeLayerMap[e.source]||0, tgtL=nodeLayerMap[e.target]||0;
    const isBwd=srcL>tgtL&&!isToLogit;
    if (!isToLogit&&e.abs_flow<currentFlowThresh) return;
    const strokeW=Math.max(0.3,(e.abs_flow/maxFlow)*5);
    const opacity=isToLogit?Math.max(0.12,(e.abs_flow/maxFlow)*currentOpacity*3):currentOpacity*(isNeg?0.4:1.0);
    let color,arrowId;
    if (isSN&&!isToLogit){const sn=node2sn[e.source];color=sn?snColor[sn]:'#3d8eff';arrowId=sn?`sn-${sn}`:'fwd';}
    else if(isSN&&isToLogit){const sn=node2sn[e.source];color=sn?snColor[sn]:'#f5a623';arrowId=sn?`sn-${sn}`:'logit';}
    else{color=isToLogit?'#f5a623':isNeg?'#8b5cf6':isBwd?'#f87171':'#3d8eff';arrowId=isToLogit?'logit':isNeg?'neg':isBwd?'bwd':'fwd';}
    let finalOpacity=opacity;
    if(highlightedSN){const ss=node2sn[e.source],ts=node2sn[e.target];
      finalOpacity=(ss===highlightedSN||ts===highlightedSN)?Math.min(opacity*2,0.9):opacity*0.08;}
    const dx=tgt.x-src.x,dy=tgt.y-src.y,dist=Math.sqrt(dx*dx+dy*dy)+1;
    const curv=isToLogit?0:isBwd?-35:25;
    const cx2=(src.x+tgt.x)/2-dy*curv/dist, cy2=(src.y+tgt.y)/2+dx*curv/dist;
    (isToLogit?logitEdgeG:edgeG).append('path').attr('class','link').datum(e)
      .attr('d',`M${src.x},${src.y} Q${cx2},${cy2} ${tgt.x},${tgt.y}`)
      .attr('stroke',color).attr('stroke-width',strokeW).attr('opacity',finalOpacity)
      .attr('marker-end',`url(#arrow-${arrowId})`);
  });

  const nodeG=svg.append('g').attr('id','nodes');
  const nodeEls=nodeG.selectAll('.node').data(GRAPH_DATA.nodes).enter()
    .append('g').attr('class','node').attr('transform',d=>`translate(${d.x},${d.y})`)
    .style('cursor','pointer')
    .on('click',(ev,d)=>showNodeInfo(d))
    .on('mouseenter',(ev,d)=>{highlightNode(d,true);showTooltipNode(ev,d);})
    .on('mousemove',(ev)=>moveTooltip(ev))
    .on('mouseleave',(ev,d)=>{highlightNode(d,false);hideTooltip();});

  nodeEls.append('circle')
    .attr('r',d=>{ const b=currentNodeSize;
      if(d.type==='embedding') return b*1.15;
      if(d.type==='logit')     return b*1.65;
      return b*(0.5+0.9*d.node_inf); })
    .attr('fill',d=>isSN?colorNodeSN(d):colorNodeDefault(d))
    .attr('stroke',d=>{if(d.type==='logit')return'#f5a623';if(isSN){const sn=node2sn[d.id];return sn?snColor[sn]:'none';}return'none';})
    .attr('stroke-width',d=>d.type==='logit'?2:(isSN?0.8:0))
    .attr('opacity',d=>{
      if(highlightedSN){const sm=new Set(SUPERNODE_MAP[highlightedSN]||[]);return sm.has(d.id)?1.0:0.15;}
      return d.type==='transcoder'?0.82:1.0;
    });

  nodeEls.append('text')
    .attr('dy',d=>{const r=currentNodeSize*(d.type==='logit'?1.65:d.type==='embedding'?1.15:0.5+0.9*d.node_inf);return-(r+4);})
    .attr('text-anchor','middle')
    .attr('fill',d=>{if(isSN){const sn=node2sn[d.id];return sn?snColor[sn]:'#6a7f9a';}return'#6a7f9a';})
    .attr('font-size',repositioned?8:7)
    .attr('opacity',d=>{if(highlightedSN){const sm=new Set(SUPERNODE_MAP[highlightedSN]||[]);return sm.has(d.id)?0.9:0.1;}return 0.8;})
    .text(d=>clerpLabel(d));

  if(repositioned){
    nodeEls.append('text')
      .attr('dy',d=>{const r=currentNodeSize*(d.type==='logit'?1.65:d.type==='embedding'?1.15:0.5+0.9*d.node_inf);return r+12;})
      .attr('text-anchor','middle').attr('fill','#3a4a60').attr('font-size',7).attr('opacity',0.6)
      .text(d=>d.layer===0?'EMB':d.layer===27?'L27':`L${d.layer}`);
  }
}

// ── Master render dispatcher ──────────────────────────────────────────────────
function render() {
  if (mode === 'snflow') renderSNFlow();
  else renderNodeGraph();
}

// ── Tooltip helpers ───────────────────────────────────────────────────────────
function showTooltipNode(ev, d) {
  const sn=node2sn[d.id], snText=sn?` [${sn}]`:'';
  tooltip.textContent=`${d.clerp}${snText}  —  L${d.layer}`;
  tooltip.classList.add('show'); moveTooltip(ev);
}
function showTooltipText(ev, text) {
  tooltip.textContent=text; tooltip.classList.add('show'); moveTooltip(ev);
}
function moveTooltip(ev) {
  const r=document.getElementById('canvas').getBoundingClientRect();
  tooltip.style.left=(ev.clientX-r.left+12)+'px';
  tooltip.style.top=(ev.clientY-r.top-28)+'px';
}
function hideTooltip() { tooltip.classList.remove('show'); }

function highlightNode(d, on) {
  d3.selectAll('.link').attr('opacity', function(e) {
    if (!e) return on?0.02:currentOpacity;
    const isLogit=e.to_logit===true;
    const baseLogit=Math.max(0.12,(e.abs_flow/maxFlow)*currentOpacity*3);
    if (!on) {
      if(highlightedSN){const ss=node2sn[e.source],ts=node2sn[e.target];
        return(ss===highlightedSN||ts===highlightedSN)?(isLogit?baseLogit*2:currentOpacity*1.5):currentOpacity*0.08;}
      return isLogit?baseLogit:currentOpacity;
    }
    const connected=e.source===d.id||e.target===d.id;
    return connected?0.92:(isLogit?0.06:0.02);
  });
}

function showNodeInfo(d) {
  const panel=document.getElementById('node-info');
  const maxAct=Math.max(...GRAPH_DATA.nodes.map(n=>n.activation));
  const outEdges=GRAPH_DATA.edges.filter(e=>e.source===d.id);
  const inEdges=GRAPH_DATA.edges.filter(e=>e.target===d.id);
  const fl=d.inf_to_logit||null;
  const sn=node2sn[d.id], snCol=sn?snColor[sn]:null;
  const snBadge=sn?`<div class="info-sn-badge" style="background:${snCol}22;color:${snCol};border:1px solid ${snCol}55">${sn}</div>`:'';
  panel.innerHTML=`
    <h3>Node inspector</h3>${snBadge}
    <div class="info-clerp">${d.clerp}</div>
    <div class="info-row"><span class="info-label">ID</span><span class="info-value" style="font-size:8px">${d.id}</span></div>
    <div class="info-row"><span class="info-label">Layer</span><span class="info-value">${d.layer===0?'EMB':d.layer===27?'LOGIT':d.layer}</span></div>
    <div class="info-row"><span class="info-label">ctx_idx</span><span class="info-value">${d.ctx_idx}</span></div>
    <div class="bar-container">
      <div class="bar-label">Activation: ${d.activation.toFixed(2)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.activation/maxAct*100).toFixed(1)}%;background:#f59e0b"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Influence: ${d.influence.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.influence*100).toFixed(1)}%;background:#f5a623"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Attribution→logit: ${fl!==null?fl.toFixed(4):'n/a'}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${fl!==null?Math.min(100,Math.abs(fl)/40*100).toFixed(1):0}%;background:#f43f5e"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_inf: ${d.node_inf.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_inf*100).toFixed(1)}%;background:#3d8eff"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_rel: ${d.node_rel.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_rel*100).toFixed(1)}%;background:#10b981"></div></div>
    </div>
    <div style="margin-top:8px;font-size:9px;color:var(--dim)">${outEdges.length} out &middot; ${inEdges.length} in</div>`;
}

// ── Sliders ───────────────────────────────────────────────────────────────────
document.getElementById('flow-thresh').addEventListener('input', function() {
  currentFlowThresh=+this.value;
  document.getElementById('flow-val').textContent=this.value; render();
});
document.getElementById('node-size').addEventListener('input', function() {
  currentNodeSize=+this.value;
  document.getElementById('size-val').textContent=this.value; render();
});
document.getElementById('edge-opacity').addEventListener('input', function() {
  currentOpacity=this.value/100;
  document.getElementById('opacity-val').textContent=this.value; render();
});
document.getElementById('sn-edge-thresh').addEventListener('input', function() {
  currentSNEdgeThresh=+this.value;
  document.getElementById('sn-edge-val').textContent=this.value; render();
});
window.addEventListener('resize', render);

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',      type=str, default='subgraph/austin_plt.pt')
    parser.add_argument('--supernode', type=str, default='supernode_map.json')
    parser.add_argument('--sn-flow',   type=str, default=None,
                        help='Path to supernode_map_sn_flow.json (from structure_grouping.py). '
                             'Enables the SN Flow Surrogate view.')
    parser.add_argument('--out',       type=str, default='circuit_sp_rep.html')
    args = parser.parse_args()

    print(f"Loading {args.file}...")
    raw  = load_snapshot(args.file)
    data = prepare_graph_data(raw)
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Edges: {len(data['edges'])}")

    print(f"Loading supernode map: {args.supernode}...")
    with open(args.supernode, 'r') as f:
        supernode_map = json.load(f)
    print(f"  Supernodes: {len(supernode_map)}")

    # Load SN flow data if provided
    sn_flow_data = None
    if args.sn_flow:
        print(f"Loading SN flow data: {args.sn_flow}...")
        with open(args.sn_flow, 'r') as f:
            raw_sn = json.load(f)
        # Rename keys to match what the JS expects
        sn_flow_data = {
            'sn_names'       : raw_sn.get('sn_names', []),
            'sn_adj'         : raw_sn.get('sn_adj', []),
            'F_sn'           : raw_sn.get('F_sn', []),
            'sn_reach'       : raw_sn.get('sn_reach', []),
            'sn_act_norm'    : raw_sn.get('sn_act_norm', None),
            'preservation'   : raw_sn.get('preservation', 0),
            'orig_reach'     : raw_sn.get('orig_reach_total', None),
            'surr_reach'     : raw_sn.get('surr_reach_total', None),
            'dominant_paths' : raw_sn.get('dominant_paths', []),
            'bottleneck_sns' : raw_sn.get('bottleneck_sns', []),
        }
        print(f"  SN flow loaded: preservation={sn_flow_data['preservation']:.4f}")
    else:
        print("  No --sn-flow provided. SN Flow Surrogate view will be disabled.")

    graph_json   = json.dumps(data,         indent=None, separators=(',', ':'))
    sn_json      = json.dumps(supernode_map,indent=None, separators=(',', ':'))
    snflow_json  = json.dumps(sn_flow_data, indent=None, separators=(',', ':'))
    disabled_attr = '' if sn_flow_data else 'disabled title="Provide --sn-flow JSON to enable"'

    html = HTML_TEMPLATE \
        .replace('__GRAPH_DATA__',     graph_json) \
        .replace('__SUPERNODE_MAP__',  sn_json) \
        .replace('__SN_FLOW_DATA__',   snflow_json) \
        .replace('__SNFLOW_DISABLED__', disabled_attr)

    with open(args.out, 'w') as f:
        f.write(html)

    print(f"Saved → {args.out}")
    print(f"Open:  open {args.out}")


if __name__ == '__main__':
    main()