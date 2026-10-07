# =============================================================================
# ROGII Wellbore Geology Prediction — 自包含提交 Notebook(单文件,粘贴即用)
# 管线:PSG 多尺度锚定图(本地 Fold1 全趾端 pooled RMSE = 11.19 ft)
#   特征:几何/GR/DeltaCorr(±100ft 后验)/构造先验(train 井 EGFDU 点云)
#   模型:1ft→4ft-token 膨胀 TCN + EdgeHead + 多尺度锚定图 float64 求解
# 鲁棒:测试井无 TVT 列/无顶深列;自动找输入路径;GPU/CPU 自适应
# 用法:Kaggle Notebook(建议开 GPU)整段粘贴运行 → 生成 submission.csv
# =============================================================================
import os, glob, math, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

t00 = time.time()
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 45 if DEVICE != "cpu" else 18
TVT_SCALE = 20.0
OFF_B, OFF_G = 100.0, 2.0
HP_WIN, CORR_SMOOTH = 25, 15
GRID = 32
SCALES = (1, 2, 4, 8, 16, 32)
EDGE_W = {1: 1.0, 2: 0.5, 4: 0.25, 8: 0.12, 16: 0.06, 32: 0.03}
OFFSETS = np.arange(-OFF_B, OFF_B + OFF_G, OFF_G); N_OFF = len(OFFSETS)

# ---------------- 路径自动发现 ----------------
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input"),
]
OUTPUT_PATH = Path("/kaggle/working/submission.csv")


def find_competition_files():
    for root in DATA_ROOT_CANDIDATES:
        if not root.is_dir():
            continue
        sample_paths = sorted(root.rglob("sample_submission.csv"))
        horizontal_paths = sorted(root.rglob("*__horizontal_well.csv"))
        if sample_paths and horizontal_paths:
            return root, horizontal_paths, sample_paths[0]
    raise FileNotFoundError(
        "Competition files were not found under /kaggle/input. "
        "Attach the competition dataset to this notebook."
    )


DATA_ROOT, H_FILES, SS_PATH = find_competition_files()
ss = pd.read_csv(SS_PATH)
id_column = "id" if "id" in ss.columns else ss.columns[0]
target_columns = [column for column in ss.columns if column != id_column]
if len(target_columns) != 1:
    raise RuntimeError(f"Expected one target column, found {target_columns}")
target_column = target_columns[0]

id_parts = ss[id_column].astype(str).str.rsplit("_", n=1, expand=True)
if id_parts.shape[1] != 2:
    raise RuntimeError("Submission IDs must end in '_<horizontal-row-index>'.")
ss["wid"] = id_parts[0]
ss["idx"] = pd.to_numeric(id_parts[1], errors="raise").astype(int)
requested_test_wells = set(ss["wid"].unique())

print(
    f"DATA_ROOT={DATA_ROOT} | horizontal_files={len(H_FILES)} | "
    f"submission_rows={len(ss)} | requested_wells={len(requested_test_wells)} | "
    f"device={DEVICE}"
)

# ---------------- 工具 ----------------
def moving_avg(x, w):
    if w is None or w <= 1: return np.asarray(x, float)
    w = int(w); k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")

def interp_nan(x):
    x = np.asarray(x, float); good = np.isfinite(x)
    if good.sum() == 0: return np.zeros_like(x)
    if good.all(): return x
    idx = np.arange(len(x)); return np.interp(idx, idx[good], x[good])

def robust_stats(v):
    v = v[np.isfinite(v)]
    if len(v) < 5: return 0.0, 1.0
    med = np.median(v); iqr = np.percentile(v, 75) - np.percentile(v, 25)
    return float(med), float(iqr if iqr > 1e-6 else (np.std(v) + 1e-6))

def box0(A, w):
    if w <= 1: return np.asarray(A, float)
    A = np.asarray(A, float); pl = w // 2; pr = w - 1 - pl
    P = np.pad(A, [(pl, pr)] + [(0, 0)] * (A.ndim - 1), mode="edge")
    c = np.cumsum(P, axis=0)
    z = np.zeros((1,) + A.shape[1:]) if A.ndim > 1 else np.zeros(1)
    c = np.concatenate([z, c], axis=0)
    return (c[w:] - c[:-w]) / w

# ---------------- 数据载入(train/test 统一) ----------------
def load_well(hpath):
    d = pd.read_csv(hpath)
    wid = os.path.basename(hpath).split("__")[0]
    tp = os.path.join(os.path.dirname(hpath), f"{wid}__typewell.csv")
    tw = pd.read_csv(tp) if os.path.exists(tp) else None
    n = len(d)
    get = lambda c: d[c].values.astype(float) if c in d.columns else np.full(n, np.nan)
    TVT_in = get("TVT_input"); TVT = get("TVT")
    has_true = np.isfinite(TVT).sum() > np.isfinite(TVT_in).sum()   # train 井
    if not has_true: TVT = TVT_in.copy()                            # 测试井:前缀即全部已知
    mask = ~np.isfinite(TVT_in)
    ps = int(np.argmax(mask)) if mask.any() else n
    tt_s = tg_s = None
    if tw is not None and {"TVT", "GR"}.issubset(tw.columns):
        tt = tw.TVT.values.astype(float); tg = tw.GR.values.astype(float)
        ok = np.isfinite(tt) & np.isfinite(tg)
        if ok.sum() >= 10:
            o = np.argsort(tt[ok]); tt_s = tt[ok][o]; tg_s = moving_avg(tg[ok][o], 5)
    return dict(wid=wid, MD=get("MD"), X=get("X"), Y=get("Y"), Z=get("Z"), GR=get("GR"),
                TVT=TVT, TOP=get("EGFDU"), real_ps=ps, n=n, tt_s=tt_s, tg_s=tg_s,
                is_train=bool(has_true))

print("loading wells ...")
CACHE = {}
for i, h in enumerate(sorted(H_FILES)):
    w = load_well(h); CACHE[w["wid"]] = w
    if (i + 1) % 200 == 0: print(f"  {i+1}/{len(H_FILES)}")
train_ids = [k for k, w in CACHE.items() if w["is_train"]]
test_ids = [k for k, w in CACHE.items() if not w["is_train"]]
if SS_PATH:
    ss = pd.read_csv(SS_PATH)
    ss["wid"] = ss["id"].str.split("_").str[0]; ss["idx"] = ss["id"].str.split("_").str[1].astype(int)
    need = [w_ for w_ in ss.wid.unique() if w_ in CACHE]
    test_ids = sorted(set(need))                     # 以 sample_submission 为准
    train_ids = [k for k in CACHE if k not in set(test_ids) and CACHE[k]["is_train"]]
print(f"train wells={len(train_ids)}  test wells={len(test_ids)}  ({time.time()-t00:.0f}s)")

# ---------------- 构造先验(train 井 EGFDU 点云) ----------------
from scipy.spatial import cKDTree
SP = dict(h=500.0, ridge=0.05, k=96, ds=8, dec=4, far=1500.0, cwin=1500)
def build_cloud(wids):
    PX, PY, PT, PW = [], [], [], []
    for i, wid in enumerate(wids):
        w = CACHE[wid]
        if not np.isfinite(w["TOP"]).any(): continue
        dz = np.abs(np.gradient(w["Z"], w["MD"]))
        idx = np.arange(0, w["n"], SP["ds"]); idx = idx[dz[idx] < 0.3]
        v = np.isfinite(w["TOP"][idx])
        if v.sum() == 0: continue
        PX.append(w["X"][idx][v]); PY.append(w["Y"][idx][v])
        PT.append(w["TOP"][idx][v]); PW.append(np.full(int(v.sum()), i))
    PX = np.concatenate(PX); PY = np.concatenate(PY); PT = np.concatenate(PT); PW = np.concatenate(PW)
    return dict(tree=cKDTree(np.c_[PX, PY]), PX=PX, PY=PY, PT=PT, PW=PW,
                wid2idx={wid: i for i, wid in enumerate(wids)})

def struct_curve(cloud, w, exclude_idx=None):
    n = w["n"]; X = w["X"]; Y = w["Y"]; MD = w["MD"]
    dec = SP["dec"]; k = SP["k"]; h = SP["h"]; ridge = SP["ridge"]
    qi = np.arange(0, n, dec)
    if qi[-1] != n - 1: qi = np.append(qi, n - 1)
    dist, ii = cloud["tree"].query(np.c_[X[qi], Y[qi]], k=k)
    Wt = np.exp(-(dist / h) ** 2); pw = cloud["PW"][ii]
    if exclude_idx is not None:
        own = pw == exclude_idx
        Wt = np.where(own, 0.0, Wt); dist = np.where(own, np.inf, dist)
    pwm = np.where(Wt > 0.02 * (Wt.max(1, keepdims=True) + 1e-12), pw, -1)
    spw = np.sort(pwm, 1)
    nw = ((spw[:, 1:] != spw[:, :-1]) & (spw[:, 1:] >= 0)).sum(1) + (spw[:, 0] >= 0)
    x0 = cloud["PX"][ii] - X[qi][:, None]; y0 = cloud["PY"][ii] - Y[qi][:, None]; z = cloud["PT"][ii]
    A11 = Wt.sum(1); lam = ridge * (A11 + 1e-12) * h * h
    A12 = (Wt * x0).sum(1); A13 = (Wt * y0).sum(1)
    A22 = (Wt * x0 * x0).sum(1) + lam; A23 = (Wt * x0 * y0).sum(1); A33 = (Wt * y0 * y0).sum(1) + lam
    b1 = (Wt * z).sum(1); b2 = (Wt * z * x0).sum(1); b3 = (Wt * z * y0).sum(1)
    Am = np.stack([np.stack([A11, A12, A13], -1), np.stack([A12, A22, A23], -1),
                   np.stack([A13, A23, A33], -1)], 1) + np.eye(3)[None] * 1e-8
    sol = np.linalg.solve(Am, np.stack([b1, b2, b3], -1)[:, :, None])[:, :, 0]
    a, bg, cg = sol[:, 0], sol[:, 1], sol[:, 2]
    amean = b1 / (A11 + 1e-12); use_mean = nw < 2
    a = np.where(use_mean, amean, a); bg = np.where(use_mean, 0, bg); cg = np.where(use_mean, 0, cg)
    a = np.where(A11 < 1e-10, np.nan, a)
    d_other = dist.min(1)
    tx = np.gradient(X, MD); ty = np.gradient(Y, MD)
    nrm = np.sqrt(tx ** 2 + ty ** 2) + 1e-9
    dipa = bg * (tx[qi] / nrm[qi]) + cg * (ty[qi] / nrm[qi])
    idx = np.arange(n)
    TH = np.interp(idx, qi, a); TH[np.interp(idx, qi, np.isnan(a).astype(float)) > 0] = np.nan
    return (TH, np.interp(idx, qi, np.where(np.isfinite(d_other), d_other, 1e6)),
            np.interp(idx, qi, nw.astype(float)), np.interp(idx, qi, dipa))

print("struct prior ...")
CLOUD = build_cloud(train_ids)
for wid in train_ids:
    w = CACHE[wid]
    w["TH"], w["THd"], w["THn"], w["THdp"] = struct_curve(CLOUD, w, CLOUD["wid2idx"][wid])
for wid in test_ids:
    w = CACHE[wid]
    w["TH"], w["THd"], w["THn"], w["THdp"] = struct_curve(CLOUD, w, None)
print(f"struct done ({time.time()-t00:.0f}s)")

# ---------------- 特征(23 通道;与本地最佳配置一致,tort 组剔除) ----------------
FEATS = ["md_rel", "beyond", "dz_ps", "incl", "curv", "azim",
         "gr", "gr_mask", "gr_grad", "gr_rstd", "coverage", "tvt_in", "known_mask",
         "dc_mean", "dc_peak", "dc_ent",
         "struct_rel", "struct_d", "struct_n", "struct_v", "struct_dip", "struct_slope", "dzs"]
N_FEAT = len(FEATS)

def struct_c_and_prior(w, ps):
    TH = w["TH"]; n = w["n"]
    Z = w["Z"]; TVT = w["TVT"]; MD = w["MD"]
    validp = np.isfinite(TH) & (w["THd"] < SP["far"])
    lo = np.searchsorted(MD, MD[max(ps - 1, 0)] - SP["cwin"])
    sel = np.arange(lo, ps); sel = sel[validp[sel] & np.isfinite(TVT[sel])]
    if len(sel) < 50:
        sel = np.arange(0, ps); sel = sel[validp[sel] & np.isfinite(TVT[sel])]
    if len(sel) < 50: return None, np.zeros(n, bool)
    C = np.median(TVT[sel] + Z[sel] - TH[sel])
    return TH + C - Z, validp

def build_features(w, ps):
    n = w["n"]; MD = w["MD"]; Z = w["Z"]; X = w["X"]; Y = w["Y"]; GR = w["GR"]; TVT = w["TVT"]
    ps = int(min(max(ps, 5), n - 5))
    tvt_ps = TVT[ps - 1]
    idx = np.arange(n); beyond = (idx >= ps).astype(np.float32)
    md_rel = (MD - MD[ps]) / 1000.0
    dz_ps = (Z[ps] - Z) / 100.0
    dZ = np.gradient(Z, MD); incl = moving_avg(dZ, 25)
    curv = moving_avg(np.gradient(dZ, MD), 25) * 100.0
    azim = moving_avg(np.arctan2(np.gradient(Y, MD), np.gradient(X, MD)), 25) / math.pi
    m, s = robust_stats(GR[:ps])
    gr_valid = np.isfinite(GR); gr_norm = (interp_nan(GR) - m) / s
    gr_sm = moving_avg(gr_norm, 7)
    gr_grad = np.gradient(gr_sm, MD) * 10.0
    gr_rstd = moving_avg((gr_norm - gr_sm) ** 2, 15) ** 0.5
    coverage = moving_avg(gr_valid.astype(float), 25)
    tvt_in = np.where(idx < ps, np.nan_to_num(TVT - tvt_ps) / TVT_SCALE, 0.0)
    known_mask = (idx < ps).astype(np.float32)
    # DeltaCorr
    dc_mean = np.zeros(n); dc_peak = np.zeros(n); dc_ent = np.ones(n)
    grid = tvt_ps + OFFSETS; ref = np.full(N_OFF, np.nan)
    pt = TVT[:ps]; pg = gr_norm[:ps]; good = np.isfinite(pt) & np.isfinite(GR[:ps])
    if good.sum() >= 30:
        o = np.argsort(pt[good]); ref = np.interp(grid, pt[good][o], pg[good][o], left=np.nan, right=np.nan)
    if w["tt_s"] is not None and np.isnan(ref).any():
        tg_n = (w["tg_s"] - m) / s
        tw_ = np.interp(grid, w["tt_s"], tg_n, left=np.nan, right=np.nan)
        ref = np.where(np.isfinite(ref), ref, tw_)
    if np.isfinite(ref).sum() >= 3:
        ref = interp_nan(ref)
        gr_hp = gr_sm - moving_avg(gr_sm, HP_WIN)
        ref_hp = ref - moving_avg(ref, max(int(round(HP_WIN / OFF_G)), 2))
        sc = -(gr_hp[:, None] - ref_hp[None, :]) ** 2
        sc = box0(sc, CORR_SMOOTH) / (np.std(gr_hp) ** 2 + 1e-6)
        p = np.exp(sc - sc.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
        dc_mean = (p * OFFSETS[None, :]).sum(1) / TVT_SCALE
        dc_peak = p.max(1); dc_ent = -(p * np.log(p + 1e-9)).sum(1) / math.log(N_OFF)
    prior, validp = struct_c_and_prior(w, ps)
    if prior is not None:
        struct_rel = np.where(validp, np.clip((prior - tvt_ps) / TVT_SCALE, -8, 8), 0.0)
        struct_v = validp.astype(float)
        pr_f = interp_nan(np.where(validp, prior, np.nan))
        struct_slope = np.where(validp, np.clip(np.gradient(pr_f, MD), -0.2, 0.2) * 20.0, 0.0)
    else:
        struct_rel = np.zeros(n); struct_v = np.zeros(n); struct_slope = np.zeros(n)
    struct_d = np.clip(w["THd"] / 1000.0, 0, 3); struct_n = np.clip(w["THn"] / 5.0, 0, 2)
    struct_dip = np.clip(w["THdp"] * 20.0, -3, 3)
    dzs = np.clip(np.gradient(Z, MD), -1.2, 1.2)
    F = np.stack([md_rel, beyond, dz_ps, incl, curv, azim,
                  gr_sm, gr_valid.astype(float), gr_grad, gr_rstd, coverage, tvt_in, known_mask,
                  dc_mean, dc_peak, dc_ent,
                  struct_rel, struct_d, struct_n, struct_v, struct_dip, struct_slope, dzs],
                 1).astype(np.float32)
    return np.nan_to_num(F), float(tvt_ps)

# ---------------- 模型(PSG:4ft-token TCN + EdgeHead + 锚定图) ----------------
class TCNBlock(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        pad = (k - 1) * d // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.n1 = nn.GroupNorm(4, c); self.n2 = nn.GroupNorm(4, c)
        self.drop = nn.Dropout(p); self.act = nn.GELU()
    def forward(self, x):
        h = self.drop(self.act(self.n1(self.c1(x))))
        h = self.drop(self.act(self.n2(self.c2(h))))
        return x + h

class Trunk(nn.Module):
    def __init__(self, n_feat, hidden=48, dils=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        self.stem = nn.Conv1d(n_feat, hidden, 5, stride=4, padding=2)
        self.blocks = nn.ModuleList([TCNBlock(hidden, 3, d, 0.1) for d in dils])
    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks: h = b(h)
        return h

class EdgeHead(nn.Module):
    def __init__(self, hidden=48, width=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden * 3 + 2, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
    def forward(self, h_i, h_j, h_pool, dz_norm, lg_s):
        z = torch.cat([h_i, h_j - h_i, h_pool, dz_norm[:, None], lg_s[:, None]], 1)
        return self.mlp(z).squeeze(1)

class PSG(nn.Module):
    def __init__(self, n_feat, hidden=48):
        super().__init__()
        self.trunk = Trunk(n_feat, hidden); self.edge = EdgeHead(hidden)

MODEL = PSG(N_FEAT).to(DEVICE)
print(f"PSG params={sum(p.numel() for p in MODEL.parameters())/1e3:.0f}k")

def make_nodes(w, ps):
    n = w["n"]
    back = np.arange(ps - 1, -1, -GRID)[::-1]
    fwd = np.arange(ps - 1 + GRID, n, GRID)
    nodes = np.concatenate([back, fwd])
    if nodes[-1] != n - 1: nodes = np.append(nodes, n - 1)
    return nodes.astype(int), int(np.searchsorted(nodes, ps - 1))

def make_edges(nnode):
    ei, ej, es = [], [], []
    for s_ in SCALES:
        stride = max(1, s_ // 2)
        i = np.arange(0, nnode - s_, stride)
        ei.append(i); ej.append(i + s_); es.append(np.full(len(i), s_))
    return np.concatenate(ei), np.concatenate(ej), np.concatenate(es)

def solve_anchored(nnode, fixed_mask, fixed_vals, ei, ej, dse, we):
    free = ~fixed_mask; nf = int(free.sum())
    fidx = torch.cumsum(free.long(), 0) - 1
    A = torch.zeros(nf, nf, dtype=torch.float64); b = torch.zeros(nf, dtype=torch.float64)
    Sfix = torch.zeros(nnode, dtype=torch.float64); Sfix[fixed_mask] = fixed_vals
    fi = fidx[ei]; fj = fidx[ej]; freei = free[ei]; freej = free[ej]
    for cond, idxs in ((freej, fj[freej]), (freei, fi[freei])):
        A.index_put_((idxs, idxs), we[cond], accumulate=True)
    both = freei & freej
    A.index_put_((fi[both], fj[both]), -we[both], accumulate=True)
    A.index_put_((fj[both], fi[both]), -we[both], accumulate=True)
    d_eff = dse + Sfix[ei] * (~freei) - Sfix[ej] * (~freej)
    b.index_put_((fj[freej],), (we * d_eff)[freej], accumulate=True)
    b.index_put_((fi[freei],), (-(we * d_eff))[freei], accumulate=True)
    A = A + torch.eye(nf, dtype=torch.float64) * 1e-8
    L = torch.linalg.cholesky(A)
    x = torch.cholesky_solve(b[:, None], L)[:, 0]
    S = torch.zeros(nnode, dtype=torch.float64); S[fixed_mask] = fixed_vals; S[free] = x
    return S

def well_forward(w, ps, F_np):
    n = w["n"]
    x = torch.tensor(F_np.T[None]).to(DEVICE)
    h = MODEL.trunk(x)[0]; Ltok = h.shape[1]
    nodes, anchor = make_nodes(w, ps)
    tok = np.clip(nodes // 4, 0, Ltok - 1)
    ei, ej, es = make_edges(len(nodes))
    hn = h[:, torch.tensor(tok, dtype=torch.long, device=DEVICE)].T
    hc = torch.cumsum(h, dim=1)
    r_i = nodes[ei]; r_j = nodes[ej]
    t_i = np.clip(r_i // 4, 0, Ltok - 1); t_j = np.clip(r_j // 4, 0, Ltok - 1)
    pool = (hc[:, torch.tensor(t_j, device=DEVICE)] - hc[:, torch.tensor(t_i, device=DEVICE)]).T \
        / torch.tensor(np.maximum(t_j - t_i, 1)[:, None], dtype=torch.float32, device=DEVICE)
    Z = w["Z"]
    Lft = torch.tensor((r_j - r_i).astype(np.float32), device=DEVICE)
    dz_n = torch.tensor(((Z[r_j] - Z[r_i]) / (r_j - r_i)).astype(np.float32), device=DEVICE)
    lg = torch.tensor(np.log2(es).astype(np.float32), device=DEVICE)
    slope = MODEL.edge(hn[torch.tensor(ei, device=DEVICE)], hn[torch.tensor(ej, device=DEVICE)],
                       pool, dz_n, lg)
    dse = (dz_n + 0.15 * torch.tanh(slope)) * Lft
    S_true_pre = np.nan_to_num(w["TVT"]) + Z
    fixed = torch.tensor(nodes <= (ps - 1))
    fixed_vals = torch.tensor(S_true_pre[nodes[nodes <= (ps - 1)]], dtype=torch.float64)
    we = torch.tensor(np.array([EDGE_W[s_] for s_ in es]) / np.sqrt(es), dtype=torch.float64)
    S_nodes = solve_anchored(len(nodes), fixed, fixed_vals,
                             torch.tensor(ei), torch.tensor(ej), dse.cpu().double(), we)
    rows = np.arange(ps, n)
    seg = np.clip(np.searchsorted(nodes, rows, side="right") - 1, 0, len(nodes) - 2)
    tfr = torch.tensor((rows - nodes[seg]) / (nodes[seg + 1] - nodes[seg]), dtype=torch.float64)
    S_rows = S_nodes[seg] * (1 - tfr) + S_nodes[seg + 1] * tfr
    tvt_pred = S_rows - torch.tensor(Z[rows], dtype=torch.float64)
    y_edge = torch.tensor((S_true_pre[r_j] - S_true_pre[r_i]).astype(np.float64))
    return tvt_pred, rows, dse.cpu().double(), y_edge, es

# ---------------- 训练(全部 train 井,10% 留出早停) ----------------
rng = np.random.RandomState(SEED)
tr = list(train_ids); rng.shuffle(tr)
n_hold = max(20, len(tr) // 10)
hold, tr = tr[:n_hold], tr[n_hold:]
print(f"train={len(tr)} holdout={len(hold)}")
opt = torch.optim.AdamW(MODEL.parameters(), lr=1.2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

def pseudo_ps(w, rng):
    return int(rng.uniform(0.20, 0.45) * w["n"])

def eval_hold():
    MODEL.eval(); sq = 0.0; npt = 0
    with torch.no_grad():
        for wid in hold:
            w = CACHE[wid]; ps = w["real_ps"]
            if ps >= w["n"] or ps < 300: continue
            F, _ = build_features(w, ps)
            pred, rows, *_ = well_forward(w, ps, F)
            d = pred.numpy() - w["TVT"][rows]
            d = d[np.isfinite(d)]
            sq += float(np.sum(d * d)); npt += len(d)
    return math.sqrt(sq / max(npt, 1))

best = (1e9, None); bad = 0; ACC = 6
for ep in range(EPOCHS):
    MODEL.train(); order = list(tr); rng.shuffle(order)
    opt.zero_grad(); nb = 0
    for bi, wid in enumerate(order):
        w = CACHE[wid]
        ps = pseudo_ps(w, rng)
        if ps >= w["n"] - 100 or ps < 300: continue
        F, _ = build_features(w, ps)
        pred, rows, dse, y_edge, es = well_forward(w, ps, F)
        true = torch.tensor(w["TVT"][rows])
        mfin = torch.isfinite(true)
        loss_path = ((pred - true)[mfin] ** 2).mean()
        le = 0.0
        es_t = torch.tensor(es)
        for s_ in SCALES:
            m_ = es_t == s_
            if m_.any():
                r_ = (dse[m_] - y_edge[m_]) / (s_ * GRID)
                r_ = r_[torch.isfinite(r_)]
                if len(r_): le = le + (r_ ** 2).mean()
        loss = 0.7 * loss_path / 400.0 + 0.3 * le * 100.0
        (loss / ACC).backward(); nb += 1
        if (bi + 1) % ACC == 0:
            nn.utils.clip_grad_norm_(MODEL.parameters(), 2.0)
            opt.step(); opt.zero_grad()
    sched.step()
    hv = eval_hold()
    print(f"ep{ep+1:02d} holdout pooled={hv:.3f} ft  ({time.time()-t00:.0f}s)")
    if hv < best[0]:
        best = (hv, {k: v.detach().cpu().clone() for k, v in MODEL.state_dict().items()}); bad = 0
    else:
        bad += 1
        if bad >= 10: print("early stop"); break
MODEL.load_state_dict(best[1])
print(f"best holdout={best[0]:.3f} ft")

# ---------------- 推理测试井 → submission.csv ----------------
MODEL.eval()
pred_map = {}
with torch.no_grad():
    for wid in test_ids:
        w = CACHE[wid]; ps = w["real_ps"]
        F, tvt_ps = build_features(w, ps)
        pred, rows, *_ = well_forward(w, ps, F)
        p = pred.numpy()
        p = np.where(np.isfinite(p), p, tvt_ps)
        pred_map[wid] = dict(zip(rows.tolist(), p.tolist()))
        print(f"  test {wid}: rows={len(rows)}")
if SS_PATH:
    out = [pred_map.get(w_, {}).get(i_, np.nan) for w_, i_ in zip(ss.wid, ss.idx)]
    # 缺失兜底:该井 PS 锚点
    for j, v in enumerate(out):
        if not np.isfinite(v):
            w = CACHE.get(ss.wid.iloc[j])
            out[j] = float(w["TVT"][w["real_ps"] - 1]) if w is not None else 0.0
    pd.DataFrame({"id": ss["id"], "tvt": out}).to_csv("submission.csv", index=False)
    print(f"submission.csv written: {len(out)} rows  ({time.time()-t00:.0f}s)")
else:
    print("no sample_submission found — 请确认竞赛输入挂载")
