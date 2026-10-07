# =============================================================================
# PSG 推理专用 cell —— 加载预训练 5 折 checkpoint,对隐藏测试井推理并融合
# =============================================================================
# 前置:把本目录作为 Kaggle dataset 挂载(含 psg_oof_fold1..5.pt + psg_oof.parquet)
# 运行时:仅推理,无训练。773 训练井建构造先验 + N 测试井前向,约 2-5 分钟(CPU 亦可)。
# 位置:放在你们最终 submission.csv 写出之后。
#
# 依据(本地 5 折 OOF,与你们同一 3,783,989 行口径):
#   你们 10.370 | PSG 10.940 | bias 相关 0.632
#   融合最优 w≈0.43-0.45 -> 9.20-9.61(bias 分量精确:8.249 -> 7.596)
# =============================================================================
import os, sys, glob, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

t0 = time.time()
DEVICE = "cpu"
# Replace this with the exact value printed by cell 1.
BLEND_W = 0.440000
FUSE_CLIP = 60.0
if not (0.0 <= float(BLEND_W) <= 0.80):
    raise RuntimeError(f"BLEND_W must be in [0, 0.80], got {BLEND_W}")

# ----------------------------- 路径发现 -----------------------------
def _find(pattern, roots=("/kaggle/input", ".")):
    for r in roots:
        hits = glob.glob(os.path.join(r, "**", pattern), recursive=True)
        if hits: return sorted(hits)
    return []

CKPTS = _find("psg_oof_fold*.pt")
assert CKPTS, "未找到 psg_oof_fold*.pt,请确认 dataset 已挂载"
import re
fold_numbers = []
for checkpoint_path in CKPTS:
    match = re.search(r"psg_oof_fold(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match is None:
        raise RuntimeError(f"Cannot parse fold number from {checkpoint_path}")
    fold_numbers.append(int(match.group(1)))
if sorted(fold_numbers) != [1, 2, 3, 4, 5]:
    raise RuntimeError(
        f"Expected exactly fold checkpoints 1..5, got {list(zip(fold_numbers, CKPTS))}"
    )
DATA = None
for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
          "/kaggle/input/rogii-wellbore-geology-prediction"]:
    if os.path.isdir(os.path.join(c, "train")): DATA = c; break
assert DATA, "未找到竞赛数据目录"
print(f"ckpts={len(CKPTS)}  data={DATA}  device={DEVICE}")

# ----------------------------- 常量/工具(与训练端一致) -----------------------------
TVT_SCALE = 20.0; OFF_B, OFF_G = 100.0, 2.0; HP_WIN, CORR_SMOOTH = 25, 15
GRID = 32; SCALES = (1, 2, 4, 8, 16, 32)
EDGE_W = {1: 1.0, 2: .5, 4: .25, 8: .12, 16: .06, 32: .03}
OFFSETS = np.arange(-OFF_B, OFF_B + OFF_G, OFF_G); N_OFF = len(OFFSETS)
SP = dict(h=500.0, ridge=0.05, k=96, ds=8, dec=4, far=1500.0, cwin=1500)
# 通道顺序必须与训练端逐字一致(26 通道;tort_* 训练时被置零但保留通道位)
FEATS = ["md_rel","beyond","dz_ps","incl","curv","azim","gr","gr_mask","gr_grad","gr_rstd",
         "coverage","tvt_in","known_mask","dc_mean","dc_peak","dc_ent",
         "tort_s","tort_l","tort_cum",                    # 恒为 0(训练时 ablate)
         "struct_rel","struct_d","struct_n","struct_v","struct_dip","struct_slope","dzs"]
N_FEAT = len(FEATS)   # = 26

def moving_avg(x, w):
    if w is None or w <= 1: return np.asarray(x, float)
    w = int(w); k = np.ones(w)/w
    return np.convolve(np.pad(x,(w//2,w-1-w//2),mode="edge"), k, mode="valid")
def interp_nan(x):
    x = np.asarray(x,float); g = np.isfinite(x)
    if g.sum()==0: return np.zeros_like(x)
    if g.all(): return x
    i = np.arange(len(x)); return np.interp(i, i[g], x[g])
def robust_stats(v):
    v = v[np.isfinite(v)]
    if len(v)<5: return 0.,1.
    m = np.median(v); q = np.percentile(v,75)-np.percentile(v,25)
    return float(m), float(q if q>1e-6 else np.std(v)+1e-6)
def box0(A,w):
    if w<=1: return np.asarray(A,float)
    A = np.asarray(A,float); pl=w//2; pr=w-1-pl
    P = np.pad(A,[(pl,pr)]+[(0,0)]*(A.ndim-1),mode="edge")
    c = np.cumsum(P,axis=0)
    z = np.zeros((1,)+A.shape[1:]) if A.ndim>1 else np.zeros(1)
    return (np.concatenate([z,c],0)[w:]-np.concatenate([z,c],0)[:-w])/w

def load_well(hp):
    d = pd.read_csv(hp); wid = os.path.basename(hp).split("__")[0]
    tp = os.path.join(os.path.dirname(hp), f"{wid}__typewell.csv")
    tw = pd.read_csv(tp) if os.path.exists(tp) else None
    n = len(d); get = lambda c: d[c].values.astype(float) if c in d.columns else np.full(n, np.nan)
    ti = get("TVT_input"); tv = get("TVT")
    has = np.isfinite(tv).sum() > np.isfinite(ti).sum()
    if not has: tv = ti.copy()
    msk = ~np.isfinite(ti); ps = int(np.argmax(msk)) if msk.any() else n
    tt_s = tg_s = None
    if tw is not None and {"TVT","GR"}.issubset(tw.columns):
        a = tw.TVT.values.astype(float); b = tw.GR.values.astype(float)
        ok = np.isfinite(a)&np.isfinite(b)
        if ok.sum()>=10:
            o = np.argsort(a[ok]); tt_s = a[ok][o]; tg_s = moving_avg(b[ok][o],5)
    return dict(wid=wid, MD=get("MD"), X=get("X"), Y=get("Y"), Z=get("Z"), GR=get("GR"),
                TVT=tv, TOP=get("EGFDU"), real_ps=ps, n=n, tt_s=tt_s, tg_s=tg_s, is_train=bool(has))

# ----------------------------- 构造先验 -----------------------------
def build_cloud(cache, wids):
    PX,PY,PT,PW = [],[],[],[]
    for i,wid in enumerate(wids):
        w = cache[wid]
        if not np.isfinite(w["TOP"]).any(): continue
        dz = np.abs(np.gradient(w["Z"], w["MD"]))
        idx = np.arange(0,w["n"],SP["ds"]); idx = idx[dz[idx]<0.3]
        if len(idx)==0: continue
        v = np.isfinite(w["TOP"][idx])
        if v.sum()==0: continue
        PX.append(w["X"][idx][v]); PY.append(w["Y"][idx][v]); PT.append(w["TOP"][idx][v])
        PW.append(np.full(int(v.sum()), i))
    PX=np.concatenate(PX);PY=np.concatenate(PY);PT=np.concatenate(PT);PW=np.concatenate(PW)
    return dict(tree=cKDTree(np.c_[PX,PY]),PX=PX,PY=PY,PT=PT,PW=PW,
                wid2idx={w:i for i,w in enumerate(wids)})

def struct_curve(cloud, w, ex=None):
    n=w["n"];X=w["X"];Y=w["Y"];MD=w["MD"];dec=SP["dec"];k=SP["k"];h=SP["h"];rg=SP["ridge"]
    qi=np.arange(0,n,dec)
    if qi[-1]!=n-1: qi=np.append(qi,n-1)
    dist,ii=cloud["tree"].query(np.c_[X[qi],Y[qi]],k=k)
    Wt=np.exp(-(dist/h)**2); pw=cloud["PW"][ii]
    if ex is not None:
        own=pw==ex; Wt=np.where(own,0.,Wt); dist=np.where(own,np.inf,dist)
    pwm=np.where(Wt>0.02*(Wt.max(1,keepdims=True)+1e-12),pw,-1); spw=np.sort(pwm,1)
    nw=((spw[:,1:]!=spw[:,:-1])&(spw[:,1:]>=0)).sum(1)+(spw[:,0]>=0)
    x0=cloud["PX"][ii]-X[qi][:,None]; y0=cloud["PY"][ii]-Y[qi][:,None]; z=cloud["PT"][ii]
    A11=Wt.sum(1); lam=rg*(A11+1e-12)*h*h
    A12=(Wt*x0).sum(1);A13=(Wt*y0).sum(1);A22=(Wt*x0*x0).sum(1)+lam
    A23=(Wt*x0*y0).sum(1);A33=(Wt*y0*y0).sum(1)+lam
    b1=(Wt*z).sum(1);b2=(Wt*z*x0).sum(1);b3=(Wt*z*y0).sum(1)
    Am=np.stack([np.stack([A11,A12,A13],-1),np.stack([A12,A22,A23],-1),
                 np.stack([A13,A23,A33],-1)],1)+np.eye(3)[None]*1e-8
    sol=np.linalg.solve(Am,np.stack([b1,b2,b3],-1)[:,:,None])[:,:,0]
    a,bg,cg=sol[:,0],sol[:,1],sol[:,2]
    um=nw<2; a=np.where(um,b1/(A11+1e-12),a); bg=np.where(um,0,bg); cg=np.where(um,0,cg)
    a=np.where(A11<1e-10,np.nan,a); do=dist.min(1)
    tx=np.gradient(X,MD);ty=np.gradient(Y,MD);nr=np.sqrt(tx**2+ty**2)+1e-9
    dip=bg*(tx[qi]/nr[qi])+cg*(ty[qi]/nr[qi])
    idx=np.arange(n)
    TH=np.interp(idx,qi,a); TH[np.interp(idx,qi,np.isnan(a).astype(float))>0]=np.nan
    return (TH, np.interp(idx,qi,np.where(np.isfinite(do),do,1e6)),
            np.interp(idx,qi,nw.astype(float)), np.interp(idx,qi,dip))

def struct_prior(w, ps):
    TH=w["TH"]; Z=w["Z"]; TVT=w["TVT"]; MD=w["MD"]; n=w["n"]
    vp=np.isfinite(TH)&(w["THd"]<SP["far"])
    lo=np.searchsorted(MD,MD[max(ps-1,0)]-SP["cwin"])
    sel=np.arange(lo,ps); sel=sel[vp[sel]&np.isfinite(TVT[sel])]
    if len(sel)<50:
        sel=np.arange(0,ps); sel=sel[vp[sel]&np.isfinite(TVT[sel])]
    if len(sel)<50: return None, np.zeros(n,bool)
    C=np.median(TVT[sel]+Z[sel]-TH[sel])
    return TH+C-Z, vp

def build_features(w, ps):
    n=w["n"];MD=w["MD"];Z=w["Z"];X=w["X"];Y=w["Y"];GR=w["GR"];TVT=w["TVT"]
    ps=int(min(max(ps,5),n-5)); tvt_ps=TVT[ps-1]; idx=np.arange(n)
    beyond=(idx>=ps).astype(np.float32)
    md_rel=(MD-MD[ps])/1000.; dz_ps=(Z[ps]-Z)/100.
    dZ=np.gradient(Z,MD); incl=moving_avg(dZ,25)
    curv=moving_avg(np.gradient(dZ,MD),25)*100.
    azim=moving_avg(np.arctan2(np.gradient(Y,MD),np.gradient(X,MD)),25)/math.pi
    m,s=robust_stats(GR[:ps]); gv=np.isfinite(GR); gn=(interp_nan(GR)-m)/s
    gs=moving_avg(gn,7); gg=np.gradient(gs,MD)*10.
    grs=moving_avg((gn-gs)**2,15)**.5; cov=moving_avg(gv.astype(float),25)
    tvt_in=np.where(idx<ps,np.nan_to_num(TVT-tvt_ps)/TVT_SCALE,0.)
    km=(idx<ps).astype(np.float32)
    dcm=np.zeros(n);dcp=np.zeros(n);dce=np.ones(n)
    grid=tvt_ps+OFFSETS; ref=np.full(N_OFF,np.nan)
    pt=TVT[:ps];pg=gn[:ps];good=np.isfinite(pt)&np.isfinite(GR[:ps])
    if good.sum()>=30:
        o=np.argsort(pt[good]); ref=np.interp(grid,pt[good][o],pg[good][o],left=np.nan,right=np.nan)
    if w["tt_s"] is not None and np.isnan(ref).any():
        tn=(w["tg_s"]-m)/s
        ref=np.where(np.isfinite(ref),ref,np.interp(grid,w["tt_s"],tn,left=np.nan,right=np.nan))
    if np.isfinite(ref).sum()>=3:
        ref=interp_nan(ref)
        ghp=gs-moving_avg(gs,HP_WIN); rhp=ref-moving_avg(ref,max(int(round(HP_WIN/OFF_G)),2))
        sc=-(ghp[:,None]-rhp[None,:])**2
        sc=box0(sc,CORR_SMOOTH)/(np.std(ghp)**2+1e-6)
        p=np.exp(sc-sc.max(1,keepdims=True)); p=p/p.sum(1,keepdims=True)
        dcm=(p*OFFSETS[None,:]).sum(1)/TVT_SCALE; dcp=p.max(1)
        dce=-(p*np.log(p+1e-9)).sum(1)/math.log(N_OFF)
    pr,vp=struct_prior(w,ps)
    if pr is not None:
        sr=np.where(vp,np.clip((pr-tvt_ps)/TVT_SCALE,-8,8),0.); sv=vp.astype(float)
        prf=interp_nan(np.where(vp,pr,np.nan))
        sl=np.where(vp,np.clip(np.gradient(prf,MD),-.2,.2)*20.,0.)
    else:
        sr=np.zeros(n);sv=np.zeros(n);sl=np.zeros(n)
    sd=np.clip(w["THd"]/1000.,0,3); sn=np.clip(w["THn"]/5.,0,2)
    sdp=np.clip(w["THdp"]*20.,-3,3); dzs=np.clip(np.gradient(Z,MD),-1.2,1.2)
    z0=np.zeros(n)   # tort_s / tort_l / tort_cum:训练时被 ablate 置零
    F=np.stack([md_rel,beyond,dz_ps,incl,curv,azim,gs,gv.astype(float),gg,grs,cov,
                tvt_in,km,dcm,dcp,dce,z0,z0,z0,sr,sd,sn,sv,sdp,sl,dzs],1).astype(np.float32)
    return np.nan_to_num(F), float(tvt_ps)

# ----------------------------- 模型 -----------------------------
class TCNBlock(nn.Module):
    def __init__(s,c,k,d,p):
        super().__init__(); pad=(k-1)*d//2
        s.c1=nn.Conv1d(c,c,k,padding=pad,dilation=d); s.c2=nn.Conv1d(c,c,k,padding=pad,dilation=d)
        s.n1=nn.GroupNorm(4,c); s.n2=nn.GroupNorm(4,c); s.drop=nn.Dropout(p); s.act=nn.GELU()
    def forward(s,x):
        h=s.drop(s.act(s.n1(s.c1(x)))); h=s.drop(s.act(s.n2(s.c2(h)))); return x+h
class Trunk(nn.Module):
    def __init__(s,nf,hid=48,dils=(1,2,4,8,16,32,64,128)):
        super().__init__(); s.stem=nn.Conv1d(nf,hid,5,stride=4,padding=2)
        s.blocks=nn.ModuleList([TCNBlock(hid,3,d,0.1) for d in dils])
    def forward(s,x):
        h=s.stem(x)
        for b in s.blocks: h=b(h)
        return h
class EdgeHead(nn.Module):
    def __init__(s,hid=48,wid=64,ev=0):
        super().__init__(); s.ev=ev
        s.mlp=nn.Sequential(nn.Linear(hid*3+2+ev,wid),nn.GELU(),nn.Linear(wid,wid),nn.GELU(),nn.Linear(wid,1))
    def forward(s,hi,hj,hp,dz,lg,evv=None):
        z=[hi,hj-hi,hp,dz[:,None],lg[:,None]]
        if s.ev>0: z.append(evv[:,:s.ev])
        return s.mlp(torch.cat(z,1)).squeeze(1)
class PSG(nn.Module):
    def __init__(s,nf,hid=48,ev=0):
        super().__init__(); s.trunk=Trunk(nf,hid); s.edge=EdgeHead(hid,ev=ev)

def make_nodes(w,ps):
    n=w["n"]; back=np.arange(ps-1,-1,-GRID)[::-1]; fwd=np.arange(ps-1+GRID,n,GRID)
    nodes=np.concatenate([back,fwd])
    if nodes[-1]!=n-1: nodes=np.append(nodes,n-1)
    return nodes.astype(int), int(np.searchsorted(nodes,ps-1))
def make_edges(nn_):
    ei,ej,es=[],[],[]
    for s_ in SCALES:
        st=max(1,s_//2); i=np.arange(0,nn_-s_,st)
        ei.append(i);ej.append(i+s_);es.append(np.full(len(i),s_))
    return np.concatenate(ei),np.concatenate(ej),np.concatenate(es)
def solve_anchored(nnode,fixed,fvals,ei,ej,dse,we):
    free=~fixed; nf=int(free.sum()); fidx=np.cumsum(free)-1
    A=np.zeros((nf,nf)); b=np.zeros(nf)
    Sfix=np.where(fixed,0.,0.); Sfix=np.zeros(nnode); Sfix[fixed]=fvals
    for i,j,d,w_ in zip(ei,ej,dse,we):
        fi_=fidx[i] if free[i] else -1; fj_=fidx[j] if free[j] else -1
        de=d+(Sfix[i] if not free[i] else 0.)-(Sfix[j] if not free[j] else 0.)
        if fj_>=0: A[fj_,fj_]+=w_; b[fj_]+=w_*de
        if fi_>=0: A[fi_,fi_]+=w_; b[fi_]-=w_*de
        if fi_>=0 and fj_>=0: A[fi_,fj_]-=w_; A[fj_,fi_]-=w_
    S=np.zeros(nnode); S[fixed]=fvals
    if nf: S[free]=np.linalg.solve(A+np.eye(nf)*1e-9,b)
    return S
def predict(model,w,ps,F):
    n=w["n"]
    with torch.no_grad():
        h=model.trunk(torch.tensor(F.T[None]).to(DEVICE))[0]
        Lt=h.shape[1]; nodes,anc=make_nodes(w,ps)
        tok=np.clip(nodes//4,0,Lt-1); ei,ej,es=make_edges(len(nodes))
        hn=h[:,torch.tensor(tok,dtype=torch.long,device=DEVICE)].T
        hc=torch.cumsum(h,1); ri=nodes[ei]; rj=nodes[ej]
        ti=np.clip(ri//4,0,Lt-1); tj=np.clip(rj//4,0,Lt-1)
        pool=(hc[:,torch.tensor(tj,device=DEVICE)]-hc[:,torch.tensor(ti,device=DEVICE)]).T \
             /torch.tensor(np.maximum(tj-ti,1)[:,None],dtype=torch.float32,device=DEVICE)
        Z=w["Z"]; Lft=torch.tensor((rj-ri).astype(np.float32),device=DEVICE)
        dzn=torch.tensor(((Z[rj]-Z[ri])/(rj-ri)).astype(np.float32),device=DEVICE)
        lg=torch.tensor(np.log2(es).astype(np.float32),device=DEVICE)
        ev=torch.zeros(len(ei),4,device=DEVICE)
        sl=model.edge(hn[torch.tensor(ei,device=DEVICE)],hn[torch.tensor(ej,device=DEVICE)],pool,dzn,lg,ev)
        dse=((dzn+0.15*torch.tanh(sl))*Lft).cpu().numpy().astype(float)
    Spre=np.nan_to_num(w["TVT"])+w["Z"]
    fixed=nodes<=(ps-1)
    we=(np.array([EDGE_W[s_] for s_ in es])/np.sqrt(es)).astype(float)
    Sn=solve_anchored(len(nodes),fixed,Spre[nodes[fixed]],ei,ej,dse,we)
    rows=np.arange(ps,n)
    return np.interp(rows,nodes,Sn)-w["Z"][rows], rows

# ----------------------------- 主流程 -----------------------------
print("loading wells …")
CACHE={}
for p in sorted(glob.glob(os.path.join(DATA,"train","*__horizontal_well.csv"))):
    w=load_well(p); CACHE[w["wid"]]=w
train_ids=[k for k,w in CACHE.items() if w["is_train"]]
test_paths=sorted(glob.glob(os.path.join(DATA,"test","*__horizontal_well.csv")))
for p in test_paths:
    w=load_well(p); CACHE[w["wid"]]=w
test_ids=[os.path.basename(p).split("__")[0] for p in test_paths]
print(f"train={len(train_ids)} test={len(test_ids)}  ({time.time()-t0:.0f}s)")

CLOUD=build_cloud(CACHE,train_ids)
for wid in test_ids:
    w=CACHE[wid]; w["TH"],w["THd"],w["THn"],w["THdp"]=struct_curve(CLOUD,w,None)
print(f"struct prior done ({time.time()-t0:.0f}s)")

models=[]
for ck in sorted(CKPTS):
    payload=torch.load(ck,map_location="cpu")
    if isinstance(payload,dict) and "model" in payload: sd=payload["model"]
    elif isinstance(payload,dict) and "state_dict" in payload: sd=payload["state_dict"]
    else: sd=payload
    ev=int(sd["edge.mlp.0.weight"].shape[1]-(48*3+2))
    m=PSG(N_FEAT,48,ev).to(DEVICE); m.load_state_dict(sd); m.eval(); models.append(m)
print(f"loaded {len(models)} PSG models")

PSG_PRED={}
for wid in test_ids:
    w=CACHE[wid]; ps=w["real_ps"]
    if ps>=w["n"]: continue
    F,tvt_ps=build_features(w,ps)
    acc=None
    for m in models:
        p,rows=predict(m,w,ps,F)
        acc=p if acc is None else acc+p
    p=acc/len(models)
    p=np.where(np.isfinite(p),p,tvt_ps)
    PSG_PRED[wid]=dict(zip(rows.tolist(),p.tolist()))
print(f"PSG inference done: {len(PSG_PRED)} wells  ({time.time()-t0:.0f}s)")

# ----------------------------- 融合进 submission -----------------------------
WORK="/kaggle/working" if os.path.isdir("/kaggle/working") else "."
submission_path=os.path.join(WORK,"submission.csv")
sub=pd.read_csv(submission_path)
if list(sub.columns)!=["id","tvt"]:
    raise RuntimeError(f"submission.csv must have exactly ['id','tvt'], got {list(sub.columns)}")
if sub["id"].duplicated().any():
    raise RuntimeError("submission.csv contains duplicate ids")
parsed=sub["id"].astype(str).str.extract(r"^([0-9a-fA-F]{8})_(\d+)$")
if parsed.isna().any().any():
    raise RuntimeError(f"Cannot parse submission ids: {sub.loc[parsed.isna().any(axis=1),'id'].head().tolist()}")
sub["well"]=parsed[0].str.lower(); sub["row"]=parsed[1].astype(int)
base=sub["tvt"].values.astype(float).copy()
if not np.isfinite(base).all():
    raise RuntimeError("Upstream submission contains non-finite TVT values")
pd.DataFrame({"id":sub["id"],"tvt":base}).to_csv(
    os.path.join(WORK,"submission_before_psg.csv"),index=False
)

out=base.copy(); psg_full=np.full(len(sub),np.nan); available=np.zeros(len(sub),bool)
eligible=np.zeros(len(sub),bool); nw=nr=0
for wid,g in sub.groupby("well"):
    if wid not in PSG_PRED: continue
    idx=g.index.values; rows=g["row"].values
    pv=np.array([PSG_PRED[wid].get(int(r),np.nan) for r in rows])
    have=np.isfinite(pv)
    ok=have&(np.abs(pv-base[idx])<FUSE_CLIP)
    psg_full[idx[have]]=pv[have]; available[idx[have]]=True; eligible[idx[ok]]=True
    out[idx[ok]]=(1-BLEND_W)*base[idx[ok]]+BLEND_W*pv[ok]
    nw+=int(have.any()); nr+=int(ok.sum())

availability=float(available.mean()); eligible_fraction=float(eligible.mean())
if availability<0.99:
    missing_ids=sub.loc[~available,"id"].head(20).tolist()
    raise RuntimeError(
        f"PSG prediction availability {availability:.4%} < 99%; examples={missing_ids}"
    )
if not np.isfinite(out).all():
    raise RuntimeError("PSG blend produced non-finite TVT values")

pd.DataFrame({"id":sub["id"],"tvt":psg_full}).to_csv(
    os.path.join(WORK,"submission_psg_raw.csv"),index=False
)
candidate_name=f"submission_psg_blend_w{BLEND_W:.6f}.csv"
candidate_path=os.path.join(WORK,candidate_name)
final_frame=pd.DataFrame({"id":sub["id"],"tvt":out})
final_frame.to_csv(candidate_path,index=False)
final_frame.to_csv(submission_path,index=False)

audit=[]
for wid,g in sub.groupby("well"):
    idx=g.index.values
    audit.append({
        "well":wid,
        "rows":int(len(idx)),
        "psg_available_rows":int(available[idx].sum()),
        "blended_rows":int(eligible[idx].sum()),
        "mean_abs_psg_move":float(np.nanmean(np.abs(psg_full[idx]-base[idx]))),
        "mean_abs_applied_move":float(np.mean(np.abs(out[idx]-base[idx]))),
        "max_abs_applied_move":float(np.max(np.abs(out[idx]-base[idx]))),
    })
pd.DataFrame(audit).to_csv(os.path.join(WORK,"psg_blend_audit_bywell.csv"),index=False)

report={
    "BLEND_W":float(BLEND_W),
    "FUSE_CLIP":float(FUSE_CLIP),
    "checkpoints":CKPTS,
    "checkpoint_folds":fold_numbers,
    "submission_rows":int(len(sub)),
    "submission_wells":int(sub["well"].nunique()),
    "psg_prediction_wells":int(nw),
    "psg_availability_fraction":availability,
    "blend_eligible_rows":int(nr),
    "blend_eligible_fraction":eligible_fraction,
    "mean_abs_applied_move":float(np.mean(np.abs(out-base))),
    "p95_abs_applied_move":float(np.quantile(np.abs(out-base),.95)),
    "max_abs_applied_move":float(np.max(np.abs(out-base))),
    "base_snapshot":os.path.join(WORK,"submission_before_psg.csv"),
    "psg_raw":os.path.join(WORK,"submission_psg_raw.csv"),
    "candidate":candidate_path,
    "final":submission_path,
    "elapsed_seconds":float(time.time()-t0),
}
with open(os.path.join(WORK,"psg_blend_report.json"),"w",encoding="utf-8") as handle:
    import json
    json.dump(report,handle,indent=2)
print("PSG BLEND COMPLETE")
print(json.dumps(report,indent=2))
