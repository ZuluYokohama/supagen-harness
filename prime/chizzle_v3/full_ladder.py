import json, math, numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

A = json.load(open('loops.json')); B = json.load(open('loops_new.json'))
loops = [L for L in A+B if len(L) >= 5]
print(f"total loops {len(loops)}  (expect 48 = 8 seeds x 6 rungs)")

SEEDS = ["A claim may not open unless","Strain that no choice of dihedrals",
 "The decoder was offline while","An instrument that returns the same",
 "Sequence sets the cost of ring","Where the mismatch is written down",
 "A lift earns its keep only if","Longer chains tolerate more accumulated"]
sidx = lambda L: next((i for i,s in enumerate(SEEDS) if L[0].startswith(s)), -1)

# rung = position in the sweep; harness iterates rung outer, seed inner
for k, L in enumerate(loops): L.append(f"__rung{k//8}")
def rung(L): return int(L[-1].replace('__rung',''))
def body(L): return L[:-1]

def _rot(u,v):
    c=float(np.clip(u@v,-1,1)); ax=np.cross(u,v); s=norm(ax)
    if s<1e-12: return np.eye(3) if c>0 else -np.eye(3)
    ax=ax/s; th=math.atan2(s,c)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+math.sin(th)*K+(1-math.cos(th))*K@K
def bishop(P):
    n=len(P); T=np.array([P[(i+1)%n]-P[i] for i in range(n)])
    nz=norm(T,axis=1,keepdims=True)
    if (nz<1e-12).any(): return float('nan')
    T=T/nz; u=np.cross(T[0],[0.,0.,1.])
    if norm(u)<1e-8: u=np.cross(T[0],[0.,1.,0.])
    u0=u/norm(u); u=u0.copy()
    for i in range(n): u=_rot(T[i],T[(i+1)%n])@u
    u-= (u@T[0])*T[0]; u/=norm(u)
    return float(math.atan2(float(np.cross(u0,u)@T[0]),float(np.clip(u0@u,-1,1))))
def proj3(E):
    X=E-E.mean(0); _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:3].T

m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
rows=[]
for L in loops:
    b = body(L); E = np.asarray(m.encode(b, show_progress_bar=False), float)
    En = E/np.linalg.norm(E,axis=1,keepdims=True)
    th = bishop(proj3(E))
    rows.append(dict(rung=rung(L), seed=sidx(L), N=len(b), theta=th,
                     ret=float(En[0]@En[-1]),
                     lam=2-2*math.cos(th/len(b)) if np.isfinite(th) else float('nan')))

print(f"\n  {'rung':>5}{'n':>4}{'median ret':>12}{'closed@.6':>11}{'median|th|':>12}{'median lam':>12}")
for r in sorted(set(x['rung'] for x in rows)):
    g=[x for x in rows if x['rung']==r]
    ret=np.median([x['ret'] for x in g]); cl=np.mean([x['ret']>=0.6 for x in g])
    print(f"  {r:>5}{len(g):>4}{ret:>12.3f}{cl:>11.2f}"
          f"{np.median([abs(x['theta']) for x in g]):>12.3f}"
          f"{np.median([x['lam'] for x in g]):>12.5f}")

print("\n=== THE ACTUAL S-CURVE: closure rate vs rung ===")
xs=[];ys=[]
for r in sorted(set(x['rung'] for x in rows)):
    g=[x for x in rows if x['rung']==r]
    y=float(np.mean([x['ret']>=0.5 for x in g])); xs.append(r); ys.append(y)
    print(f"  rung {r}: closure {y:.3f}   {'#'*int(40*y)}")
xs=np.array(xs,float); ys=np.array(ys)
msk=(ys>1e-6)&(ys<1-1e-6)
if msk.sum()>=3:
    z=np.log(ys[msk]/(1-ys[msk])); Aa=np.vstack([xs[msk],np.ones(msk.sum())]).T
    k,bb=np.linalg.lstsq(Aa,z,rcond=None)[0]
    print(f"\n  logistic fit: midpoint {-bb/k:.2f} rungs, slope {k:.3f}, implied sigma {(-1/k)/0.47:.3f}")
else:
    print(f"\n  only {msk.sum()} interior points -- cannot fit. Curve is saturated.")
json.dump(rows, open('ladder.json','w'), default=float)
