import os, sys, tempfile, json, numpy as np, argparse
T=tempfile.mkdtemp(); os.environ["CS781_P7"]=T
sys.path.insert(0, r"E:\cs781\capstone_Project\phase7\code")
import phase7_data as D, phase7_rank as R
D.P7=T; R.P7=T
import phase7_ot as O
O.P7=T
rng=np.random.default_rng(0)
def make(split,Q,C,seed):
    r=np.random.default_rng(seed)
    qs=[f"{split}q{i}" for i in range(Q)]
    pool=[f"{split}c{j}" for j in range(C)]+qs[:Q//5]   # some queries in pool
    # each non-query candidate cited by exactly one query (~C/Q golds each)
    owner=r.integers(0,Q,C)
    gold={q:[pool[j] for j in range(C) if owner[j]==i] for i,q in enumerate(qs)}
    hub=np.zeros(len(pool)); hub[r.choice(C,C//15,replace=False)]=2.0
    runs={}
    for i,q in enumerate(qs):
        s=r.normal(0,1,len(pool))+hub
        for j in range(C):
            if owner[j]==i: s[j]+=2.2
        for k,c in enumerate(pool):
            if c==q: s[k]=-1e9
        top=np.argsort(-s)[:100]
        runs[q]=([pool[t] for t in top], s[top].astype(np.float32))
    return qs,pool,gold,runs
DATA={}
for sp,Q,C,sd in (("train",120,700,1),("devx",40,330,2),("test",60,420,3)):
    DATA[sp]=make(sp,Q,C,sd)
D.queries=lambda sp: [{"id":q,"sents":["x"],"gold":DATA[sp][2][q]} for q in DATA[sp][0]]
D.candidates=lambda sp: {c:["t"] for c in DATA[sp][1]}
D.golds=lambda sp: {q:list(v) for q,v in DATA[sp][2].items()}
D.golds_in_pool=D.golds
R.self_excluded=lambda sp:{}
for t in ("citep","maxp2","stack_new"):
    for sp in DATA:
        if t=="stack_new" and sp=="train": continue
        R.save_run(t,sp,{q:v[0] for q,v in DATA[sp][3].items()},{q:v[1] for q,v in DATA[sp][3].items()})
D.popularity_order=lambda sp: list(DATA[sp][1])
D.validate=lambda pj: True

import phase7_ot2 as O2; O2.P7=T; O2.cmd_all(argparse.Namespace(minutes=10,nzip=4))
import glob; zs=glob.glob(T+"/submissions/_upload/*.zip"); print("ZIPS",zs)
import zipfile
for z in zs:
    pj=json.loads(zipfile.ZipFile(z).read("prediction.json")); assert len(pj)==60 and all(len(set(v))==7 for v in pj.values())
print("SMOKE OK")
