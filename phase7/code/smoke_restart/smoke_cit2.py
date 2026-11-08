import os, sys, tempfile, random, argparse, glob, zipfile, json
T=tempfile.mkdtemp(); os.environ["CS781_P7"]=T
sys.path.insert(0, r"E:\cs781\capstone_Project\phase7\code")
import phase7_data as D, phase7_rank as R, phase7_citator as C
D.P7=R.P7=C.P7=T
r=random.Random(0)
def mk(sp,Q,Cn):
    pool={}; qs=[]
    keys={}
    for j in range(Cn):
        keys[f"c{sp}{j}"]=(1950+j%40, 100+j)
    gold={}
    for i in range(Q):
        gold[f"q{sp}{i}"]=r.sample(list(keys),5)
    citedby={c:[] for c in keys}
    for i in range(Q):
        for c in gold[f"q{sp}{i}"]: citedby[c].append((1995, 500+i))
    for c,(y,pg) in keys.items():
        sc= c.endswith(('0','1','2','3','4','5'))   # 60% are SC-format
        cit=" ".join(f"R {a} SC{b} (3)" for a,b in citedby[c])
        pool[c]=[f"DATE OF JUDGMENT: 01/01/{y} CITATION: {y} AIR {pg} CITATOR INFO : {cit} ACT: x HEADNOTE: y"] if sc else [f"CITATION: {y} AIR {pg} plain high court text"]
    for i in range(Q):
        qs.append({"id":f"q{sp}{i}","sents":[f"DATE OF JUDGMENT: 01/01/1995 CITATION: 1995 AIR {500+i} 1995 SCR (2) 9 CITATOR INFO : ACT: z"],"gold":gold[f"q{sp}{i}"] if sp!="test" else []})
    return qs,pool
DATA={s:mk(s,40,300) for s in ("train","devx","test")}
D.queries=lambda s: DATA[s][0]; D.candidates=lambda s: DATA[s][1]
D.golds=lambda s:{q["id"]:q["gold"] for q in DATA[s][0]}; D.golds_in_pool=D.golds
D.popularity_order=lambda s: list(DATA[s][1]); D.validate=lambda pj: True
os.makedirs(T+"/results/overlap",exist_ok=True)
for sp in ("devx","test"):
    keys2={c:(1950+int(c[len("c"+sp):])%40,100+int(c[len("c"+sp):])) for c in DATA[sp][1]}
    rec={q["id"]:[f"relied on air {keys2[c][0]} sc {keys2[c][1]}" for c in (q["gold"] or r.sample(list(keys2),3))[:3]] for q in DATA[sp][0]}
    json.dump(rec,open(T+f"/results/overlap/recovered_{sp}.json","w"))
for sp in DATA:
    R.save_run("ot2_zst3_p" if sp!="train" else "ot2_zlex2_p", sp, {q["id"]: r.sample(list(DATA[sp][1]),50) for q in DATA[sp][0]})
C.cmd_all(argparse.Namespace(bases="ot2_zst3_p",train_bases="ot2_zlex2_p",nzip=3,no_infer=False,variants="abdl,abdlt,abdlvt",suffix="_n90"))
for z in glob.glob(T+"/submissions/_upload/*.zip"):
    pj=json.loads(zipfile.ZipFile(z).read("prediction.json")); assert all(len(set(v))==7 for v in pj.values())
print("SMOKE OK")
