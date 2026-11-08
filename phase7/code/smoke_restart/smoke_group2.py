import os, sys, tempfile, json, argparse, random, glob
T=tempfile.mkdtemp(); os.environ["CS781_P7"]=T
sys.path.insert(0, r"E:\cs781\capstone_Project\phase7\code")
import phase7_data as D, phase7_rank as R, phase7_names as N, phase7_citator as C
D.P7=R.P7=N.P7=C.P7=T; N.OUT=T+"/results/names"
r=random.Random(1)
P=["union of india","delhi cloth general mills","kartar singh","state of punjab","voltas ltd","a k roy","tata iron steel","ram prasad","state of bihar","municipal board agra","sita devi","hari chand"]
def mk(sp):
    pool={}; titles={}
    for j in range(60):
        a,b=r.sample(P,2); y=1950+j
        pool[f"c{sp}{j}"]=[f"PETITIONER: <NAME> Vs. RESPONDENT: <NAME> DATE OF JUDGMENT: 01/01/{y} body"]
        titles[f"c{sp}{j}"]={"p1":a,"p2":b,"year":y}
    qs=[]; rec={}
    for i in range(20):
        g=r.sample(list(pool),3)
        txt=" ".join(f"As held in {titles[c]['p1'].title()} v. {titles[c]['p2'].title()}, {titles[c]['year']} and" for c in g)
        qs.append({"id":f"q{sp}{i}","sents":["DATE OF JUDGMENT: 01/01/2010 "+txt],"gold":g if sp!="test" else []})
        rec[f"q{sp}{i}"]=[f"relied on {titles[c]['p1']} v {titles[c]['p2']} {titles[c]['year']}" for c in (g if sp!="test" else r.sample(list(pool),3))]+["some long irrelevant text"]
    return qs,pool,titles,rec
DATA={s:mk(s) for s in ("train","devx","test")}
D.queries=lambda s: DATA[s][0]; D.candidates=lambda s: DATA[s][1]
D.golds=lambda s:{q["id"]:q["gold"] for q in DATA[s][0]}; D.golds_in_pool=D.golds
D.popularity_order=lambda s: list(DATA[s][1]); D.validate=lambda pj: True
os.makedirs(T+"/results/names",exist_ok=True); os.makedirs(T+"/results/overlap",exist_ok=True)
for sp in ("devx","test"):
    json.dump(DATA[sp][2],open(f"{T}/results/names/titles_{sp}.json","w"))
    json.dump(DATA[sp][3],open(f"{T}/results/overlap/recovered_{sp}.json","w"))
N.cmd_match(argparse.Namespace(min_precision=0.5,thr=0.6,hi=0.9))
sys.argv=["x","body","--min-precision","0.5"]; N.main()
for sp in DATA:
    R.save_run("ot2_zst3_p" if sp!="train" else "ot2_zlex2_p", sp, {q["id"]: r.sample(list(DATA[sp][1]),30) for q in DATA[sp][0]})
C.cmd_all(argparse.Namespace(bases="ot2_zst3_p",train_bases="ot2_zlex2_p",nzip=3,no_infer=False,variants="abdl,abdlg,abdlvg",suffix="_g",group_cfg="4,3.0",group_min_prec=0.0,group_tier=5))
print("SMOKE OK")
