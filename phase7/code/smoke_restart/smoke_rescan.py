import os, sys, tempfile, json, random, importlib
sys.path.insert(0, r"E:\cs781\capstone_Project\phase7\code")
def run(old):
    T=tempfile.mkdtemp(); os.environ["CS781_P7"]=T
    import phase7_data as D, phase7_overlap as O, phase7_names as NM, phase7_hc as H
    D.P7=O.P7=NM.P7=H.P7=T
    r=random.Random(0)
    W=[f"word{i}xyz" for i in range(3000)]
    names=[f"person{i}" for i in range(50)]
    def doc(n):
        out=[]
        for i in range(n):
            out.append(r.choice(names)+"name" if i%15==7 else r.choice(W))
        return " ".join(out)
    full_q=doc(300)+" relied on kartar singh v state of punjab 1994 3 scc 569 and "+doc(300)
    import re
    masked_q=re.sub(r"person\d+name","<NAME>",full_q).replace("kartar singh v state of punjab 1994 3 scc 569","<CITATION>")
    cand_full=doc(500); cand=re.sub(r"person\d+name","<NAME>",cand_full)
    corpus_c="Kartar Singh vs State Of Punjab on 11 March, 1994 "+cand_full
    qs={"devx":[{"id":"q1","sents":[masked_q],"gold":["c1"]}],"test":[{"id":"t1","sents":[masked_q],"gold":[]}]}
    cs={"devx":{"c1":[cand]},"test":{"c1":[cand]}}
    D.queries=lambda s: qs[s]; D.candidates=lambda s: cs[s]
    path=os.path.join(T,"corpus.jsonl")
    with open(path,"w") as f:
        for t in [doc(400), full_q, corpus_c, doc(400)]: f.write(json.dumps({"text":t})+"\n")
    import huggingface_hub
    huggingface_hub.hf_hub_download=lambda *a,**k: path
    sys.argv=["x","--minutes","5"]+(["--old-fps","--needles","10","--min-hits","3"] if old else [])
    H.main()
    print("OLD" if old else "NEW", json.load(open(T+"/results/overlap/recovered_devx.json")),
          json.load(open(T+"/results/names/titles_devx.json")))
run(True); run(False)
