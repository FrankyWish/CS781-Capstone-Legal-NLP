import sys, random; sys.path.insert(0, r"E:\cs781\capstone_Project\phase7\code")
import phase7_group as G
r=random.Random(0)
cands=[f"c{i}" for i in range(200)]
Pc={c:{"cit":set(),"air":None} for c in cands}
qs=[];Pq={};base={};gold={}
for i in range(20):
    q=f"q{i}"; g=r.sample(cands,4); key=(1990+i%5, 100+i)
    for c in g: Pc[c]["cit"].add(key)
    noise=r.sample(cands,40)
    rk=noise[:]; 
    for j,c in enumerate(g): rk.insert(r.randint(0,15),c)
    qs.append({"id":q}); Pq[q]={"air":None,"date":None,"yest":1985}; base[q]=rk; gold[q]=g
for c in r.sample(cands,60): Pc[c]["cit"].add((1999, r.randint(1,30)))   # distractor keys
out=G.assign(qs,{c:1 for c in cands},Pq,Pc,base,3,1.3)
n,tp=G.precision(out,gold); print("queries",len(out),"pairs",n,"correct",tp)
import phase7_citator as C
C.GROUPS.clear(); C.GROUPS.update(out)
A={q:set() for q in base}; L=A; B=A
rk=C.combine(qs,{c:[""] for c in cands},{q:{"date":None} for q in base},{},A,B,L,base,1,1,0,1,use_g=1)
q="q0"; print("base top7", base[q][:7]); print("g    top7", rk[q][:7]); print("gold", out.get(q))
