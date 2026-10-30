"""Evaluate vector, lexical, and RRF fusion rankings against fixed relevance judgments."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"services"/"ranking")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ranking_service import fuse
from common import write_result

def dcg(rels,k): return sum((2**r-1)/__import__("math").log2(i+2) for i,r in enumerate(rels[:k]))
def score(ids,judgments,k):
 rels=[judgments.get(i,0) for i in ids[:k]]; ideal=sorted(judgments.values(),reverse=True)[:k]; idcg=dcg(ideal,k)
 return {"mrr_at_10":next((1/(i+1) for i,r in enumerate(rels) if r>0),0.0),"ndcg_at_10":dcg(rels,k)/idcg if idcg else 0.0,
  "recall_at_5":sum(1 for i in ids[:5] if judgments.get(i,0)>0)/max(1,sum(1 for r in judgments.values() if r>0))}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--rrf-k",type=int,default=60); p.add_argument("--output",default="evaluations/results"); a=p.parse_args()
 fixtures=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/"fixtures"/"hybrid.jsonl").read_text().splitlines() if x.strip()]; cases=[]
 for f in fixtures:
  lists={name:[{"key":key,"rank":i+1} for i,key in enumerate(f[name])] for name in ("vector","lexical")}
  fused=fuse(lists,top_k=100,k=a.rrf_k); fused_ids=[x["key"] for x in fused]; judgments=f["relevance"]
  cases.append({"query_id":f["query_id"],"vector":score(f["vector"],judgments,10),"lexical":score(f["lexical"],judgments,10),
   "rrf":score(fused_ids,judgments,10),"rrf_k":a.rrf_k,"rrf_order":fused_ids})
 out=write_result("hybrid",{"method":"production ranking_service.fuse","cases":cases},a.output); print(out)
if __name__=="__main__": main()
