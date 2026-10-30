"""Evaluate the pinned local cross-encoder on labeled query/document candidates."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"services"/"embeddings")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import package_versions,summarize_ms,write_result

def dcg(rels,k): return sum((2**rel-1)/__import__("math").log2(i+2) for i,rel in enumerate(rels[:k]))
def metrics(ranked,k=10):
 rels=[x["relevance"] for x in ranked[:k]]; ideal=sorted((x["relevance"] for x in ranked),reverse=True)[:k]; idcg=dcg(ideal,k)
 rr=next((1/(i+1) for i,x in enumerate(ranked[:k]) if x["relevance"]>0),0.0)
 return {"mrr_at_10":rr,"ndcg_at_10":dcg(rels,k)/idcg if idcg else 0.0}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--batch-sizes",type=int,nargs="+",default=[8,16,32]); p.add_argument("--warmups",type=int,default=1); p.add_argument("--iterations",type=int,default=3); p.add_argument("--output",default="evaluations/results"); a=p.parse_args()
 if min(a.batch_sizes)<1 or a.warmups<0 or a.iterations<1: p.error("batch sizes and iterations must be positive; warmups must be non-negative")
 import reranking_service as rs
 os.environ["HF_HUB_OFFLINE"]="1"
 os.environ["TRANSFORMERS_OFFLINE"]="1"
 from sentence_transformers import CrossEncoder
 model_dir=Path(rs.MODEL_DIR)
 if not model_dir.is_dir(): p.error(f"pinned local reranker is missing: {model_dir}; no model downloads are performed")
 t=time.perf_counter(); model=CrossEncoder(str(model_dir),device="cpu"); load_ms=(time.perf_counter()-t)*1000
 fixtures=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/"fixtures"/"reranking.jsonl").read_text().splitlines() if x.strip()]
 cases=[]
 for f in fixtures:
  docs=f["documents"]; baseline=metrics(docs); pairs=[(f["query"],d["text"]) for d in docs]
  for batch in a.batch_sizes:
   for _ in range(a.warmups): model.predict(pairs,batch_size=batch,show_progress_bar=False)
   samples=[]; scores=None
   for _ in range(a.iterations):
    t=time.perf_counter(); scores=model.predict(pairs,batch_size=batch,show_progress_bar=False); samples.append((time.perf_counter()-t)*1000)
   ranked=[dict(d,score=float(s)) for d,s in zip(docs,scores)]; ranked.sort(key=lambda x:-x["score"])
   cases.append({"query_id":f["query_id"],"batch_size":batch,"candidate_count":len(docs),"baseline_input_order":baseline,
    "reranked":metrics(ranked),"timing":summarize_ms(samples),"pairs_per_second":len(pairs)/(sum(samples)/1000/len(samples)),
    "ranking_ids":[x["id"] for x in ranked]})
 out=write_result("reranking",{"model":{"id":rs.MODEL_ID,"revision":rs.MODEL_REVISION,"directory":str(model_dir)},
  "runtime":{"device":"cpu","packages":package_versions(["sentence-transformers","torch","transformers","numpy"])},
  "load_ms":load_ms,"warmups":a.warmups,"iterations":a.iterations,"cases":cases},a.output); print(out)
if __name__=="__main__": main()
