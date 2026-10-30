"""Measure chunking profiles for throughput, coverage, offsets, and determinism."""
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"services"/"chunking")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chunking_service import chunk
from common import summarize_ms, write_result

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--iterations",type=int,default=20); p.add_argument("--output",default="evaluations/results"); a=p.parse_args()
 fixtures=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/"fixtures"/"chunking.jsonl").read_text().splitlines() if x.strip()]
 cases=[]
 for f in fixtures:
  fn=lambda: chunk(f["profile"],f["source_id"],f["text"],f["media_type"])
  expected=fn(); samples=[]
  for _ in range(a.iterations):
   t=time.perf_counter(); records=fn(); samples.append((time.perf_counter()-t)*1000)
  text_len=len(f["text"]); intervals=sorted((r["start"],r["end"]) for r in records); covered=0; cursor=0
  for start,end in intervals:
   if end>cursor: covered+=end-max(start,cursor); cursor=max(cursor,end)
  cases.append({"case":f["case"],"profile":f["profile"],"input_chars":text_len,"chunk_count":len(records),"timing":summarize_ms(samples),
   "chars_per_second":text_len/(sum(samples)/1000/len(samples)) if samples else None,"deterministic":records==expected,
   "offsets_valid":all(f["text"][r["start"]:r["end"]]==r["text"] for r in records),"covered_characters_including_overlap":covered,
   "coverage_ratio":(covered / text_len) if text_len else 1.0})
 out=write_result("chunking",{"iterations":a.iterations,"cases":cases},a.output); print(out)
if __name__=="__main__": main()
