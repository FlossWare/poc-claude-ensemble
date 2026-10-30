"""Benchmark only already-installed Ollama models; this runner never pulls models.

Example: python3 evaluations/llm/run_ollama.py --models qwen3.5:2b
"""
import argparse,json,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import command_output,write_result


def request(base,path,payload=None,timeout=600):
 data=json.dumps(payload).encode() if payload is not None else None
 req=urllib.request.Request(base+path,data=data,method="POST" if payload is not None else "GET",headers={"content-type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as response:return json.loads(response.read())

def rate(tokens,duration_ns): return tokens/(duration_ns/1e9) if tokens and duration_ns else None

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--models",nargs="+",required=True); p.add_argument("--host",default="http://127.0.0.1:11434"); p.add_argument("--iterations",type=int,default=2); p.add_argument("--context",type=int,default=8192); p.add_argument("--output",default="evaluations/results"); a=p.parse_args()
 if a.iterations<1 or a.context<1: p.error("iterations and context must be positive")
 root=Path(__file__).resolve().parents[1]; prompts=[json.loads(x) for x in (root/"fixtures"/"llm-prompts.jsonl").read_text().splitlines() if x.strip()]
 try: tags=request(a.host,"/api/tags",timeout=5)
 except Exception as exc:p.error(f"Ollama API unavailable at {a.host}: {exc}")
 installed={m["name"]:m for m in tags.get("models",[])}; cases=[]
 for name in a.models:
  if name not in installed:
   cases.append({"model":name,"status":"not_installed","note":"Skipped. The runner never downloads models."}); continue
  model=installed[name]; model_cases=[]
  for fixture in prompts:
   samples=[]; outputs=[]; last=None
   for iteration in range(a.iterations):
    start=time.perf_counter()
    try:
     last=request(a.host,"/api/generate",{"model":name,"prompt":fixture["prompt"],"stream":False,
      "options":{"temperature":0,"num_ctx":a.context}},timeout=900)
    except Exception as exc:
     model_cases.append({"case":fixture["case"],"status":"error","error":str(exc)}); last=None; break
    samples.append((time.perf_counter()-start)*1000); outputs.append(last.get("response",""))
   if last is None:continue
   text=outputs[-1]; expects_json=fixture["case"] in {"intent-classification","structured-extraction","json-compliance"}
   json_valid=True
   if expects_json:
    try:json.loads(text)
    except (ValueError,TypeError):json_valid=False
   model_cases.append({"case":fixture["case"],"status":"ok","wall_ms_samples":samples,
    "wall_ms_mean":sum(samples)/len(samples),"prompt_tokens":last.get("prompt_eval_count"),
    "prompt_tokens_per_second":rate(last.get("prompt_eval_count"),last.get("prompt_eval_duration")),
    "generated_tokens":last.get("eval_count"),"generation_tokens_per_second":rate(last.get("eval_count"),last.get("eval_duration")),
    "load_duration_ns":last.get("load_duration"),"total_duration_ns":last.get("total_duration"),
    "json_expected":expects_json,"json_valid":json_valid,"response":text})
  ps=command_output(["ollama","ps"])
  cases.append({"model":name,"status":"evaluated","digest":model.get("digest"),"size_bytes":model.get("size"),"details":model.get("details"),
   "ollama_ps_after_run":ps,"cases":model_cases})
 out=write_result("llm",{"runtime":{"ollama_api":a.host,"context":a.context,"iterations":a.iterations},"cases":cases},a.output);print(out)
if __name__=="__main__":main()
