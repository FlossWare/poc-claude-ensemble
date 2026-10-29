"""Decision Support analytics over Graph and Memory REST services.

This module intentionally keeps sample count, observed success rate, and recommendation
selection separate. It does not manufacture probabilistic confidence from sample size.
"""
from __future__ import annotations
import json, os, urllib.error, urllib.request
from collections import defaultdict
from typing import Any

def _post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req=urllib.request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type","application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as r: return json.loads(r.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"service unavailable: {exc}") from exc

class LearningAnalytics:
    def __init__(self, graph_service_url: str|None=None, memory_service_url: str|None=None):
        self.graph_service_url=graph_service_url or os.environ.get("ENSEMBLE_GRAPH_URL","http://127.0.0.1:8766")
        self.memory_service_url=memory_service_url or os.environ.get("ENSEMBLE_MEMORY_URL","http://127.0.0.1:8767")

    def best_models_for(self, task_type: str, scope: str|None=None, limit: int=5) -> dict[str,Any]:
        success=_post(self.graph_service_url,"/graph/query",{"type":"edges","from_type":"model","to_type":"task_type","relationship":"succeeded_on"}).get("edges",[])
        failure=_post(self.graph_service_url,"/graph/query",{"type":"edges","from_type":"model","to_type":"task_type","relationship":"failed_on"}).get("edges",[])
        stats=defaultdict(lambda: {"success":0,"failure":0,"costs":[]})
        for edge in success:
            if edge.get("to_id")==f"task:{task_type}":
                model=str(edge.get("from_id","")).removeprefix("model:")
                stats[model]["success"]+=1
                cost=edge.get("properties",{}).get("cost")
                if isinstance(cost,(int,float)): stats[model]["costs"].append(float(cost))
        for edge in failure:
            if edge.get("to_id")==f"task:{task_type}":
                model=str(edge.get("from_id","")).removeprefix("model:")
                stats[model]["failure"]+=1
        rows=[]
        for model,s in stats.items():
            sample=s["success"]+s["failure"]
            if not sample: continue
            rows.append({
                "model":model,
                "empirical_success_rate":round(s["success"]/sample,4),
                "sample_count":sample,
                "avg_cost":round(sum(s["costs"])/len(s["costs"]),4) if s["costs"] else None,
            })
        rows.sort(key=lambda x:(-x["empirical_success_rate"], x["avg_cost"] is None, x["avg_cost"] if x["avg_cost"] is not None else 0.0, x["model"]))
        return {"ok":True,"error":None,"data":rows[:limit],"sample_count":sum(r["sample_count"] for r in rows)}

    def cost_quality_tradeoff(self, task_type: str) -> dict[str,Any]:
        result=self.best_models_for(task_type,limit=20)
        rows=result.get("data",[])
        if not rows: return {"ok":False,"error":f"No data for {task_type}","data":None}
        priced=[row for row in rows if row["avg_cost"] is not None]
        if not priced:
            return {"ok":False,"error":"No observed cost data","error_code":"no_data","data":None}
        cheap=min(priced,key=lambda x:x["avg_cost"])
        quality=max(rows,key=lambda x:x["empirical_success_rate"])
        balanced=min(priced,key=lambda x:abs(x["empirical_success_rate"]-quality["empirical_success_rate"]) + abs(x["avg_cost"]-quality["avg_cost"]))
        return {"ok":True,"error":None,"data":{"task_type":task_type,"cheap":cheap,"quality":quality,"balanced":balanced},"sample_count":result.get("sample_count",0)}

    def failure_analysis(self, task_type: str) -> dict[str,Any]:
        memory=_post(self.memory_service_url,"/memory/search",{"query":f"{task_type} inconclusive failed","limit":50}).get("results",[])
        failures=[r for r in memory if "failed" in str(r.get("content","")).lower() or "inconclusive" in str(r.get("content","")).lower()]
        edges=_post(self.graph_service_url,"/graph/query",{"type":"edges","from_type":"model","to_type":"task_type","relationship":"failed_on"}).get("edges",[])
        affected=sorted({str(e.get("from_id","")).removeprefix("model:") for e in edges if e.get("to_id")==f"task:{task_type}"})
        return {"ok":True,"error":None,"data":{"task_type":task_type,"failure_count":len(failures),"affected_models":affected,"empirical_failure_count":len(failures),"sample_count":len(failures)},"sample_count":len(failures)}

    def scope_cost_analysis(self) -> dict[str,Any]:
        edges=_post(self.graph_service_url,"/graph/query",{"type":"edges","from_type":"scope","to_type":"task_type"}).get("edges",[])
        costs=defaultdict(list)
        for e in edges:
            cost=e.get("properties",{}).get("cost")
            if isinstance(cost,(int,float)): costs[str(e.get("from_id","")).removeprefix("scope:")].append(float(cost))
        data={scope:{"avg_cost":sum(v)/len(v),"min":min(v),"max":max(v),"sample_count":len(v)} for scope,v in costs.items() if v}
        return {"ok":True,"error":None,"data":data,"sample_count":sum(x["sample_count"] for x in data.values())}
