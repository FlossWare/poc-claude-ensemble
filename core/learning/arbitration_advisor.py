"""Decision Support advisor built on observed learning data."""
from __future__ import annotations
from typing import Any
from learning_analytics import LearningAnalytics

class ArbitrationAdvisor:
    def __init__(self, analytics: LearningAnalytics|None=None):
        self.analytics=analytics or LearningAnalytics()

    def recommend_models(self, task_type: str, scope: str="medium", budget: float|None=None, count: int=3)->dict[str,Any]:
        result=self.analytics.best_models_for(task_type,scope,count+2)
        if not result.get("ok") or not result.get("data"):
            return {"ok":False,"error":f"No historical data for {task_type}","error_code":"no_data","data":None}
        rows=[dict(x) for x in result["data"] if x["avg_cost"] is not None and (budget is None or x["avg_cost"]<=budget)]
        if not rows: return {"ok":False,"error":f"No models available within {budget} budget","error_code":"no_data","data":None}
        for i,row in enumerate(rows[:count]):
            row["reason"]="highest observed success rate" if i==0 else "observed alternative"
        rows=rows[:count]
        return {"ok":True,"error":None,"data":{"task_type":task_type,"scope":scope,"budget":budget,"recommended":rows,"sample_count":sum(x["sample_count"] for x in rows)},"sample_count":sum(x["sample_count"] for x in rows)}

    def recommend_phases(self, task_type: str, scope: str, budget: float|None=None)->dict[str,Any]:
        recommendation="3" if scope=="large" else "2"
        costs=self.analytics.scope_cost_analysis()
        scope_data=costs.get("data",{}).get(scope,{})
        if budget is not None and scope_data.get("avg_cost",0)>budget: recommendation="2"
        failures=self.analytics.failure_analysis(task_type)
        if failures.get("data",{}).get("failure_count",0)>5: recommendation="3"
        return {"ok":True,"error":None,"data":{"task_type":task_type,"scope":scope,"recommendation":recommendation,"empirical_failure_count":failures.get("data",{}).get("failure_count",0),"estimated_costs":scope_data},"sample_count":failures.get("sample_count",0)}

    def estimate_cost(self, models:list[str], task_type:str, phases:int=2)->dict[str,Any]:
        result=self.analytics.best_models_for(task_type,limit=20)
        costs={x["model"]:x["avg_cost"] for x in result.get("data",[]) if x.get("avg_cost") is not None}
        if not models or any(model not in costs for model in models):
            return {"ok":False,"error":"No observed cost data for every requested model","error_code":"no_data","data":None}
        raw_arbiter_cost = __import__("os").environ.get("ENSEMBLE_ARBITER_COST")
        if raw_arbiter_cost is None:
            return {"ok":False,"error":"ENSEMBLE_ARBITER_COST is not configured","error_code":"no_data","data":None}
        try:
            arbiter_cost=float(raw_arbiter_cost)
        except ValueError:
            return {"ok":False,"error":"ENSEMBLE_ARBITER_COST must be numeric","error_code":"internal_error","data":None}
        worker=sum(costs[model] for model in models)
        total=worker+arbiter_cost
        return {"ok":True,"error":None,"data":{"task_type":task_type,"phases":phases,"workers":models,"worker_cost_per_phase":worker,"arbiter_cost_per_phase":arbiter_cost,"total_per_phase":total,"total_all_phases":total*phases,"sample_count":result.get("sample_count",0)},"sample_count":result.get("sample_count",0)}

    def full_recommendation(self, task_type:str, scope:str="medium", budget:float|None=None)->dict[str,Any]:
        models=self.recommend_models(task_type,scope,budget)
        phases=self.recommend_phases(task_type,scope,budget)
        if not models.get("ok") or not phases.get("ok"):
            return {"ok":False,"error":"Could not generate recommendations","error_code":models.get("error_code") or phases.get("error_code") or "internal_error","data":{"models_error":models.get("error"),"phases_error":phases.get("error")}}
        chosen=[x["model"] for x in models["data"]["recommended"]]
        cost=self.estimate_cost(chosen,task_type,int(phases["data"]["recommendation"]))
        if not cost.get("ok"):
            return {
                "ok":False,
                "error":cost.get("error") or "Could not estimate recommendation cost",
                "error_code":cost.get("error_code") or "internal_error",
                "data":{
                    "task_type":task_type,
                    "scope":scope,
                    "budget":budget,
                    "details":{"models":models,"phases":phases,"cost":cost},
                },
                "sample_count":models.get("sample_count",0),
            }
        return {"ok":True,"error":None,"data":{"task_type":task_type,"scope":scope,"budget":budget,"recommendation":{"models":chosen,"phases":int(phases["data"]["recommendation"]),"estimated_cost":cost["data"]["total_all_phases"]},"details":{"models":models,"phases":phases,"cost":cost}},"sample_count":models.get("sample_count",0)}

