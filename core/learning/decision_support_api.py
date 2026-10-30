#!/usr/bin/env python3
"""
Decision Support REST API Handler

Exposes decision support modules (LearningAnalytics, ArbitrationAdvisor,
DiagnosticQueries) as REST endpoints via ensemble_server.py

REST Endpoints (via ensemble_server):
  POST /decision/recommend             → Full recommendation
  POST /decision/advisor/models        → Recommend models
  POST /decision/advisor/phases        → Recommend phases
  POST /decision/advisor/cost          → Estimate cost
  POST /decision/analytics/best-models → Best models for task
  POST /decision/analytics/tradeoff    → Cost-quality tradeoff
  POST /decision/analytics/failure     → Failure analysis
  GET  /decision/analytics/scope-costs → Scope cost analysis
  POST /decision/query/semantic        → Semantic search
  POST /decision/query/graph           → Graph traversal
  POST /decision/query/patterns        → Model patterns
  GET  /decision/query/problems        → Problematic tasks
  GET  /decision/query/outliers        → Cost outliers
  POST /decision/query/trend           → Trend analysis
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import decision support modules
try:
    from learning_analytics import LearningAnalytics
    from arbitration_advisor import ArbitrationAdvisor
    from diagnostic_queries import DiagnosticQueries
except ImportError as e:
    logger.error(f"Could not import decision support modules: {e}")
    MODULES_AVAILABLE = False
else:
    MODULES_AVAILABLE = True


class DecisionSupportAPI:
    """Handler for decision support REST endpoints"""

    def __init__(self, graph_service_url: Optional[str] = None, memory_service_url: Optional[str] = None):
        if not MODULES_AVAILABLE:
            logger.warning("Decision support modules not available")
            return

        self.analytics = LearningAnalytics(graph_service_url, memory_service_url)
        self.advisor = ArbitrationAdvisor(self.analytics)
        self.diagnostics = DiagnosticQueries(graph_service_url, memory_service_url)

    def _add_metadata(self, sample_count: int = 0, days_old: int = 0) -> Dict[str, Any]:
        """Generate metadata fields for API responses.

        Args:
            sample_count: Number of outcomes/samples used in analysis
            days_old: Age of most recent data point in days

        Returns:
            Dictionary with sample_count, data_freshness, and warnings
        """
        warnings: List[str] = []

        # Check for no historical data
        if sample_count == 0:
            warnings.append("ERROR: No historical data available - using defaults")
        # Check for very few samples
        elif sample_count < 5:
            warnings.append(f"WARNING: Very few samples (N={sample_count}) - recommendations unreliable")
        # Check for limited samples
        elif sample_count < 30:
            warnings.append(f"WARNING: Limited samples (N={sample_count}) - high variance expected")

        # Check for stale data
        if days_old > 30:
            warnings.append(f"WARNING: Data is {days_old} days old - patterns may have changed")

        return {
            "sample_count": sample_count,
            "data_freshness": days_old,
            "warnings": warnings
        }

    # === FULL RECOMMENDATION ENDPOINT ===

    def handle_recommend(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/recommend

        Full recommendation: models, phases, cost, risk

        Args:
          task_type: 'code_review', 'security_audit', etc.
          scope: 'small', 'medium', 'large'
          budget: max cost allowed (optional)
        """
        try:
            task_type = request_data.get('task_type')
            scope = request_data.get('scope', 'medium')
            budget = request_data.get('budget')

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.advisor.full_recommendation(task_type, scope, budget)
            sample_count = result.get('sample_count', 0) or 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in recommend: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    # === ADVISOR ENDPOINTS ===

    def handle_recommend_models(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/advisor/models

        Recommend which models to use
        """
        try:
            task_type = request_data.get('task_type')
            scope = request_data.get('scope', 'medium')
            budget = request_data.get('budget')
            count = request_data.get('count', 3)

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.advisor.recommend_models(task_type, scope, budget, count)
            sample_count = result.get('sample_count', 0) or 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in recommend_models: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_recommend_phases(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/advisor/phases

        Recommend phase count (2 vs 3)
        """
        try:
            task_type = request_data.get('task_type')
            scope = request_data.get('scope', 'medium')
            budget = request_data.get('budget')

            if not task_type or not scope:
                return {
                    'ok': False,
                    'error': 'Missing task_type or scope',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.advisor.recommend_phases(task_type, scope, budget)
            sample_count = result.get('sample_count', 0) or 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in recommend_phases: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_estimate_cost(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/advisor/cost

        Estimate total cost for model combination
        """
        try:
            models = request_data.get('models', [])
            task_type = request_data.get('task_type')
            phases = request_data.get('phases', 2)

            if not models or not task_type:
                return {
                    'ok': False,
                    'error': 'Missing models or task_type',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.advisor.estimate_cost(models, task_type, phases)
            sample_count = result.get('sample_count', 0) or 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in estimate_cost: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    # === ANALYTICS ENDPOINTS ===

    def handle_best_models(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/analytics/best-models

        Get best models for task type
        """
        try:
            task_type = request_data.get('task_type')
            scope = request_data.get('scope')
            limit = request_data.get('limit', 5)

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'data': None
                }

            result = self.analytics.best_models_for(task_type, scope, limit)
            sample_count = result.get('sample_count', 0) if isinstance(result, dict) else len(result) if isinstance(result, list) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in best_models: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_cost_quality_tradeoff(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/analytics/tradeoff

        Analyze cost vs quality tradeoff
        """
        try:
            task_type = request_data.get('task_type')

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'data': None
                }

            result = self.analytics.cost_quality_tradeoff(task_type)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in cost_quality_tradeoff: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_failure_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/analytics/failure

        Analyze why tasks fail
        """
        try:
            task_type = request_data.get('task_type')

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'data': None
                }

            result = self.analytics.failure_analysis(task_type)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in failure_analysis: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_scope_costs(self) -> Dict[str, Any]:
        """GET /decision/analytics/scope-costs

        Get cost analysis by scope
        """
        try:
            result = self.analytics.scope_cost_analysis()
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in scope_costs: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    # === DIAGNOSTIC QUERY ENDPOINTS ===

    def handle_semantic_search(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/query/semantic

        Semantic search on Memory Service
        """
        try:
            query = request_data.get('query')
            limit = request_data.get('limit', 10)

            if not query:
                return {
                    'ok': False,
                    'error': 'Missing query',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.diagnostics.semantic_search(query, limit)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in semantic_search: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_graph_traversal(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/query/graph

        Traverse graph from node
        """
        try:
            start_node = request_data.get('start')
            max_depth = request_data.get('max_depth', 3)

            if not start_node:
                return {
                    'ok': False,
                    'error': 'Missing start node',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.diagnostics.graph_traversal(start_node, max_depth)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in graph_traversal: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_model_patterns(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/query/patterns

        Find patterns for specific model
        """
        try:
            model = request_data.get('model')

            if not model:
                return {
                    'ok': False,
                    'error': 'Missing model',
                    'error_code': 'invalid_request',
                    'data': None
                }

            result = self.diagnostics.find_model_patterns(model)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in model_patterns: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_problematic_tasks(self) -> Dict[str, Any]:
        """GET /decision/query/problems

        Find tasks with high failure rates
        """
        try:
            result = self.diagnostics.find_problematic_tasks()
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in problematic_tasks: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_cost_outliers(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """GET /decision/query/outliers

        Find unusually expensive runs
        """
        try:
            percentile = request_data.get('percentile', 0.9)
            result = self.diagnostics.cost_outliers(percentile)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in cost_outliers: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }

    def handle_trend(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /decision/query/trend

        Analyze trends over time
        """
        try:
            task_type = request_data.get('task_type')
            metric = request_data.get('metric', 'cost')

            if not task_type:
                return {
                    'ok': False,
                    'error': 'Missing task_type',
                    'data': None
                }

            result = self.diagnostics.trend_analysis(task_type, metric)
            sample_count = result.get('sample_count', 0) or 0 if isinstance(result, dict) else 0
            metadata = self._add_metadata(sample_count=sample_count, days_old=0)
            if isinstance(result, dict) and result.get('ok'):
                result['metadata'] = metadata
            return result

        except Exception as e:
            logger.error(f"Error in trend: {e}")
            return {
                'ok': False,
                'error': str(e),
                'error_code': 'dependency_unavailable' if isinstance(e, RuntimeError) and 'service unavailable' in str(e) else 'internal_error',
                'data': None
            }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    api = DecisionSupportAPI()

    # Example: full recommendation
    print("Full recommendation:")
    result = api.handle_recommend({
        'task_type': 'code_review',
        'scope': 'small',
        'budget': 0.50
    })
    print(json.dumps(result, indent=2, default=str))

    # Example: best models
    print("\nBest models for security_audit:")
    result = api.handle_best_models({'task_type': 'security_audit'})
    print(json.dumps(result, indent=2, default=str))

    # Example: cost-quality tradeoff
    print("\nCost-quality tradeoff:")
    result = api.handle_cost_quality_tradeoff({'task_type': 'code_review'})
    print(json.dumps(result, indent=2, default=str))
