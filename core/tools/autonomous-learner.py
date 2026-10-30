#!/usr/bin/env python3
"""
Autonomous Learner Integration

Wraps learning.autonomous_learning.py to capture real task outcomes
and continuously improve Thompson routing based on actual performance.

Usage:
  from autonomous_learner import AutonomousLearner

  learner = AutonomousLearner()
  learner.record_task(
    task_id="code-review-001",
    task_type="code_review",
    thompson_selected="claude-sonnet-5",
    quality_score=0.92,
    cost=0.045,
    actual_best_model="claude-opus-5-5"  # if known post-hoc
  )
"""

import sys
from pathlib import Path

# Add parent to path for imports
toolkit_root = Path(__file__).parent.parent
sys.path.insert(0, str(toolkit_root))

from learning.autonomous_learning import AutonomousLearningSystem
import logging

logger = logging.getLogger(__name__)


class AutonomousLearner:
    """Integration wrapper for autonomous learning system"""

    def __init__(self):
        """Initialize the autonomous learning system"""
        try:
            self.system = AutonomousLearningSystem()
            self.enabled = True
            logger.info("✓ Autonomous learning system initialized")
        except Exception as e:
            logger.warning(f"Could not initialize autonomous learning: {e}")
            self.enabled = False
            self.system = None

    def record_task(self, task_id: str, task_type: str, thompson_selected: str,
                   quality_score: float, cost: float, latency_ms: float = 0.0,
                   actual_best_model: str = None, alternatives_tested: dict = None,
                   notes: str = ""):
        """
        Record a task outcome for learning

        Args:
            task_id: Unique task identifier
            task_type: code_review, bug_analysis, security_audit, etc.
            thompson_selected: Model Thompson chose
            quality_score: Result quality (0-1)
            cost: API cost in dollars
            latency_ms: Execution time in milliseconds
            actual_best_model: Ground truth best model (if available)
            alternatives_tested: {model_name: quality_score} for alternatives
            notes: Additional metadata
        """
        if not self.enabled or not self.system:
            return

        try:
            self.system.log_outcome(
                task_id=task_id,
                task_type=task_type,
                thompson_selected=thompson_selected,
                thompson_candidates=[thompson_selected],  # Would need to track all candidates
                quality_score=quality_score,
                latency_ms=latency_ms,
                cost=cost,
                actual_best_model=actual_best_model,
                alternatives_tested=alternatives_tested or {},
                notes=notes
            )
            logger.debug(f"Logged outcome: {task_id}")
        except Exception as e:
            logger.error(f"Failed to log outcome: {e}")

    def get_routing_accuracy(self, task_type: str = None) -> dict:
        """Get Thompson's routing accuracy metrics"""
        if not self.enabled or not self.system:
            return {}

        try:
            return self.system.get_accuracy_metrics(task_type)
        except Exception as e:
            logger.error(f"Failed to get accuracy metrics: {e}")
            return {}

    def trigger_learning(self) -> dict:
        """Trigger learning from accumulated outcomes"""
        if not self.enabled or not self.system:
            return {}

        try:
            results = self.system.run_learning_cycle()
            logger.info(f"Learning cycle complete: {results}")
            return results
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            return {}

    def export_improved_parameters(self) -> dict:
        """Export learned parameters for Thompson router"""
        if not self.enabled or not self.system:
            return {}

        try:
            return self.system.export_learned_routing_params()
        except Exception as e:
            logger.error(f"Failed to export parameters: {e}")
            return {}


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)
    learner = AutonomousLearner()

    if learner.enabled:
        print("✓ Autonomous learner ready")
        metrics = learner.get_routing_accuracy()
        print(f"Accuracy metrics: {metrics}")
    else:
        print("✗ Autonomous learner not available")
