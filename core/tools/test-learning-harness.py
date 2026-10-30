#!/usr/bin/env python3
"""
Test harness for feeding tasks through Thompson/Learning pipeline.

Usage:
  python3 tools/test-learning-harness.py --tasks 5 --seed 42

Records outcomes in the learning service so Thompson can learn which
models perform best for different task types.
"""

import sys
import json
import uuid
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "learning"))

from shared.thompson_client import ThompsonClient
from learning.learning_client import LearningClient


TASK_TYPES = [
    "code_review",
    "documentation",
    "testing",
    "architecture_design",
    "bug_analysis",
    "security_audit",
    "refactoring",
]

MODELS = ["haiku", "sonnet", "opus", "gemini", "gpt-4"]


class SimulatedTask:
    """Simulates a task with model selection and outcome."""

    def __init__(self, task_type: str, task_id: str = None):
        self.task_id = task_id or str(uuid.uuid4())[:8]
        self.task_type = task_type
        self.model = None
        self.tokens = 0
        self.cost = 0.0
        self.rating = 0

    def select_model(self, tc: ThompsonClient):
        """Use Thompson to select a model for this task."""
        self.model = tc.select_model(self.task_type)

    def execute(self):
        """Simulate task execution with random outcome."""
        # Simulate token usage (varies by model and task)
        base_tokens = random.randint(200, 1000)
        model_multiplier = {"haiku": 1.0, "sonnet": 1.5, "opus": 2.0, "gemini": 0.8, "gpt-4": 1.8}
        self.tokens = int(base_tokens * model_multiplier.get(self.model, 1.0))

        # Simulate cost based on tokens and model
        cost_per_1k = {
            "haiku": 0.003,
            "sonnet": 0.015,
            "opus": 0.060,
            "gemini": 0.001,  # Cheapest
            "gpt-4": 0.030,
        }
        self.cost = (self.tokens / 1000.0) * cost_per_1k.get(self.model, 0.01)

        # Simulate quality rating (haiku and gemini typically lower, opus/sonnet higher)
        quality_baseline = {
            "haiku": 3.0,
            "sonnet": 4.2,
            "opus": 4.7,
            "gemini": 3.5,
            "gpt-4": 4.5,
        }
        base_quality = quality_baseline.get(self.model, 3.5)
        self.rating = min(5.0, max(1.0, base_quality + random.gauss(0, 0.5)))

    def record(self, lc: LearningClient) -> bool:
        """Record outcome in learning service."""
        success = lc.process_outcome(
            task_id=self.task_id,
            task_type=self.task_type,
            model=self.model,
            rating=int(round(self.rating)),
            tokens=self.tokens,
            cost=self.cost
        )
        return success


def run_harness(num_tasks: int = 5, seed: int = None, verbose: bool = False):
    """Run learning harness with simulated tasks."""

    if seed is not None:
        random.seed(seed)

    tc = ThompsonClient()
    lc = LearningClient()

    print(f"\n{'='*60}")
    print(f"Claude Ensemble Learning Harness")
    print(f"{'='*60}")
    print(f"Simulating {num_tasks} tasks across {len(TASK_TYPES)} task types")
    print(f"Thompson will select from {len(MODELS)} available models")
    print(f"{'='*60}\n")

    results = []

    for i in range(num_tasks):
        task_type = random.choice(TASK_TYPES)
        task = SimulatedTask(task_type)

        # Step 1: Thompson selects model
        task.select_model(tc)

        # Step 2: Task executes (simulated)
        task.execute()

        # Step 3: Record outcome in Learning
        recorded = task.record(lc)

        result = {
            "task_id": task.task_id,
            "task_type": task_type,
            "model": task.model,
            "tokens": task.tokens,
            "cost": f"${task.cost:.4f}",
            "rating": f"{task.rating:.1f}/5",
            "recorded": recorded,
        }
        results.append(result)

        status = "✓" if recorded else "✗"
        print(f"[{i+1}/{num_tasks}] {status} {task.task_type:20s} → {task.model:8s} "
              f"(rating: {task.rating:.1f}, cost: ${task.cost:.4f})")

    # Get final report
    report = lc.get_report()

    print(f"\n{'='*60}")
    print(f"Learning Summary Report")
    print(f"{'='*60}")

    if report.get('ok'):
        print(f"Total outcomes recorded: {report.get('total_outcomes', 0)}")
        print(f"Average rating: {report.get('average_rating', 0):.2f}/5")

        # Model performance breakdown if available
        if 'by_model' in report:
            print(f"\nModel Performance:")
            for model, stats in sorted(report['by_model'].items()):
                print(f"  {model:10s}: {stats.get('outcomes', 0):2d} tasks, "
                      f"avg rating {stats.get('avg_rating', 0):.1f}")

        # Task type breakdown if available
        if 'by_task_type' in report:
            print(f"\nTask Type Performance:")
            for task_type, stats in sorted(report['by_task_type'].items()):
                print(f"  {task_type:20s}: {stats.get('outcomes', 0):2d} tasks, "
                      f"avg rating {stats.get('avg_rating', 0):.1f}")
    else:
        print(f"⚠ Learning service unavailable: {report.get('error', 'unknown')}")

    print(f"\n{'='*60}")
    print(f"Results saved to: learning/post_task_outcomes/")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run learning harness with simulated tasks"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=5,
        help="Number of tasks to simulate (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    results = run_harness(
        num_tasks=args.tasks,
        seed=args.seed,
        verbose=args.verbose
    )
