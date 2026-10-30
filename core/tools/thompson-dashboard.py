#!/usr/bin/env python3
"""
Thompson Router Dashboard

Shows routing decisions and accuracy over time.
Tracks: which models Thompson chooses, accuracy vs actual best, quality achieved.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.autonomous_learning import AutonomousLearningSystem


class ThompsonDashboard:
    """Dashboard for Thompson routing performance"""

    def __init__(self):
        try:
            self.system = AutonomousLearningSystem()
            self.outcomes = self.system.outcome_logger.outcomes
        except Exception as e:
            print(f"Warning: Could not initialize learning system: {e}")
            self.outcomes = {}

    def get_routing_accuracy(self) -> Dict:
        """Calculate Thompson routing accuracy"""
        if not self.outcomes:
            return {}

        correct = 0
        total = 0
        by_task_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        for outcome in self.outcomes.values():
            if outcome.was_thompson_correct() is not None:
                total += 1
                if outcome.was_thompson_correct():
                    correct += 1

                task_type = outcome.task_type
                by_task_type[task_type]['total'] += 1
                if outcome.was_thompson_correct():
                    by_task_type[task_type]['correct'] += 1

        overall_accuracy = correct / total if total > 0 else 0
        by_type = {
            task: {
                'accuracy': v['correct'] / v['total'],
                'decisions': v['total']
            }
            for task, v in by_task_type.items()
        }

        return {
            'overall_accuracy': overall_accuracy,
            'correct_decisions': correct,
            'total_decisions': total,
            'by_task_type': by_type
        }

    def get_model_rankings(self) -> Dict:
        """How well did Thompson rank its chosen model?"""
        if not self.outcomes:
            return {}

        rankings = defaultdict(list)
        for outcome in self.outcomes.values():
            rank = outcome.thompson_ranking()
            rankings[outcome.thompson_selected].append(rank)

        result = {}
        for model, ranks in rankings.items():
            avg_rank = sum(ranks) / len(ranks)
            result[model] = {
                'decisions': len(ranks),
                'avg_rank': avg_rank,
                'best_rank': min(ranks),
                'worst_rank': max(ranks)
            }

        return result

    def get_quality_by_model(self) -> Dict:
        """Average quality achieved by each model"""
        if not self.outcomes:
            return {}

        by_model = defaultdict(list)
        for outcome in self.outcomes.values():
            model = outcome.actual_model_used or outcome.thompson_selected
            by_model[model].append(outcome.quality_score)

        result = {}
        for model, scores in by_model.items():
            avg_quality = sum(scores) / len(scores)
            result[model] = {
                'tasks': len(scores),
                'avg_quality': avg_quality,
                'min_quality': min(scores),
                'max_quality': max(scores)
            }

        return result

    def get_cost_by_model(self) -> Dict:
        """Total cost by model"""
        if not self.outcomes:
            return {}

        by_model = defaultdict(float)
        for outcome in self.outcomes.values():
            model = outcome.actual_model_used or outcome.thompson_selected
            by_model[model] += outcome.cost

        return dict(by_model)

    def print_report(self):
        """Print formatted dashboard"""
        print("\n" + "="*80)
        print("  THOMPSON ROUTER DASHBOARD")
        print("="*80)

        # Accuracy
        accuracy = self.get_routing_accuracy()
        if accuracy:
            print(f"\nROUTING ACCURACY")
            print("-" * 80)
            print(f"Overall: {accuracy['overall_accuracy']:.1%} ({accuracy['correct_decisions']}/{accuracy['total_decisions']} correct)")
            print("\nBy Task Type:")
            for task_type, metrics in sorted(accuracy.get('by_task_type', {}).items()):
                print(f"  {task_type:20s}: {metrics['accuracy']:.1%} ({metrics['decisions']} decisions)")

        # Model Rankings
        rankings = self.get_model_rankings()
        if rankings:
            print(f"\nMODEL RANKING PERFORMANCE")
            print("-" * 80)
            print("How well did Thompson rank its choices? (1 = best, 2 = 2nd, etc.)")
            for model in sorted(rankings.keys()):
                m = rankings[model]
                print(f"  {model:30s}: avg rank {m['avg_rank']:.2f} ({m['best_rank']}-{m['worst_rank']}) [{m['decisions']} decisions]")

        # Quality
        quality = self.get_quality_by_model()
        if quality:
            print(f"\nQUALITY BY MODEL")
            print("-" * 80)
            for model in sorted(quality.keys()):
                m = quality[model]
                print(f"  {model:30s}: {m['avg_quality']:.2f} avg ({m['min_quality']:.2f}-{m['max_quality']:.2f}) [{m['tasks']} tasks]")

        # Cost
        cost = self.get_cost_by_model()
        if cost:
            print(f"\nCOST BY MODEL")
            print("-" * 80)
            total_cost = sum(cost.values())
            for model in sorted(cost.keys(), key=lambda m: cost[m], reverse=True):
                pct = 100 * cost[model] / total_cost if total_cost > 0 else 0
                print(f"  {model:30s}: ${cost[model]:8.4f} ({pct:5.1f}%)")
            print(f"  {'TOTAL':30s}: ${total_cost:8.4f}")

        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    dashboard = ThompsonDashboard()
    dashboard.print_report()
