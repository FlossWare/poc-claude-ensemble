#!/usr/bin/env python3
"""
Autonomous Learning Dashboard

Shows how Thompson is improving over time:
- Learning outcomes (what did each task teach?)
- Prior updates (Bayesian belief changes)
- Capability matrix evolution (model scores changing)
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.autonomous_learning import AutonomousLearningSystem


class AutonomousLearningDashboard:
    """Dashboard for autonomous learning progress"""

    def __init__(self):
        try:
            self.system = AutonomousLearningSystem()
            self.outcomes_dir = Path(__file__).parent.parent / 'learning' / 'autonomous_outcomes'
            self.priors_dir = Path(__file__).parent.parent / 'learning' / 'autonomous_priors'
        except Exception as e:
            print(f"Warning: Could not initialize learning system: {e}")
            self.outcomes_dir = None
            self.priors_dir = None

    def count_outcomes_by_type(self) -> Dict:
        """Count learning outcomes by task type"""
        if not self.outcomes_dir or not self.outcomes_dir.exists():
            return {}

        by_type = defaultdict(int)
        for outcome_file in self.outcomes_dir.glob('*.json'):
            task_type = outcome_file.stem.rsplit('_', 1)[0]
            by_type[task_type] += 1

        return dict(by_type)

    def count_priors_by_model(self) -> Dict:
        """Count updated priors by model"""
        if not self.priors_dir or not self.priors_dir.exists():
            return {}

        by_model = defaultdict(int)
        for prior_file in self.priors_dir.glob('*.json'):
            # Filename format: {model}_{task_type}.json
            parts = prior_file.stem.split('_', 1)
            if len(parts) >= 1:
                model = parts[0]
                by_model[model] += 1

        return dict(by_model)

    def get_learning_summary(self) -> Dict:
        """Overall learning system status"""
        outcomes = self.count_outcomes_by_type()
        priors = self.count_priors_by_model()

        total_outcomes = sum(outcomes.values())
        total_models_learning = len(priors)

        return {
            'total_task_outcomes': total_outcomes,
            'outcomes_by_task': outcomes,
            'models_with_updated_priors': total_models_learning,
            'priors_by_model': priors
        }

    def get_learning_progress(self) -> Dict:
        """Track learning cycle progress"""
        if not self.outcomes_dir or not self.outcomes_dir.exists():
            return {}

        outcome_files = list(self.outcomes_dir.glob('*.json'))
        prior_files = list(self.priors_dir.glob('*.json')) if self.priors_dir and self.priors_dir.exists() else []

        # Load most recent outcome
        latest_outcome = None
        if outcome_files:
            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_file) as f:
                    latest_outcome = json.load(f)
            except:
                pass

        return {
            'total_outcomes_collected': len(outcome_files),
            'total_priors_updated': len(prior_files),
            'latest_outcome': latest_outcome
        }

    def print_report(self):
        """Print formatted dashboard"""
        print("\n" + "="*80)
        print("  AUTONOMOUS LEARNING DASHBOARD")
        print("="*80)

        summary = self.get_learning_summary()
        if summary:
            print(f"\nLEARNING SUMMARY")
            print("-" * 80)
            print(f"Task Outcomes Collected: {summary['total_task_outcomes']}")
            if summary['outcomes_by_task']:
                print("  By Task Type:")
                for task_type in sorted(summary['outcomes_by_task'].keys()):
                    count = summary['outcomes_by_task'][task_type]
                    print(f"    {task_type:25s}: {count:3d} outcomes")

            print(f"\nModels Learning: {summary['models_with_updated_priors']}")
            if summary['priors_by_model']:
                print("  Bayesian Priors Updated:")
                for model in sorted(summary['priors_by_model'].keys()):
                    count = summary['priors_by_model'][model]
                    print(f"    {model:25s}: {count:3d} priors")

        progress = self.get_learning_progress()
        if progress:
            print(f"\nLEARNING PROGRESS")
            print("-" * 80)
            print(f"Total Outcomes Collected: {progress['total_outcomes_collected']}")
            print(f"Total Priors Updated: {progress['total_priors_updated']}")

            if progress['latest_outcome']:
                outcome = progress['latest_outcome']
                print(f"\nLatest Outcome:")
                print(f"  Task: {outcome.get('task_type')} (ID: {outcome.get('task_id')})")
                print(f"  Thompson Selected: {outcome.get('thompson_selected')}")
                print(f"  Actual Best: {outcome.get('actual_best_model')}")
                print(f"  Quality Achieved: {outcome.get('quality_score', 0):.2f}")
                print(f"  Cost: ${outcome.get('cost', 0):.4f}")

        print("\n" + "="*80 + "\n")
        print("ℹ️  Autonomous learning happens after every real task.")
        print("    Thompson improves automatically from actual performance data.")
        print("="*80 + "\n")


if __name__ == '__main__':
    dashboard = AutonomousLearningDashboard()
    dashboard.print_report()
