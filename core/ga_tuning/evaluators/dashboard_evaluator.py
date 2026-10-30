#!/usr/bin/env python3
"""
Dashboard GA Evaluator

Evaluates dashboard/learning parameters:
- learning_rate (0.01-0.2): Speed of parameter updates
- exploration_decay (0.85-0.99): How fast exploration decays
- alert_threshold (0.3-0.9): When to alert on anomalies

Fitness = learning_speed * stability
- learning_speed: How fast quality improves (target 20%+ improvement)
- stability: How stable without overfitting (quality regression < 5%)

Simulates Thompson learning curve over 500 synthetic tasks.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import beta as scipy_beta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskMetric:
    """Metrics for a single task"""
    task_id: int
    quality: float
    cost: float
    selected_model: str
    optimal_model: str


class DashboardEvaluator:
    """Evaluate dashboard learning parameters"""

    MODELS = ['haiku', 'sonnet', 'opus']

    def __init__(self):
        pass

    def _generate_synthetic_tasks(
        self,
        n_tasks: int = 500,
        drift: bool = True,
    ) -> List[TaskMetric]:
        """
        Generate synthetic task stream with optional drift.

        Quality distribution changes over time (concept drift).
        """
        tasks = []

        # Initial model quality (Beta distribution parameters)
        model_params = {
            'haiku': {'alpha': 1.0, 'beta': 2.0},  # Initially poor
            'sonnet': {'alpha': 2.0, 'beta': 1.5},  # Balanced
            'opus': {'alpha': 3.0, 'beta': 1.0},  # Initially good
        }

        for task_id in range(n_tasks):
            # Simulate concept drift: quality shifts over time
            if drift:
                drift_factor = 1.0 + 0.3 * np.sin(task_id / 100)  # Oscillating drift
            else:
                drift_factor = 1.0

            # Sample quality from each model
            model_qualities = {}
            for model, params in model_params.items():
                alpha = params['alpha'] * drift_factor
                beta = params['beta'] / drift_factor
                quality = scipy_beta.rvs(alpha, beta)
                model_qualities[model] = quality

            # Optimal model (highest quality)
            optimal = max(model_qualities, key=model_qualities.get)
            optimal_quality = model_qualities[optimal]

            # Random selection (naive baseline)
            selected = np.random.choice(self.MODELS)
            selected_quality = model_qualities[selected]

            # Cost model
            cost_map = {'haiku': 0.1, 'sonnet': 1.0, 'opus': 10.0}

            task = TaskMetric(
                task_id=task_id,
                quality=selected_quality,
                cost=cost_map[selected],
                selected_model=selected,
                optimal_model=optimal,
            )

            tasks.append(task)

        return tasks

    def _simulate_thompson_learning(
        self,
        tasks: List[TaskMetric],
        learning_rate: float,
        exploration_decay: float,
        alert_threshold: float,
    ) -> Tuple[List[float], float, float]:
        """
        Simulate Thompson learning over task stream.

        Returns: (quality_history, final_accuracy, stability_score)
        """
        # Initialize Thompson state per model
        thompson_state = {
            model: {'alpha': 1.0, 'beta': 1.0, 'quality_sum': 0.0, 'count': 0}
            for model in self.MODELS
        }

        quality_history = []
        correct_selections = 0
        quality_regressions = 0
        max_quality_seen = 0
        window_sizes = [50, 100, 200]  # Check stability at different windows

        exploration_factor = 1.0

        for task_id, task in enumerate(tasks):
            # Decay exploration
            exploration_factor = exploration_decay ** (task_id / len(tasks))

            # Thompson sample for each model (with exploration)
            samples = {}
            for model, state in thompson_state.items():
                # Sample from posterior
                sample = scipy_beta.rvs(state['alpha'], state['beta'])
                # Mix with random exploration
                exploration_prob = np.random.random()
                if exploration_prob < (1.0 - exploration_factor):
                    sample = np.random.random()  # Random exploration
                samples[model] = sample

            # Select best model
            selected_model = max(samples, key=samples.get)

            # Get actual quality (from pre-generated task)
            actual_quality = task.quality
            is_correct = selected_model == task.optimal_model

            # Update Thompson state
            quality_threshold = 0.7
            success_weight = learning_rate

            if actual_quality >= quality_threshold:
                thompson_state[selected_model]['alpha'] += success_weight
            else:
                thompson_state[selected_model]['beta'] += success_weight

            # Track metrics
            if is_correct:
                correct_selections += 1

            quality_history.append(actual_quality)
            max_quality_seen = max(max_quality_seen, actual_quality)

            # Detect quality regression (moving average)
            if task_id > 50:
                recent_avg = np.mean(quality_history[-50:])
                if recent_avg < max_quality_seen * 0.95:
                    quality_regressions += 1

        # Calculate accuracy
        accuracy = correct_selections / len(tasks)

        # Stability score: inverse of regression frequency
        regression_rate = quality_regressions / (len(tasks) / 50) if len(tasks) > 50 else 0
        stability_score = 1.0 - min(1.0, regression_rate * 0.5)

        logger.debug(
            f"Thompson learning: accuracy={accuracy:.2%}, "
            f"regression_rate={regression_rate:.2%}, stability={stability_score:.2%}"
        )

        return quality_history, accuracy, stability_score

    def _calculate_learning_speed(self, quality_history: List[float]) -> float:
        """
        Calculate learning speed as improvement rate.

        Target: 20%+ improvement from start to end.
        """
        if len(quality_history) < 100:
            return 0.0

        first_window = np.mean(quality_history[:50])
        last_window = np.mean(quality_history[-50:])

        if first_window <= 0:
            return 0.0

        improvement = (last_window - first_window) / first_window
        return max(0, improvement)

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate dashboard learning parameters.

        Fitness = learning_speed * stability
        """
        learning_rate = parameters['learning_rate']
        exploration_decay = parameters['exploration_decay']
        alert_threshold = parameters['alert_threshold']

        # Generate task stream
        tasks = self._generate_synthetic_tasks(n_tasks=500, drift=True)

        # Simulate Thompson learning
        quality_history, accuracy, stability = self._simulate_thompson_learning(
            tasks, learning_rate, exploration_decay, alert_threshold
        )

        # Calculate learning speed
        learning_speed = self._calculate_learning_speed(quality_history)

        # Fitness components
        # 1. Learning speed (target 20%+)
        speed_component = min(1.0, max(0, learning_speed) / 0.2)

        # 2. Stability (target high, penalize regression)
        stability_component = stability

        # 3. Accuracy bonus
        accuracy_component = min(1.0, accuracy * 1.2)

        # Combined fitness
        fitness = (
            0.4 * speed_component +
            0.4 * stability_component +
            0.2 * accuracy_component
        )

        logger.debug(
            f"Dashboard: learning_rate={learning_rate:.3f}, "
            f"exploration_decay={exploration_decay:.3f}, alert={alert_threshold:.2f} -> "
            f"speed={learning_speed:.2%}, stability={stability:.2%}, fitness={fitness:.6f}"
        )

        return np.clip(fitness, 0.0, 1.0)


if __name__ == '__main__':
    # Test evaluator
    evaluator = DashboardEvaluator()

    test_params = {
        'learning_rate': 0.1,
        'exploration_decay': 0.95,
        'alert_threshold': 0.5,
    }

    fitness = evaluator.evaluate(test_params)
    print(f"Test fitness: {fitness:.6f}")
