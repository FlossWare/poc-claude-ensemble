#!/usr/bin/env python3
"""
Thompson Router GA Evaluator

Evaluates Thompson sampling parameters:
- alpha_prior (0.5-3.0): Beta distribution alpha (success belief)
- beta_prior (0.5-3.0): Beta distribution beta (failure belief)
- cost_weight (0.1-0.5): Weight given to cost vs quality

Fitness = cost_savings % * quality_maintained (target: >0.90)

Simulates routing of 100 synthetic tasks with:
- Task types: code_review, deployment, documentation, testing, debugging
- Model costs: cheap (0.001/1M), mid (0.01/1M), expensive (0.1/1M)
- Quality distribution: Normal(0.85, 0.1) per model

Thompson learns which models are best for each task type.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.stats import beta as scipy_beta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task categories"""
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"


@dataclass
class Model:
    """Model with cost and quality characteristics"""
    name: str
    cost_per_mtok: float  # $ per million tokens
    base_quality: float  # Expected quality (0-1)
    cost_efficiency: float  # How much cost affects quality (0-1)


class ThompsonEvaluator:
    """Evaluate Thompson routing parameters"""

    # Model fleet with realistic costs
    MODELS = [
        Model("haiku", 0.0008, 0.80, 0.8),  # Cheap, okay quality
        Model("sonnet", 0.003, 0.88, 0.9),  # Balanced
        Model("opus", 0.015, 0.95, 0.95),  # Expensive, best quality
        Model("gemini", 0.002, 0.82, 0.85),  # Cheap alternative
    ]

    # Task type to optimal model mapping
    TASK_OPTIMAL = {
        TaskType.CODE_REVIEW: "opus",
        TaskType.DEPLOYMENT: "sonnet",
        TaskType.DOCUMENTATION: "haiku",
        TaskType.TESTING: "sonnet",
        TaskType.DEBUGGING: "opus",
    }

    def __init__(self):
        self.models = {m.name: m for m in self.MODELS}

    def _generate_synthetic_tasks(self, n_tasks: int = 100) -> List[Tuple[TaskType, float]]:
        """Generate synthetic task distribution"""
        tasks = []
        task_types = list(TaskType)

        # Realistic distribution
        distribution = {
            TaskType.CODE_REVIEW: 0.2,
            TaskType.DEPLOYMENT: 0.15,
            TaskType.DOCUMENTATION: 0.25,
            TaskType.TESTING: 0.25,
            TaskType.DEBUGGING: 0.15,
        }

        for task_type in task_types:
            n = int(n_tasks * distribution[task_type])
            for _ in range(n):
                tasks.append(task_type)

        # Shuffle
        np.random.shuffle(tasks)
        return tasks[:n_tasks]

    def _get_model_quality(self, model_name: str, task_type: TaskType) -> float:
        """Get quality score for model on task type"""
        model = self.models[model_name]

        # Base quality with task-specific variation
        base = model.base_quality

        # Bonus for optimal model
        if self.TASK_OPTIMAL[task_type] == model_name:
            base += 0.05

        # Add noise
        noise = np.random.normal(0, 0.05)
        quality = np.clip(base + noise, 0.5, 1.0)

        return quality

    def _thompson_sample(self, alpha: float, beta: float) -> float:
        """Sample from Beta distribution (Thompson sampling)"""
        return scipy_beta.rvs(alpha, beta)

    def _simulate_routing(
        self,
        alpha_prior: float,
        beta_prior: float,
        cost_weight: float,
        n_tasks: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Simulate routing with Thompson sampling.

        Returns: (total_cost, avg_quality, routing_accuracy)
        """
        tasks = self._generate_synthetic_tasks(n_tasks)

        # Initialize Thompson state per model
        thompson_state = {model: {'alpha': alpha_prior, 'beta': beta_prior}
                         for model in self.models.keys()}

        total_cost = 0.0
        total_quality = 0.0
        correct_routing = 0

        for task_type in tasks:
            # Thompson sample for each model
            scores = {}
            for model_name, state in thompson_state.items():
                model = self.models[model_name]

                # Sample quality belief from Beta
                quality_belief = self._thompson_sample(state['alpha'], state['beta'])

                # Cost factor (lower is better)
                cost_factor = 1.0 - (cost_weight * (model.cost_per_mtok / 0.015))

                # Combined score
                scores[model_name] = quality_belief * cost_factor

            # Select best model
            selected_model = max(scores, key=scores.get)

            # Get actual quality for this selection
            actual_quality = self._get_model_quality(selected_model, task_type)
            cost = self.models[selected_model].cost_per_mtok

            # Check if selection was optimal
            optimal_model = self.TASK_OPTIMAL[task_type]
            if selected_model == optimal_model:
                correct_routing += 1

            # Update Thompson state
            quality_threshold = 0.85
            if actual_quality >= quality_threshold:
                thompson_state[selected_model]['alpha'] += 1
            else:
                thompson_state[selected_model]['beta'] += 1

            total_cost += cost
            total_quality += actual_quality

        avg_quality = total_quality / n_tasks
        routing_accuracy = correct_routing / n_tasks
        avg_cost = total_cost / n_tasks

        return avg_cost, avg_quality, routing_accuracy

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate Thompson routing parameters.

        Fitness = cost_savings * quality_maintained
        """
        alpha_prior = parameters['alpha_prior']
        beta_prior = parameters['beta_prior']
        cost_weight = parameters['cost_weight']

        # Run multiple simulations for stability
        n_simulations = 3
        total_fitness = 0.0

        for sim in range(n_simulations):
            avg_cost, avg_quality, routing_accuracy = self._simulate_routing(
                alpha_prior, beta_prior, cost_weight, n_tasks=100
            )

            # Baseline: routing without Thompson (average cost)
            baseline_cost = np.mean([m.cost_per_mtok for m in self.MODELS])

            # Cost savings percentage
            cost_savings = max(0, (baseline_cost - avg_cost) / baseline_cost)

            # Quality maintained (target 0.85+)
            quality_maintained = max(0, (avg_quality - 0.80) / 0.15)

            # Routing accuracy bonus
            accuracy_bonus = routing_accuracy * 0.2

            # Fitness = cost_savings * quality_maintained + accuracy bonus
            fitness = (cost_savings * 0.5 + quality_maintained * 0.5 + accuracy_bonus)

            total_fitness += fitness

        avg_fitness = total_fitness / n_simulations

        logger.debug(
            f"Thompson: alpha={alpha_prior:.2f}, beta={beta_prior:.2f}, cost_weight={cost_weight:.2f} -> "
            f"fitness={avg_fitness:.6f}"
        )

        return np.clip(avg_fitness, 0.0, 1.0)


if __name__ == '__main__':
    # Test evaluator
    evaluator = ThompsonEvaluator()

    test_params = {
        'alpha_prior': 1.5,
        'beta_prior': 1.5,
        'cost_weight': 0.3,
    }

    fitness = evaluator.evaluate(test_params)
    print(f"Test fitness: {fitness:.6f}")
