#!/usr/bin/env python3
"""
Example: Integrating cost tracking into orchestrate_smart.py

Shows how to:
1. Instrument Thompson Sampling router with cost tracking
2. Track routing decisions (Thompson, GA, fallback)
3. Capture complexity analysis and model selection
4. Log all decisions to disk for later analysis

Usage:
    python example_orchestrate_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cost_tracking.integration import (
    CostLogger,
    ThompsonRouterHook,
    RoutingDecision,
    ThompsonDecision,
)


class InstrumentedSmartOrchestrator:
    """
    Wrapper around SmartOrchestrator that instruments Thompson routing with cost tracking

    This demonstrates the minimal changes needed to add cost tracking:
    - Create CostLogger instance
    - Wrap the select_model method with ThompsonRouterHook
    - That's it! No changes to business logic
    """

    def __init__(self, orchestrator_instance, cost_logger: CostLogger):
        """
        Args:
            orchestrator_instance: SmartOrchestrator instance from orchestrate_smart.py
            cost_logger: CostLogger for tracking
        """
        self.orchestrator = orchestrator_instance
        self.logger = cost_logger
        self.hook = ThompsonRouterHook()

        # Instrument the router
        original_select = self.orchestrator.select_model
        self.orchestrator.select_model = self.hook.instrument_select_model(self.logger, original_select)

    def orchestrate_task(self, task_description, workers=None, task_type='general_qa',
                        workflow_name='', max_tokens=4000, max_retries=2):
        """
        Delegates to original orchestrator with instrumentation
        All routing decisions are automatically logged
        """
        return self.orchestrator.orchestrate_task(
            task_description=task_description,
            workers=workers,
            task_type=task_type,
            workflow_name=workflow_name,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def get_cost_summary(self):
        """Get cost tracking summary"""
        return self.logger.get_summary()


# ============================================================================
# EXAMPLE: Using instrumented orchestrator
# ============================================================================

def example_basic_usage():
    """Example 1: Basic cost tracking with orchestrator"""
    print("Example 1: Basic Cost Tracking\n")

    # Initialize cost logger
    logger = CostLogger()

    # This would be your real SmartOrchestrator from orchestrate_smart.py
    # For now, we'll show the pattern
    # from orchestrate_smart import SmartOrchestrator
    # orchestrator = SmartOrchestrator()

    # Wrap with cost tracking
    # instrumented = InstrumentedSmartOrchestrator(orchestrator, logger)

    # Use normally - all routing decisions are tracked
    # result = instrumented.orchestrate_task(
    #     task_description="Implement a keyset pagination algorithm",
    #     task_type="implementation",
    #     workflow_name="smart_orchestrator",
    # )

    # Get summary
    summary = logger.get_summary()
    print("Cost tracking summary:")
    print(f"  Total routing decisions: {summary['total_calls']}")
    print(f"  Routing methods used: {summary['routing_decisions']}")
    print(f"  Models selected: {summary['model_selection']}")
    print()

    logger.stop()


def example_track_specific_task():
    """Example 2: Track a specific task with multiple routing attempts"""
    print("Example 2: Track Specific Task\n")

    logger = CostLogger()

    # Manually log routing decisions for demonstration
    from cost_tracking.integration import ThompsonDecision, RoutingDecision

    # Simulate task routing workflow
    task_description = "Review CPSEARCH-10981 keyset pagination implementation"
    complexity_info = {
        'complexity_category': 'COMPLEX',
        'predicted_confidence': 0.75,
    }

    # Simulate Thompson Sampling decision
    routing = ThompsonDecision(
        routing_decision=RoutingDecision.THOMPSON_SAMPLING,
        selected_model='claude-opus-4',
        alternative_models=['claude-sonnet-4', 'claude-haiku-3'],
        ucb_scores=[0.92, 0.85, 0.68],
        confidence=0.92,
        complexity_category=complexity_info['complexity_category'],
        preferred_stronger_model=True,  # Upgraded due to complexity
        task_type='code_review',
        workflow_name='smart_orchestrator',
    )

    request_id = logger.log_routing_decision(routing, "req-cpsearch-10981")
    print(f"✓ Logged Thompson routing decision: {request_id}")
    print(f"  Selected model: {routing.selected_model}")
    print(f"  Confidence: {routing.confidence}")
    print(f"  Complexity: {routing.complexity_category}")
    print()

    # Get summary
    summary = logger.get_summary()
    print("Summary after one routing decision:")
    print(f"  Total calls: {summary['total_calls']}")
    print(f"  Models selected: {summary['model_selection']}")
    print()

    logger.stop()


def example_analyze_routing_patterns():
    """Example 3: Analyze routing patterns after multiple decisions"""
    print("Example 3: Analyze Routing Patterns\n")

    logger = CostLogger()

    from cost_tracking.integration import ThompsonDecision, RoutingDecision

    # Simulate multiple tasks
    tasks = [
        {
            'description': 'Simple bug fix',
            'complexity': 'SIMPLE',
            'decision': RoutingDecision.AUTO_PROFILER,
            'model': 'claude-haiku-3',
        },
        {
            'description': 'Complex architecture review',
            'complexity': 'VERY_COMPLEX',
            'decision': RoutingDecision.THOMPSON_SAMPLING_UPGRADED,
            'model': 'claude-opus-4',
        },
        {
            'description': 'Medium refactoring task',
            'complexity': 'MEDIUM',
            'decision': RoutingDecision.THOMPSON_SAMPLING,
            'model': 'claude-sonnet-4',
        },
    ]

    # Log each task's routing
    for i, task in enumerate(tasks):
        routing = ThompsonDecision(
            routing_decision=task['decision'],
            selected_model=task['model'],
            alternative_models=[],
            ucb_scores=[],
            confidence=0.85,
            complexity_category=task['complexity'],
            preferred_stronger_model='UPGRADED' in task['decision'].value.upper(),
            task_type='general_qa',
            workflow_name='smart_orchestrator',
        )
        logger.log_routing_decision(routing, f"task-{i}")
        print(f"✓ Task {i}: {task['description']}")
        print(f"  Complexity: {task['complexity']} → Model: {task['model']}")

    print()

    # Analyze patterns
    summary = logger.get_summary()
    print("Routing analysis:")
    print(f"  Total tasks: {summary['total_calls']}")
    print(f"  Routing strategies used:")
    for strategy, count in summary['routing_decisions'].items():
        print(f"    {strategy}: {count}")
    print(f"  Model distribution:")
    for model, count in summary['model_selection'].items():
        pct = (count / summary['total_calls']) * 100
        print(f"    {model}: {count} ({pct:.0f}%)")
    print()

    logger.stop()


def example_cost_analysis():
    """Example 4: Show cost savings from routing decisions"""
    print("Example 4: Cost Analysis\n")

    logger = CostLogger()

    from cost_tracking.integration import ThompsonDecision, RoutingDecision

    # Model pricing (per 1M tokens, example)
    model_pricing = {
        'claude-haiku-3': 0.80,      # cheapest
        'claude-sonnet-4': 3.00,     # medium
        'claude-opus-4': 15.00,      # expensive
    }

    # Simulate routing decisions with model costs
    decisions = [
        ('SIMPLE task', RoutingDecision.AUTO_PROFILER, 'claude-haiku-3', 200),
        ('COMPLEX task', RoutingDecision.THOMPSON_SAMPLING_UPGRADED, 'claude-opus-4', 800),
        ('MEDIUM task', RoutingDecision.THOMPSON_SAMPLING, 'claude-sonnet-4', 500),
    ]

    total_cost = 0
    for task, decision, model, tokens in decisions:
        routing = ThompsonDecision(
            routing_decision=decision,
            selected_model=model,
            alternative_models=[],
            ucb_scores=[],
            confidence=0.8,
            complexity_category='GENERIC',
            preferred_stronger_model=False,
            task_type='general_qa',
            workflow_name='smart_orchestrator',
        )
        logger.log_routing_decision(routing, f"cost-{task}")

        # Calculate cost
        model_cost = model_pricing[model]
        cost = (tokens / 1_000_000) * model_cost
        total_cost += cost

        print(f"✓ {task}")
        print(f"  Model: {model} (${model_cost:.2f}/M tokens)")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${cost:.6f}")

    print()
    print(f"Total cost for 3 tasks: ${total_cost:.6f}")
    print(f"Average per task: ${total_cost/3:.6f}")
    print()

    logger.stop()


def example_integration_pattern():
    """Example 5: Show the integration pattern for existing code"""
    print("Example 5: Integration Pattern\n")

    print("""
BEFORE (orchestrate_smart.py):
    class SmartOrchestrator:
        def select_model(self, task_description, task_type, workflow_name):
            # Thompson Sampling logic
            return model, routing_method


AFTER (add minimal wrapper):
    from cost_tracking.integration import CostLogger, ThompsonRouterHook

    # Initialize once at startup
    cost_logger = CostLogger()

    # Get your orchestrator
    orchestrator = SmartOrchestrator()

    # Instrument it (one line!)
    hook = ThompsonRouterHook()
    orchestrator.select_model = hook.instrument_select_model(
        cost_logger,
        orchestrator.select_model
    )

    # Use normally - all decisions are logged automatically
    model, method = orchestrator.select_model(task_description)

    # Whenever you want, get summary
    summary = cost_logger.get_summary()


KEY BENEFITS:
  ✓ Zero changes to business logic
  ✓ Single line to instrument (hook.instrument_select_model)
  ✓ All routing decisions automatically logged
  ✓ Thread-safe async logging (<1% overhead)
  ✓ Metrics saved to disk: ~/.claude/cost_tracking/
  ✓ Can analyze patterns later with cost_analysis.py

FILES CREATED:
  ~/.claude/cost_tracking/cost_metrics.json      # Summary metrics
  ~/.claude/cost_tracking/cost_summary.jsonl     # All decisions (one per line)
  ~/.claude/cost_tracking/cost_detailed.jsonl    # Detailed per-request info
  ~/.claude/cost_tracking/errors.log             # Any logging errors
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("Cost Tracking Integration Examples")
    print("=" * 70)
    print()

    example_basic_usage()
    example_track_specific_task()
    example_analyze_routing_patterns()
    example_cost_analysis()
    example_integration_pattern()

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run test_integration.py to verify functionality")
    print("  2. Integrate into orchestrate_smart.py (see Example 5)")
    print("  3. Check ~/.claude/cost_tracking/ for logged decisions")
    print("=" * 70)
