#!/usr/bin/env python3
"""
Post-Task Analyzer — Evaluate outcomes, get consensus, update Thompson priors

After each task:
1. Determine system confidence (know this task type well?)
2. If low confidence: ask user to rate
3. If medium: get consensus model evaluation (skip user)
4. If high: auto-rate from consensus (user can override)
5. Compare user vs consensus rating
6. Update Thompson priors for the model on this task type
7. Trigger alerts if quality/cost anomalies detected
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.thompson_router import StateTracker
from arbitration.api_client import MultiModelClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


@dataclass
class TaskOutcome:
    """Single task outcome for learning"""
    task_id: str
    task_type: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost: float
    user_rating: Optional[int]  # 1-5 or None
    consensus_rating: Optional[int]  # 1-5 or None
    system_confidence: float  # 0-1
    was_auto_rated: bool
    timestamp: str


class PostTaskAnalyzer:
    """Analyze task outcomes and update learning"""

    def __init__(self, repo_root: Path = None):
        if repo_root is None:
            repo_root = Path(__file__).parent.parent
        self.repo_root = repo_root
        self.thompson = StateTracker()
        self.api_client = MultiModelClient()
        self.outcomes_dir = repo_root / "learning" / "post_task_outcomes"
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)

        # Cache for consensus ratings (task_hash -> rating)
        self.consensus_cache = {}
        self._load_consensus_cache()

    def _load_consensus_cache(self):
        """Load cached consensus ratings from disk"""
        cache_file = self.outcomes_dir / "consensus_cache.json"
        if cache_file.exists():
            try:
                self.consensus_cache = json.loads(cache_file.read_text())
                logger.info(f"Loaded {len(self.consensus_cache)} cached consensus ratings")
            except Exception as e:
                logger.warning(f"Failed to load consensus cache: {e}")
                self.consensus_cache = {}

    def _save_consensus_cache(self):
        """Save consensus ratings cache to disk"""
        cache_file = self.outcomes_dir / "consensus_cache.json"
        try:
            cache_file.write_text(json.dumps(self.consensus_cache, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save consensus cache: {e}")

    def _get_cache_key(self, task_type: str, model_used: str, tokens: int, cost: float) -> str:
        """Generate cache key for a task evaluation"""
        data = f"{task_type}:{model_used}:{tokens}:{cost:.6f}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _call_consensus_model_with_timeout(
        self, consensus_model: str, prompt: str, timeout_seconds: int = 30
    ) -> Optional[str]:
        """Call consensus model with timeout and retry logic"""
        import signal
        import time

        class TimeoutException(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutException(f"API call timed out after {timeout_seconds}s")

        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            response = self.api_client.call_model(
                consensus_model,
                prompt,
                system="You are a fair evaluator of task outcomes.",
                temperature=0.5,
                max_tokens=10
            )
            signal.alarm(0)  # Cancel the alarm
            return response
        except TimeoutException as e:
            logger.warning(f"Consensus model call timed out: {e}")
            return None
        except Exception as e:
            logger.warning(f"Consensus model call failed: {e}")
            # Attempt retry once on network error
            if "network" in str(e).lower() or "connection" in str(e).lower():
                try:
                    logger.info("Retrying consensus model call...")
                    time.sleep(1)
                    response = self.api_client.call_model(
                        consensus_model,
                        prompt,
                        system="You are a fair evaluator of task outcomes.",
                        temperature=0.5,
                        max_tokens=10
                    )
                    signal.alarm(0)
                    return response
                except Exception as retry_e:
                    logger.warning(f"Retry failed: {retry_e}")
                    return None
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _parse_rating_response(self, response: str) -> Optional[int]:
        """Extract rating (1-5) from API response"""
        if not response:
            return None

        # Clean the response: remove whitespace and take first character
        cleaned = response.strip()

        # Try to extract first digit
        for char in cleaned:
            if char.isdigit():
                rating = int(char)
                if 1 <= rating <= 5:
                    return rating

        logger.warning(f"Could not parse rating from response: {response}")
        return None

    def calculate_system_confidence(self, task_type: str, model: str) -> float:
        """Determine how confident we are in this model for this task (0-1)

        Based on:
        - How many prior tasks of this type?
        - How consistent were the results?
        - How recent were they?
        """
        # Simple heuristic: check Thompson state
        perf = self.thompson.models.get(model)
        if not perf:
            return 0.0  # No prior data

        if perf.calls < 3:
            return 0.2  # Bootstrap phase
        elif perf.calls < 10:
            return 0.5  # Building confidence
        elif perf.calls < 30:
            return 0.75  # Pretty confident
        else:
            return 0.9  # Very confident

    def should_ask_user(self, task_type: str, model: str) -> bool:
        """Determine if we should ask user to rate this outcome"""
        confidence = self.calculate_system_confidence(task_type, model)

        # Bootstrap phase: always ask
        if confidence < 0.3:
            return True

        # Building confidence: ask occasionally (~30%)
        if confidence < 0.7:
            return (hash(f"{task_type}{model}") % 10) < 3

        # High confidence: skip asking (user can override)
        return False

    def get_consensus_rating(
        self, task_type: str, model_used: str, tokens: int, cost: float
    ) -> Tuple[int, Optional[str]]:
        """Get consensus model evaluation of the task outcome

        Uses a DIFFERENT model family than what was used:
        - If used Haiku (cheap): ask Sonnet (balanced)
        - If used Sonnet (balanced): ask Opus (capable)
        - If used Opus (capable): ask Sonnet (check)

        Implements:
        - Caching: Don't re-evaluate same task twice
        - Timeout: 30 second timeout with fallback to heuristic
        - Retry: One retry on network errors
        - Fallback: Use heuristic if API fails
        """

        # Check cache first
        cache_key = self._get_cache_key(task_type, model_used, tokens, cost)
        if cache_key in self.consensus_cache:
            logger.info(f"Using cached consensus rating for {task_type}")
            return self.consensus_cache[cache_key], None

        # Model rotation for diversity
        model_sequence = {
            "haiku": "claude-sonnet-4.5-20250514",
            "sonnet": "claude-opus-4.1-20250805",
            "opus": "claude-sonnet-4.5-20250514",
            "cursor": "claude-opus-4.1-20250805",
            "gemini": "claude-haiku-4-5@20251001",
        }

        consensus_model = model_sequence.get(model_used, "claude-sonnet-4.5-20250514")

        logger.info(f"Getting consensus rating from {consensus_model}")

        prompt = f"""
You are reviewing a completed task.

Task Type: {task_type}
Original Model: {model_used}
Tokens Used: {tokens}
Cost: ${cost:.6f}

Rate this outcome (was it a good use of resources?):
1 = Poor (wrong/slow/expensive for the job)
2 = Below average
3 = Acceptable (did the job adequately)
4 = Good (solid quality/cost tradeoff)
5 = Excellent (fast/cheap/high quality)

Respond with ONLY a single number 1-5, no explanation.
"""

        try:
            # Call consensus model with timeout
            response = self._call_consensus_model_with_timeout(consensus_model, prompt, timeout_seconds=30)

            if response:
                # Parse the response
                rating = self._parse_rating_response(response)
                if rating is not None:
                    # Cache successful result
                    self.consensus_cache[cache_key] = rating
                    self._save_consensus_cache()
                    logger.info(f"Consensus rating from API: {rating}")
                    return rating, None
                else:
                    logger.warning(f"Invalid response from consensus model: {response}")
                    # Fall back to heuristic
                    rating = self._get_heuristic_rating(tokens, cost)
                    logger.warning(f"Using heuristic fallback: {rating}")
                    return rating, "invalid_api_response"
            else:
                # Timeout or network error - use heuristic
                rating = self._get_heuristic_rating(tokens, cost)
                logger.warning(f"API failed, using heuristic fallback: {rating}")
                return rating, "api_timeout_or_error"

        except Exception as e:
            # Unexpected error - fall back to heuristic
            logger.warning(f"Consensus eval failed unexpectedly: {e}")
            rating = self._get_heuristic_rating(tokens, cost)
            return rating, f"exception: {str(e)}"

    def _get_heuristic_rating(self, tokens: int, cost: float) -> int:
        """Fallback heuristic rating based on token count and cost"""
        if tokens > 5000:
            return 3  # High token usage
        elif cost > 0.1:
            return 2  # Expensive
        else:
            return 4  # Reasonable

    def compare_ratings(self, user_rating: Optional[int], consensus_rating: int) -> Dict:
        """Compare user vs consensus rating"""

        if user_rating is None:
            return {
                "comparison": "no_user_rating",
                "agreement": None,
                "final_rating": consensus_rating,
                "notes": "Used consensus rating only",
            }

        diff = abs(user_rating - consensus_rating)

        if diff == 0:
            agreement = "perfect"
        elif diff <= 1:
            agreement = "close"
        else:
            agreement = "divergent"

        # Final rating: average if divergent, otherwise user takes precedence
        if agreement == "perfect":
            final_rating = user_rating
        elif agreement == "close":
            final_rating = user_rating  # Trust user on close calls
        else:
            # Divergent: split difference
            final_rating = (user_rating + consensus_rating) // 2

        return {
            "comparison": agreement,
            "user_rating": user_rating,
            "consensus_rating": consensus_rating,
            "diff": diff,
            "final_rating": final_rating,
            "notes": f"User rated {user_rating}, consensus rated {consensus_rating}",
        }

    def update_thompson_priors(
        self, model: str, task_type: str, final_rating: int, tokens: int, cost: float
    ) -> Dict:
        """Update Thompson priors based on outcome

        Ratings map to success:
        - 1-2: Failure (this model is bad for this task)
        - 3: Neutral (no signal)
        - 4-5: Success (this model is good for this task)
        """

        perf = self.thompson.models.get(model)
        if not perf:
            logger.warning(f"Model {model} not in Thompson state")
            return {}

        # Interpret rating as success/failure
        if final_rating <= 2:
            # Failure
            perf.failures += 1
        elif final_rating >= 4:
            # Success
            perf.successes += 1
        # Rating 3: neutral, no change

        perf.calls += 1
        perf.total_cost += cost
        perf.last_updated = datetime.now().isoformat()

        self.thompson.save()

        return {
            "model": model,
            "task_type": task_type,
            "final_rating": final_rating,
            "new_success_rate": perf.quality_rate,
            "total_calls": perf.calls,
            "notes": f"Updated Thompson priors: {perf.successes} successes / {perf.calls} total",
        }

    def analyze(
        self,
        task_id: str,
        task_type: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        user_rating: Optional[int] = None,
    ) -> TaskOutcome:
        """Main analysis pipeline"""

        logger.info(f"Analyzing task {task_id}: {task_type} via {model_used}")

        tokens = input_tokens + output_tokens

        # Step 1: Determine confidence
        confidence = self.calculate_system_confidence(task_type, model_used)
        logger.info(f"System confidence: {confidence:.1%}")

        # Step 2: Get consensus rating
        consensus_rating, consensus_error = self.get_consensus_rating(
            task_type, model_used, tokens, cost
        )
        logger.info(f"Consensus rating: {consensus_rating}")

        # Step 3: Compare ratings
        comparison = self.compare_ratings(user_rating, consensus_rating)
        final_rating = comparison.get("final_rating")
        logger.info(f"Final rating: {final_rating} ({comparison['comparison']})")

        # Step 4: Update Thompson
        thompson_update = self.update_thompson_priors(
            model_used, task_type, final_rating, tokens, cost
        )
        logger.info(f"Thompson updated: {thompson_update.get('notes')}")

        # Step 5: Create outcome record
        was_auto_rated = user_rating is None

        outcome = TaskOutcome(
            task_id=task_id,
            task_type=task_type,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            user_rating=user_rating,
            consensus_rating=consensus_rating,
            system_confidence=confidence,
            was_auto_rated=was_auto_rated,
            timestamp=datetime.now().isoformat(),
        )

        # Step 6: Save outcome
        outcome_file = self.outcomes_dir / f"{task_id}.json"
        outcome_file.write_text(json.dumps({
            "outcome": outcome.__dict__,
            "comparison": comparison,
            "thompson_update": thompson_update,
        }, indent=2))

        logger.info(f"Outcome saved: {outcome_file}")

        return outcome

    def get_recent_outcomes(self, days: int = 7) -> list:
        """Get recent task outcomes for reporting"""
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        outcomes = []

        for outcome_file in sorted(self.outcomes_dir.glob("*.json")):
            try:
                data = json.loads(outcome_file.read_text())
                if data["outcome"]["timestamp"] > cutoff:
                    outcomes.append(data)
            except:
                pass

        return outcomes


def main():
    """Process task outcome from hook stdin or demo"""
    analyzer = PostTaskAnalyzer()

    # Try to read task data from stdin (sent by hook)
    task_data = None
    user_rating = None

    try:
        # Check if there's data on stdin
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            json_input = sys.stdin.read()
            if json_input.strip():
                task_data = json.loads(json_input)
                logger.info(f"Received task data from hook: {task_data.get('task_id')}")
    except Exception as e:
        logger.debug(f"No stdin data or parse error: {e}")

    # If we got data from hook, use it; otherwise use demo
    if task_data:
        outcome = analyzer.analyze(
            task_id=task_data.get("task_id", "unknown"),
            task_type=task_data.get("task_type", "unknown"),
            model_used=task_data.get("model_used", "unknown"),
            input_tokens=task_data.get("input_tokens", 0),
            output_tokens=task_data.get("output_tokens", 0),
            cost=task_data.get("cost", 0),
            user_rating=task_data.get("user_rating"),  # Extract user_rating from hook
        )
    else:
        # Demo: analyze a task outcome
        outcome = analyzer.analyze(
            task_id="task_001_code_review",
            task_type="code_review",
            model_used="haiku",
            input_tokens=2000,
            output_tokens=800,
            cost=0.0065,
            user_rating=4,  # User rated it good
        )

    print("\n" + "=" * 70)
    print("OUTCOME ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Task ID:        {outcome.task_id}")
    print(f"Task Type:      {outcome.task_type}")
    print(f"Model:          {outcome.model_used}")
    print(f"User Rating:    {outcome.user_rating}")
    print(f"Consensus:      {outcome.consensus_rating}")
    print(f"System Conf:    {outcome.system_confidence:.1%}")
    print(f"Auto-Rated:     {outcome.was_auto_rated}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
