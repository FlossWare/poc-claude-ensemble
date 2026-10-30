#!/usr/bin/env python3
"""
Conversation Feedback Capture

Allows manual rating of models based on observed performance or insights
during development work. Feeds back into Learning service to update Thompson.

Usage:
  feedback_capture.py --model cursor --rating 5 --task refactoring \
    --context "Excellent code transformations, fast iteration"

  feedback_capture.py --model gemini --rating 2 --task security_audit \
    --context "Missed SQL injection vulnerability"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.learning_client import LearningClient


def capture_feedback(model: str, rating: int, task_type: str, context: str = None):
    """
    Record conversation feedback as a learning outcome.
    
    Args:
        model: Model being rated (haiku, sonnet, cursor, gemini, etc.)
        rating: Quality rating (1-5)
        task_type: Task type (code_review, refactoring, security_audit, etc.)
        context: Optional context/notes about the observation
    
    Returns:
        Task ID of the recorded outcome
    """
    
    if not 1 <= rating <= 5:
        print(f"Error: rating must be 1-5, got {rating}")
        return None
    
    client = LearningClient()
    
    task_id = f"feedback_{model}_{datetime.now().isoformat()}"
    
    result = client.process_outcome(
        task_id=task_id,
        task_type=task_type,
        model=model,
        rating=rating,
        tokens=0,  # Feedback doesn't have token cost
        cost=0.0
    )
    
    if result:
        print(f"✓ Feedback recorded:")
        print(f"  Model: {model}")
        print(f"  Rating: {rating}/5")
        print(f"  Task: {task_type}")
        if context:
            print(f"  Notes: {context}")
        print(f"  Task ID: {task_id}")
        return task_id
    else:
        print("✗ Failed to record feedback")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Capture conversation feedback to update Learning/Thompson"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model being rated (haiku, sonnet, opus, cursor, gemini, etc.)"
    )
    
    parser.add_argument(
        "--rating",
        type=int,
        required=True,
        help="Rating 1-5 (1=poor, 5=excellent)"
    )
    
    parser.add_argument(
        "--task",
        required=True,
        dest="task_type",
        help="Task type (code_review, refactoring, security_audit, etc.)"
    )
    
    parser.add_argument(
        "--context",
        help="Optional context/notes about the observation"
    )
    
    args = parser.parse_args()
    
    task_id = capture_feedback(
        model=args.model,
        rating=args.rating,
        task_type=args.task_type,
        context=args.context
    )
    
    return 0 if task_id else 1


if __name__ == "__main__":
    sys.exit(main())
