# Conversation Feedback Capture

Record observations and insights from development work to update Thompson model selection.

## Quick Start

```bash
# Cursor excelled at refactoring
python3 tools/feedback_capture.py --model cursor --rating 5 --task refactoring \
  --context "Excellent code transformations"

# Gemini struggled with security
python3 tools/feedback_capture.py --model gemini --rating 2 --task security_audit \
  --context "Missed vulnerability"

# Haiku did well
python3 tools/feedback_capture.py --model haiku --rating 4 --task documentation
```

## Why This Matters

Your **observations feed into Thompson's learning**:

1. You use a model and notice something
2. You rate it with feedback_capture.py
3. Learning records it as an outcome
4. Thompson's probabilities adjust
5. Next time you ask for that task type, Thompson is smarter

## Usage

```bash
python3 tools/feedback_capture.py \
  --model <MODEL> \
  --rating <1-5> \
  --task <TASK_TYPE> \
  [--context "optional notes"]
```

### Arguments

- `--model` (required): Model name (haiku, sonnet, opus, cursor, gemini, etc.)
- `--rating` (required): Quality rating 1-5
  - 1 = Poor (wrong/slow/expensive)
  - 2 = Below average
  - 3 = Acceptable
  - 4 = Good
  - 5 = Excellent
- `--task` (required): Task type (code_review, refactoring, security_audit, documentation, testing, bug_analysis, etc.)
- `--context` (optional): Notes explaining the observation

## Examples

### Recording a Success

```bash
python3 tools/feedback_capture.py \
  --model cursor \
  --rating 5 \
  --task refactoring \
  --context "Excellent variable renaming and code organization"
```

### Recording a Failure

```bash
python3 tools/feedback_capture.py \
  --model gemini \
  --rating 1 \
  --task security_audit \
  --context "Missed SQL injection vulnerability in prepared statement"
```

### Quick Rating (no context)

```bash
python3 tools/feedback_capture.py --model haiku --rating 4 --task documentation
```

## How It Affects Thompson

**Each rating updates Thompson:**

Rating 1-2 (Failure):
- Model.failures += 1
- Lowers probability for that task type next time

Rating 3 (Neutral):
- Just records (no signal)
- Doesn't change probability

Rating 4-5 (Success):
- Model.successes += 1
- Raises probability for that task type next time

## Viewing Impact

Check Learning report:

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from learning.learning_client import LearningClient
report = LearningClient().get_report()
print(f"Total outcomes: {report['total_outcomes']}")
print(f"By model: {report['by_model']}")
