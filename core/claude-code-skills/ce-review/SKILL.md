---
name: ce-review
description: "[Lotse] Review the whole code base, or specific local code changes, and open issues for confirmed findings using Claude Ensemble's worker/arbiter workflow through the authenticated Claude Code runtime. Use when asked to review code with CE, without requiring a Jira ticket. Never send repository source to external reviewer providers."
---

# Claude Ensemble code review

Run this workflow in the user's current Claude Code session. Do not claim the review ran unless the subprocess completed and produced a report.

## Invocation

Review the whole code base of the current repository:
```bash
/ce-review --whole-repo
```

Or review a specific file:
```bash
/ce-review path/to/file.py
```

Or review the current repository's changes (no arguments):
```bash
/ce-review
```

Optional parameters:
- `--base BRANCH`: Review against a branch other than `main`.
- `--whole-repo`: Review the whole code base instead of only the diff against the base branch. This works even when the working tree is clean.
- `--open-issues`: Open GitHub issues for confirmed findings. Every issue description must have `[Lotse]` as its first part.

## 1. Verify the target and runtime

1. **Parse arguments:** Extract optional file path, `--base BRANCH`, `--whole-repo`, and `--open-issues` from the skill args if present.
2. **Resolve the target path:** 
   - If a file path is provided, resolve it relative to the current working directory and validate it exists.
   - If no file path is provided, use the current working directory.
   - Confirm the target (file or directory) is inside a Git repository.
3. Do not reset, stash, checkout, clean, amend, or otherwise mutate the target repository.
4. **Check for reviewable content:**
   - If `--whole-repo` was requested, the whole code base is the target; a clean working tree is not a blocker, so continue.
   - If the target is a directory (repository review) and `--whole-repo` was not requested, verify there are changes against the base branch. 
   - If there are no changes, no `--whole-repo` flag, and no file path was specified, explain this and show the exact command for a file-specific review: `python3 $CE_REPO/tools/arbitrate.py code-review <filepath> --target-branch main --phases 1 --tiers 1`
   - Do not run the review if there are no changes and no specific file path was provided.
5. Confirm the CE gateway is healthy:
   `curl -fsS --max-time 5 http://127.0.0.1:8080/api/v1/health`
6. Locate the active CE checkout. Prefer `POC_CLAUDE_ENSEMBLE_REPO_ROOT` when set. Otherwise inspect the active `claude-ensemble.service` unit with `systemctl --user cat claude-ensemble.service` and check the known FlossWare CE checkout paths. Accept a candidate only if it contains `tools/arbitrate.py` and its Git remote identifies the intended FlossWare/claude-ensemble repository. Do not silently use an arbitrary stale checkout. If no valid checkout is found, stop and report the blocker.

## 2. Run the CE worker/arbiter review

Use the active CE checkout's CLI and the user's authenticated Claude Code runtime. Do not invoke web reviewers, Gemini, Grok, Perplexity, or other external reviewer providers for repository source.

The command is the same for repository, whole-code-base, and file reviews; `arbitrate.py` accepts either a directory or a file:

```bash
python3 "$CE_REPO/tools/arbitrate.py" code-review "$TARGET_PATH" \
  --target-branch "$BASE_BRANCH" --phases 1 --tiers 1 $WHOLE_REPO
```

Where `$TARGET_PATH` is either:
- A directory (repository review with changes against `--target-branch`, or the whole code base when `$WHOLE_REPO` is `--whole-repo`), or
- A file path (review a specific file even if it has no staged changes).

Set `$WHOLE_REPO` to `--whole-repo` only when the user asked to review the whole code base, otherwise leave it empty.

Capture the command exit status and complete report. Wait for completion. A nonzero exit, unavailable worker, or empty report is a failed/incomplete review, not success. Do not edit the target repository while reviewing it.

## 3. Report findings

**On successful review completion:**
- target path (file, repository, or whole code base) and base branch
- whether CE completed successfully (exit code 0)
- findings ordered by severity, with file/line references when available
- missing tests and likely regressions
- worker/arbiter errors, unavailable services, or limitations

**On missing content (clean working tree with no file path specified):**
- Clearly state there are no changes to review against the base branch.
- Show the exact command for a file-specific review.
- Do not claim a review ran.

Distinguish confirmed defects from questions and suggestions. Verify every finding against the supplied source. Do not claim a finding is validated unless the source supports it.

## 4. Open issues (optional)

Only when `--open-issues` was requested, open one GitHub issue per confirmed finding. The issue description must have `[Lotse]` as its first part:

Do **not** require or invoke the GitHub CLI (`gh`). Lotse environments may not have it installed. Create issues directly through the GitHub REST API using Python's standard library and a token already supplied in the environment.

For each confirmed finding, use this procedure (the repository is derived from the checked-out `origin` remote; no repository source is sent to an external reviewer):

```bash
export ISSUE_TITLE="$TITLE"
export ISSUE_BODY="[Lotse] $DESCRIPTION"
python3 - <<'PY'
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
if not token:
    raise SystemExit("Cannot open issue: set GITHUB_TOKEN or GH_TOKEN with repository issue-write permission.")

remote = subprocess.check_output(
    ["git", "remote", "get-url", "origin"], text=True
).strip()
match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", remote)
if not match:
    raise SystemExit("Cannot open issue: origin is not a GitHub repository.")
repository = match.group(1)
payload = {
    "title": os.environ["ISSUE_TITLE"],
    "body": os.environ["ISSUE_BODY"],
}
request = urllib.request.Request(
    f"https://api.github.com/repos/{repository}/issues",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Lotse-CE-review",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        issue = json.load(response)
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", "replace")
    raise SystemExit(f"GitHub issue creation failed ({exc.code}): {detail[:1000]}")
print(issue["html_url"])
PY
```

- Open issues only for findings verified against the supplied source; never for unverified questions or suggestions.
- Include file/line references and relevant command/report context in the description.
- Ensure the body begins with `[Lotse]`.
- Report the URL returned by GitHub for every issue opened.
- If the token is missing or GitHub rejects the request, report the failure. Do not claim the issue was opened.
- Do not open issues when `--open-issues` was not requested.
