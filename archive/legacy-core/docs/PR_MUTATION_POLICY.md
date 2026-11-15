# Autonomous PR Mutation Policy

Autonomous workflows must not decide for themselves whether they are allowed to mutate a pull request.

The policy is deterministic and deny-by-default. A mutation is permitted only when its action, repository, and base branch satisfy the explicit operator configuration.

## Trust boundary

Repository identity and the PR base branch used for authorization are **not model inputs**.

Before a mutation is attempted, the workflow obtains:

- repository identity from the local `origin` remote
- the PR base branch from the platform CLI (`gh pr view` for GitHub or `glab mr view` for GitLab)

If either lookup fails, authorization fails closed.

The model may analyze the PR and propose an action, but it cannot choose the repository or branch against which that authorization is evaluated.

## Configuration

Set CLAUDE_ENSEMBLE_PR_MUTATIONS to a comma-separated allowlist:

- comment
- approve
- request_changes
- merge
- close

Optionally restrict repositories with CLAUDE_ENSEMBLE_PR_REPOSITORIES, using exact owner/name values.

Optionally restrict base branches with CLAUDE_ENSEMBLE_PR_BASE_BRANCHES. If omitted, only main and master are permitted.

Example:

    CLAUDE_ENSEMBLE_PR_MUTATIONS=comment,approve,request_changes
    CLAUDE_ENSEMBLE_PR_REPOSITORIES=FlossWare/claude-ensemble
    CLAUDE_ENSEMBLE_PR_BASE_BRANCHES=main

Leaving CLAUDE_ENSEMBLE_PR_MUTATIONS unset disables autonomous PR mutations while still allowing the workflow to perform read-only analysis.

Repository and branch matching is exact and case-sensitive.

## Scope

`merge` and `close` are reserved policy actions. The current autonomous review workflow does not execute either action.

## Design rule

The model or arbiter may recommend an action, but it does not grant authorization. Authorization is evaluated by deterministic code immediately before the mutation is attempted.
