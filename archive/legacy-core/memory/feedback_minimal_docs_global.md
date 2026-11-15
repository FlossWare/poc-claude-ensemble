---
name: minimal-docs-no-bloat-global
description: User wants minimal documentation across all projects - no AI-generated status/review/roadmap files
metadata:
  type: feedback
---

**Rule**: Keep documentation lean across ALL projects. Avoid creating AI-generated bloat.

**Why**: User deleted 13,774 lines of AI-generated noise from VirtOS (21 files of status reports, roadmaps, reviews). Explicitly requested this principle apply to all repositories (FlossWare, solenopsis, sfloess, VirtOS, etc.). Said: "can u learn for all sessions that we want less bloat like this"

**How to apply**:

1. **Never create** these file types in ANY repository:
   - Status reports (STATUS.md, *_STATUS.md, *_SUMMARY.md)
   - Review summaries (REVIEW*.md, PROJECT_REVIEW.md)
   - Roadmaps (ROADMAP*.md, *_ROADMAP.md)
   - Metrics/tracking (METRICS.md, TESTING_METRICS.md)
   - Business/marketing (BUSINESS_CASE.md, COMPARISON.md, BRANDING.md)
   - AI architecture docs (AI-*.md)
   - Future feature speculation (KUBERNETES.md, FEDERATION.md, MICROSERVICES.md)
   - Executive summaries (EXECUTIVE_SUMMARY.md)

2. **Only create/update** essential docs:
   - README.md (project entry point only)
   - CLAUDE.md (project instructions if needed)
   - Core technical docs (ARCHITECTURE.md, API.md)
   - User guides (how to use the code)
   - Working examples (runnable demos)
   - Test evidence (validation results)
   - One honest status doc maximum per project

3. **Strong preferences**:
   - Code + inline comments > separate documentation
   - GitHub issues > planning documents
   - Git commit messages > change summaries
   - Working code > design documents
   - Show, don't tell

4. **Red flags** - ask before creating:
   - File names with: STATUS, REVIEW, ROADMAP, SUMMARY, METRICS, EXECUTIVE
   - Multiple docs on same topic
   - Speculation about future features
   - Marketing content in technical repos
   - Any file that's "explaining what we're doing" vs "explaining how to use it"

5. **When documentation IS needed**:
   - Make it concise (prefer 1 page over 3)
   - Make it actionable (commands to run, not philosophy)
   - Make it maintainable (less is more)
   - Put it close to the code (inline comments, docstrings)

**The principle**: User wants **code first, minimal essential docs only**. If tempted to create a markdown file, ask: "Does this help someone use or contribute to the code RIGHT NOW?" If no, don't create it.

This applies to ALL repositories the user works on, not just VirtOS.
