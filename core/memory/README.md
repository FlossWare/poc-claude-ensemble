# Global Memory System

This directory contains **global memory** shared across:
- All Claude Code sessions
- Multi-model arbiters
- Multi-model workers
- All projects using `claude-global-skills`

## Purpose

Unlike session-specific memory (in `~/.claude/projects/<project>/memory/`), this global memory is:
- **Version controlled** - tracked in git for reproducibility
- **Shared** - available to all sessions, arbiters, and workers
- **Persistent** - survives session cleanup and project changes
- **Collaborative** - learns from all user interactions across all projects

## Memory Types

### Feedback
User corrections and confirmations about approach, style, and behavior.

**When to save:**
- User corrects approach ("no not that", "don't", "stop doing X")
- User confirms non-obvious approach worked ("yes exactly", "perfect")

**Structure:**
```markdown
---
name: feedback-slug
description: One-line summary
metadata:
  type: feedback
---

Rule itself.

**Why:** Reason/incident that led to this rule
**How to apply:** When/where this guidance kicks in
```

### User
Information about user's role, goals, responsibilities, knowledge, and preferences.

**When to save:**
- Learn details about user's role, preferences, or expertise
- Discover how user wants to collaborate

### Project
Ongoing work, goals, initiatives, bugs, incidents within projects.

**When to save:**
- Who is doing what, why, or by when
- Always convert relative dates to absolute (e.g., "Thursday" → "2026-03-05")

**Structure:**
```markdown
---
name: project-slug
description: One-line summary
metadata:
  type: project
---

Fact or decision.

**Why:** Motivation/constraint/deadline
**How to apply:** How this should shape suggestions
```

### Reference
Pointers to external systems and resources.

**When to save:**
- Learn about resources in external systems
- Bug trackers, docs, dashboards, channels

## What NOT to Save

- Code patterns, conventions, architecture (derive from current code)
- Git history (use `git log`)
- Debugging solutions (already in code/commits)
- Content already in CLAUDE.md files
- Ephemeral task details
- Current conversation context

## Usage

### For Sessions
Sessions automatically load `MEMORY.md` index. To access specific memory:
```bash
Read memory/<name>.md
```

### For Arbiters and Workers
In workflow scripts:
```javascript
// Arbiters and workers can access global memory
const memory = await agent('Read and apply relevant global memory for <task>', {
  phase: 'Context'
});
```

### Updating Memory
1. Create/update memory file with frontmatter
2. Add pointer to `MEMORY.md` index (keep under 200 lines)
3. Commit and push to git

## Symlink Setup

### For RH Team (Recommended)

The shared repo uses symlinks for clean sharing:

```bash
# Clone the repo
git clone <this-repo> ~/Development/.../claude-global-skills

# Symlink memory into your ~/.claude
ln -sf ~/Development/.../claude-global-skills/memory ~/.claude/memory
```

Now all sessions automatically use the shared memory.

### For Project-Specific Memory

If you want both global and project-specific memory:

```bash
# Keep global memory symlinked
ln -sf ~/Development/.../claude-global-skills/memory ~/.claude/memory

# And project-specific memory separate
~/.claude/projects/<project>/memory/          # local, project-specific
~/.claude/memory/                             # shared globally
```

Sessions will check both: local memory first (project context), then global memory (shared learnings).

## Best Practices

1. **Link related memories** - Use `[[other-memory-name]]` in content
2. **Keep descriptions specific** - Index uses them for relevance matching
3. **Update stale memories** - Verify before acting on memory claims
4. **Organize semantically** - Group by topic, not chronologically
5. **No duplicates** - Update existing memory instead of creating new

## Migration from Session Memory

Current memories have been copied to this global directory. Future updates should happen here and be committed to git.
