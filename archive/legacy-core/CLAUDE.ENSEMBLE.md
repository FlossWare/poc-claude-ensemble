# Claude Code Practices

**Last Updated:** 2026-09-28  
**Project:** Claude Ensemble - Multi-AI Task Orchestration Toolkit  
**Approved Models:** Claude (Haiku, Sonnet, Opus), Google Gemini, JetBrains Cursor  
**Security:** All API work uses your own configured credentials; never use personal keys

---

## ⚠️ ALWAYS START HERE: Load Project Memory

**On every session start, immediately:**
1. Read `~/.claude/projects/memory/MEMORY.md` — Lists all available memories
2. Check if there are relevant memories for your task
3. Use `mem_search <keyword>` or `Read ~/.claude/projects/memory/<name>.md` to load context

**Available memories:**
- `feedback_*.md` — User preferences, past corrections
- `project_*.md` — Current project state, decisions
- `plan_*.md` — Plans in progress

If user asks about something and you haven't checked memory, **ask to check memory first**.

---

## Review with Arbiter/Workers Pattern

**Works on anything:** Code, documentation, design, decisions, schemas, proposals.

**Each `meta-` prefix adds another review tier:**

```bash
# 1-TIER (standard review)
review PR#123                    # Workers → Arbiter
review ./docs/API.md             # Works on any artifact

# 2-TIER (review + re-review)
meta-review PR#123               # Higher confidence
meta-review ./docs/API.md        # Recommended for critical changes

# 3-TIER (ultra-thorough)
meta-meta-review PR#456          # Triple-checked

# 4+-TIER (extreme vetting)
meta-meta-meta-review PR#789     # Ultra-critical artifacts
```

**What happens automatically:**
1. Workers analyze code from different angles
2. Arbiter synthesizes findings into recommendation
3. Results saved to memory (searchable later)
4. Findings alerted if critical
5. Architecture decisions captured

See **REVIEW_SHORTHAND.md** for full details and options.

---

## Core Principles

1. **User is the arbiter** — Models provide analysis; you make final decisions
2. **Model diversity prevents blind spots** — Don't use same model family for critical reviews
3. **Memory tracks context** — Decisions, project state, preferences persist across sessions
4. **Multi-AI consensus for critical work** — Multiple models vote on bugs, security, breaking changes
5. **Worktrees for branches** — Keep main repo clean, isolate branch work
6. **Always ask before git push** — Never auto-push; user confirms first

---

## When to Use Each Model

### **Haiku 4.5** (Fastest, cheapest)
- Code reading and navigation
- Simple refactoring
- Test writing
- Documentation
- Debugging straightforward issues
- Default for routine tasks

### **Sonnet 5** (Balanced)
- Code review and architecture analysis
- Feature design
- Multi-step problem solving
- Initial review of complex changes (use before Opus)
- Domain-specific validation (SQL, APIs, etc.)

### **Opus 5.5** (Strongest reasoning)
- Critical bug analysis (security, logic flaws)
- Adversarial review of other findings
- Deep technical design and architecture
- Breaking change impact analysis
- Final arbitration on disagreement

### **Google Gemini** (Different reasoning style)
- Alternative perspective on disputed issues
- Breaking confirmation bias when models agree but you doubt
- Independent validation from a different architecture

### **JetBrains Cursor** (IDE-integrated)
- Coding directly in IDE
- Real-time fix suggestions
- Test-driven development
- Interactive iteration

---

## Multi-AI Consensus Rules

### **Use consensus when:**
- ✅ Security findings or vulnerabilities
- ✅ Breaking changes or API redesigns
- ✅ Critical bugs (data loss, silent failures)
- ✅ Domain-specific correctness (SQL, specialized syntax, etc.)
- ✅ Architectural decisions
- ✅ Code destined for production merge

### **Consensus process:**

**Step 1: Initial Review**
- One model reviews the code, design, or issue
- Flag findings with confidence/severity
- Surface open questions

**Step 2: Adversarial Challenge**
- Different model challenges the initial findings
- Find false alarms or missed edge cases
- Push back on recommendations
- Validate domain-specific claims

**Step 3: You Decide**
- Read both reviews
- You are the arbiter (not another model)
- Ask clarifying questions if needed
- Make the call

### **Safe pairings (avoid confirmation bias):**
- ✅ Sonnet + Opus (different reasoning styles)
- ✅ Opus + Gemini (completely different architecture)
- ✅ Claude + Cursor (different interfaces)
- ❌ Two Claude models (too similar)
- ❌ Same model reviewing itself (circular)

---

## Memory System

### **What to save:**

**Feedback (preferences & past corrections):**
```
name: always_load_memory_first
description: Load memory at start of every response before doing anything
metadata:
  type: feedback
```

**Reference (how things work):**
```
name: github_api_patterns
description: GitHub API endpoints, token handling, PR creation
metadata:
  type: reference
```

**Project (current initiatives, deadlines):**
```
name: feature_x_status
description: Feature X - known bugs, next steps, dependencies
metadata:
  type: project
```

### **Storage & Access:**

**Central Memory Service (synced across sessions):**
- Access via memory service (running at startup)
- Query with shell functions (if `ensemble-init.sh` sourced):
  ```bash
  mem_list                              # List all memory files
  mem_read project_arbitration_pattern  # Read specific memory
  mem_search arbiter                    # Search for keyword
  ```

**Or use query-memory.py directly:**
  ```bash
  query-memory.py list
  query-memory.py read <name>
  query-memory.py search <keyword>
  ```

**Local files** (automatically synced to memory service):
- `~/.claude/projects/memory/MEMORY.md` — Index of all memories
- `~/.claude/projects/memory/feedback_*.md` — User preferences
- `~/.claude/projects/memory/project_*.md` — Project context
- `~/.claude/projects/memory/plan_*.md` — Plans in progress

### **Don't save:**
- Code snippets (use git history)
- Architecture (read CLAUDE.md and codebase)
- Recent git changes (use `git log`)

---

## Best Practices

### **API Keys & Credentials**
- ✅ Use your own configured credentials
- ✅ Store in environment variables or secure vault
- ❌ Never commit .env files or credentials
- ❌ Never use personal API keys from other services

### **Git Workflow**
```bash
# Create isolated worktree for branch work
git worktree add /tmp/feature-branch -b feature-name

# Work in isolated tree (doesn't touch main repo)

# Push only after user confirmation
# "ready to push? [describe changes]"

# Never force-push main or published branches
```

### **Model Selection**
- **Default:** Haiku (cheapest, fastest)
- **Complex tasks:** Sonnet (balanced)
- **Critical work:** Multi-model consensus
- **Speed matters:** Opus instead of Sonnet

### **Approval Gates**
- **Before implementing:** Show plan, ask approval
- **Before pushing:** Show diff, ask confirmation
- **Before merging:** Review the PR, ask user to merge
- **For breaking changes:** Get stakeholder buy-in first

---

## Anti-Blind-Spot Practices

### **Avoid confirmation bias:**
- When two models agree, ask: "Are we both missing something?"
- Use adversarial review (second model challenges, not validates)
- If a human already flagged issues, trust that first

### **Catch domain-specific gaps:**
- Models good at: syntax, structure, logic flow
- Models bad at: specialized semantics (crypto, specific APIs, etc.)
- If unsure: ask a domain expert before shipping

### **Verify before documenting:**
- Don't write design docs before consensus
- Don't claim "all tests pass" without running them
- Don't mark "complete" if you haven't tested the actual feature

---

## Practical Checklist for Critical Work

**Before implementing:**
- [ ] Read memory for context
- [ ] Understand what changed from last session
- [ ] Ask approval for approach

**During implementation:**
- [ ] Run tests as you go
- [ ] Don't assume—verify

**Before pushing:**
- [ ] Run full test suite
- [ ] Get consensus review (multi-AI if critical)
- [ ] Show diff to user
- [ ] Ask user to confirm push

**For breaking changes:**
- [ ] Get stakeholder input
- [ ] Multi-model consensus review
- [ ] Release notes explaining change
- [ ] Deprecation plan if replacing old behavior

---

## Session Flow

**Session start:**
- [ ] Load memory for this project
- [ ] Check git status (any uncommitted changes?)
- [ ] Ask what you're working on

**During work:**
- [ ] Update memory with new findings
- [ ] Ask before risky actions (force push, delete files)
- [ ] Multi-model consensus for critical changes

**Session end:**
- [ ] Offer to save any new feedback/learnings
- [ ] Remind about uncommitted changes
- [ ] Clean up any temp worktrees

---

## Questions? 

See the project README.md for setup, installation, and feature documentation.

**Version:** Claude Ensemble 1.0  
**Last Updated:** 2026-09-28
