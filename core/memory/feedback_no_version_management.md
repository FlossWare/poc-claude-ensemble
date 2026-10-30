---
name: no-version-management
description: Never create git tags or manage GitLab revisions - user handles versioning
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 47d07339-1426-4013-ba1a-65875db76a2c
---

Do not create git tags or manage GitLab revisions/releases. The user manages all version control tagging.

**Why:** Version management and release tagging is the user's responsibility, not the AI's. Every commit to main is a release candidate, and the user decides when to create official tags.

**How to apply:** 
- Never run `git tag` commands
- Never push tags to GitLab
- Do not create or manage release tags
- Focus on code, documentation, and tests
- User will handle `git tag` and release management themselves

Related: Versions use X.Y format (not X.Y.Z) per [[versioning-policy]]
