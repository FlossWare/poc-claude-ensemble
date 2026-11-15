---
name: versioning-policy
description: "Project uses X.Y version format, every main commit is a release candidate"
metadata: 
  node_type: memory
  type: project
  originSessionId: 47d07339-1426-4013-ba1a-65875db76a2c
---

Project versioning policy:
- Versions use **X.Y format** (not X.Y.Z)
- Examples: v1.0, v1.1, v1.2, v2.0
- Every commit to main is considered a **release candidate**

**Why:** Simplified versioning scheme. No patch versions - just major.minor. Main branch is always releasable.

**How to apply:**
- When referencing versions in documentation, use X.Y format (v1.1 not v1.1.0)
- Don't create git tags - user manages those
- Every commit to main should be production-ready quality
- Release notes should use X.Y naming convention

Related: [[no-version-management]] - User manages git tags
