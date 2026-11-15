export const meta = {
  name: 'code-release-notes',
  description: 'Generate release notes from commits with multi-AI categorization',
  phases: [
    { title: 'Detect Platform', detail: 'Identify GitHub/GitLab' },
    { title: 'Find Last Release', detail: 'Get previous release tag' },
    { title: 'Analyze Commits', detail: 'Categorize commits since last release' },
    { title: 'Multi-AI Categorization', detail: 'Consensus on categories' },
    { title: 'Impact Analysis', detail: 'Prioritize changes by impact' },
    { title: 'Generate Notes', detail: 'Create structured release notes' },
    { title: 'User Confirmation', detail: 'User reviews and approves' },
    { title: 'Publish Release', detail: 'Tag and publish release' },
  ],
}

// Configuration
const AUTONOMOUS = args?.autonomous === true  // INTERACTIVE by default
const VERSION = args?.version || args?.[0] || null  // e.g., "v1.2.3" or auto-increment

log(`🤖 Mode: ${AUTONOMOUS ? 'AUTONOMOUS' : 'INTERACTIVE (prompts before publishing)'}`)
if (!AUTONOMOUS) {
  log(`💡 Use release-notes-auto for fully autonomous mode`)
}

if (VERSION) {
  log(`📦 Target version: ${VERSION}`)
} else {
  log(`📦 Version: Will auto-increment from last release`)
}

// ============================================================================
// PHASE 1: Detect Platform
// ============================================================================

phase('Detect Platform')

log('🔧 Detecting platform...')

const platform = await agent(`Detect platform.

Execute:
if git remote -v | grep -q 'github.com'; then echo "github"
elif git remote -v | grep -q 'gitlab'; then echo "gitlab"
else echo "unknown"
fi

Also check CLI availability.`, {
  label: 'Detect Platform',
  schema: {
    type: 'object',
    properties: {
      platform: { type: 'string', enum: ['github', 'gitlab', 'unknown'] },
      cli: { type: 'string' }
    }
  }
})

log(`✅ Platform: ${platform.platform}`)

// ============================================================================
// PHASE 2: Find Last Release
// ============================================================================

phase('Find Last Release')

log('🔍 Finding last release...')

const lastRelease = await agent(`Find the last release tag.

Execute:
git fetch --tags
git describe --tags --abbrev=0 2>/dev/null || echo "none"

Return the last release tag or "none" if no releases exist.`, {
  label: 'Last Release',
  schema: {
    type: 'object',
    properties: {
      tag: { type: 'string' },
      exists: { type: 'boolean' }
    }
  }
})

const hasLastRelease = lastRelease.exists && lastRelease.tag !== 'none'

if (hasLastRelease) {
  log(`✅ Last release: ${lastRelease.tag}`)
} else {
  log(`ℹ️  No previous releases found (first release)`)
}

// ============================================================================
// PHASE 3: Analyze Commits
// ============================================================================

phase('Analyze Commits')

log('📊 Analyzing commits since last release...')

const commitRange = hasLastRelease ? `${lastRelease.tag}..HEAD` : 'HEAD'

const commits = await agent(`Get all commits since last release.

Execute:
git log ${commitRange} --pretty=format:"%H|%s|%b|%an|%ad" --date=short

Return structured commit data.`, {
  label: 'Get Commits',
  schema: {
    type: 'object',
    properties: {
      commits: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            hash: { type: 'string' },
            subject: { type: 'string' },
            body: { type: 'string' },
            author: { type: 'string' },
            date: { type: 'string' }
          }
        }
      },
      total: { type: 'number' }
    }
  }
})

log(`✅ Found ${commits.total || 0} commits`)

if (!commits.commits || commits.commits.length === 0) {
  log(`⚠️  No commits since last release - nothing to release!`)
  return {
    status: 'no_changes',
    message: 'No commits to release'
  }
}

// ============================================================================
// PHASE 4: Multi-AI Categorization
// ============================================================================

phase('Multi-AI Categorization')

log('🤖 Categorizing commits with multi-AI consensus...')

// Load config (required)
let userModelConfig = null
try {
  const path = require('path')
  const fs = require('fs')
  const os = require('os')
  const configPath = path.join(os.homedir(), '.claude/toolkit-models.yaml')
  if (!fs.existsSync(configPath)) {
    log(`❌ Configuration Required: ~/.claude/toolkit-models.yaml not found`)
    process.exit(1)
  }
  const yaml = require('js-yaml')
  userModelConfig = yaml.load(fs.readFileSync(configPath, 'utf8'))
  if (!userModelConfig || typeof userModelConfig !== 'object') {
    log(`❌ Invalid configuration file (empty or malformed YAML)`)
    process.exit(1)
  }
  log(`[Config] Loaded ~/.claude/toolkit-models.yaml`)
} catch (err) {
  log(`❌ Failed to load config: ${err.message}`)
  process.exit(1)
}
const skillConfig = userModelConfig.skill_defaults?.['code-release-notes']
if (!skillConfig) { log(`❌ Skill config not found`); process.exit(1) }
if (!Array.isArray(skillConfig.models)) {
  log(`❌ Invalid models configuration (must be an array)`)
  process.exit(1)
}

const WORKERS = skillConfig.models

const categorizations = await Promise.all(WORKERS.map(model =>
  agent(`Categorize these commits for a release.

Commits:
${commits.commits.slice(0, 50).map(c => `- ${c.subject}`).join('\n')}

Categorize each commit into:
- **Features**: New functionality
- **Fixes**: Bug fixes
- **Breaking**: Breaking changes (API changes, incompatible updates)
- **Performance**: Performance improvements
- **Documentation**: Doc updates
- **Chore**: Dependencies, build, tooling (usually not user-facing)

Also identify:
- Notable changes (important for users)
- Breaking changes (CRITICAL for users)

Return categorized commits.`, {
    label: `Categorize (${model})`,
    model,
    schema: {
      type: 'object',
      properties: {
        features: { type: 'array', items: { type: 'string' } },
        fixes: { type: 'array', items: { type: 'string' } },
        breaking: { type: 'array', items: { type: 'string' } },
        performance: { type: 'array', items: { type: 'string' } },
        documentation: { type: 'array', items: { type: 'string' } },
        chore: { type: 'array', items: { type: 'string' } },
        notable: { type: 'array', items: { type: 'string' } }
      }
    }
  }).catch(() => null)
))

const validCategorizations = categorizations.filter(Boolean)

log(`✅ ${validCategorizations.length} models categorized`)

// Arbiter consensus
const arbiterResult = await agent(`Merge categorizations from ${validCategorizations.length} models.

${validCategorizations.map((cat, idx) => `
Model ${idx + 1}:
- Features: ${cat.features?.length || 0}
- Fixes: ${cat.fixes?.length || 0}
- Breaking: ${cat.breaking?.length || 0}
- Notable: ${cat.notable?.length || 0}
`).join('\n')}

Create final consensus categorization.
Resolve conflicts by majority vote.
Identify truly breaking changes (not just large features).

Return merged categories.`, {
  label: 'Arbiter Consensus',
  model: 'opus',
  schema: {
    type: 'object',
    properties: {
      features: { type: 'array', items: { type: 'string' } },
      fixes: { type: 'array', items: { type: 'string' } },
      breaking: { type: 'array', items: { type: 'string' } },
      performance: { type: 'array', items: { type: 'string' } },
      documentation: { type: 'array', items: { type: 'string' } },
      chore: { type: 'array', items: { type: 'string' } },
      notable: { type: 'array', items: { type: 'string' } }
    }
  }
})

log(`✅ Consensus reached`)
log(`   Features: ${arbiterResult.features?.length || 0}`)
log(`   Fixes: ${arbiterResult.fixes?.length || 0}`)
log(`   Breaking: ${arbiterResult.breaking?.length || 0}`)

// ============================================================================
// PHASE 5: Impact Analysis
// ============================================================================

phase('Impact Analysis')

log('🎯 Analyzing impact of changes...')

// Score changes by impact
const allChanges = [
  ...(arbiterResult.breaking || []).map(c => ({ type: 'breaking', text: c, score: 100 })),
  ...(arbiterResult.features || []).map(c => ({ type: 'feature', text: c, score: 75 })),
  ...(arbiterResult.performance || []).map(c => ({ type: 'performance', text: c, score: 70 })),
  ...(arbiterResult.fixes || []).map(c => ({ type: 'fix', text: c, score: 60 })),
  ...(arbiterResult.documentation || []).map(c => ({ type: 'docs', text: c, score: 30 })),
  ...(arbiterResult.chore || []).map(c => ({ type: 'chore', text: c, score: 20 }))
]

// Boost notable changes
arbiterResult.notable?.forEach(notable => {
  const change = allChanges.find(c => c.text.includes(notable) || notable.includes(c.text))
  if (change) {
    change.score += 20
    change.notable = true
  }
})

// Sort by impact
allChanges.sort((a, b) => b.score - a.score)

log(`✅ Impact analysis complete`)
log(`   Highest impact: ${allChanges[0]?.text?.slice(0, 60)}`)

// ============================================================================
// PHASE 6: Generate Notes
// ============================================================================

phase('Generate Notes')

log('📝 Generating release notes...')

// Determine version
let releaseVersion = VERSION
if (!releaseVersion) {
  // Auto-increment patch version
  if (hasLastRelease) {
    const match = lastRelease.tag.match(/v?(\d+)\.(\d+)\.(\d+)/)
    if (match) {
      const [, major, minor, patch] = match
      releaseVersion = `v${major}.${minor}.${parseInt(patch) + 1}`
    } else {
      releaseVersion = 'v1.0.0'
    }
  } else {
    releaseVersion = 'v1.0.0'
  }
}

log(`   Version: ${releaseVersion}`)

// Generate release notes markdown
// Note: Date should be added by caller after workflow completes
let releaseNotes = `# ${releaseVersion}

`

// Breaking changes (if any)
if (arbiterResult.breaking && arbiterResult.breaking.length > 0) {
  releaseNotes += `## ⚠️ BREAKING CHANGES\n\n`
  arbiterResult.breaking.forEach(change => {
    releaseNotes += `- ${change}\n`
  })
  releaseNotes += `\n`
}

// Features
if (arbiterResult.features && arbiterResult.features.length > 0) {
  releaseNotes += `## ✨ Features\n\n`
  arbiterResult.features.forEach(feature => {
    const isNotable = arbiterResult.notable?.some(n => feature.includes(n) || n.includes(feature))
    releaseNotes += isNotable ? `- **${feature}** 🌟\n` : `- ${feature}\n`
  })
  releaseNotes += `\n`
}

// Bug fixes
if (arbiterResult.fixes && arbiterResult.fixes.length > 0) {
  releaseNotes += `## 🐛 Bug Fixes\n\n`
  arbiterResult.fixes.forEach(fix => {
    releaseNotes += `- ${fix}\n`
  })
  releaseNotes += `\n`
}

// Performance
if (arbiterResult.performance && arbiterResult.performance.length > 0) {
  releaseNotes += `## ⚡ Performance\n\n`
  arbiterResult.performance.forEach(perf => {
    releaseNotes += `- ${perf}\n`
  })
  releaseNotes += `\n`
}

// Documentation
if (arbiterResult.documentation && arbiterResult.documentation.length > 0) {
  releaseNotes += `## 📚 Documentation\n\n`
  arbiterResult.documentation.forEach(doc => {
    releaseNotes += `- ${doc}\n`
  })
  releaseNotes += `\n`
}

// Stats
releaseNotes += `---\n\n`
releaseNotes += `**Stats**: ${commits.total} commits since ${hasLastRelease ? lastRelease.tag : 'initial commit'}\n`

log(`✅ Release notes generated`)

// ============================================================================
// PHASE 7: User Confirmation
// ============================================================================

phase('User Confirmation')

if (!AUTONOMOUS) {
  log('')
  log('═'.repeat(60))
  log('📋 RELEASE NOTES PREVIEW')
  log('═'.repeat(60))
  log(releaseNotes)
  log('═'.repeat(60))
  log('')

  // ASK USER: Publish this release?
  const userDecision = await agent(`Review and approve release notes.

Version: ${releaseVersion}
Commits: ${commits.total}
Breaking changes: ${arbiterResult.breaking?.length || 0}
Features: ${arbiterResult.features?.length || 0}
Fixes: ${arbiterResult.fixes?.length || 0}

Should we publish this release?

Options:
- YES: Publish release with these notes
- EDIT: Let me edit the notes first (will provide text)
- NO: Don't publish (just show notes)

Return your decision.`, {
    label: 'User Decision',
    schema: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['YES', 'EDIT', 'NO'] },
        edited_notes: { type: 'string' },
        reasoning: { type: 'string' }
      },
      required: ['action']
    }
  })

  log(`\n👤 User Decision: ${userDecision.action}`)

  if (userDecision.action === 'NO') {
    log(`ℹ️  Release not published (user chose NO)`)
    return {
      status: 'preview_only',
      version: releaseVersion,
      notes: releaseNotes,
      message: 'Release notes generated but not published'
    }
  }

  if (userDecision.action === 'EDIT' && userDecision.edited_notes) {
    log(`✏️  Using user-edited notes`)
    releaseNotes = userDecision.edited_notes
  }
}

// ============================================================================
// PHASE 8: Publish Release
// ============================================================================

phase('Publish Release')

log('🚀 Publishing release...')

// Create git tag
await agent(`Create git tag.

Execute:
git tag -a ${releaseVersion} -m "${releaseVersion}"
git push origin ${releaseVersion}

Tag and push.`, {
  label: 'Create Tag'
})

log(`✅ Git tag created: ${releaseVersion}`)

// Create GitHub/GitLab release
const createReleaseCmd = platform.platform === 'gitlab'
  ? `glab release create ${releaseVersion} --notes "${releaseNotes.replace(/"/g, '\\"')}"`
  : `gh release create ${releaseVersion} --notes "${releaseNotes.replace(/"/g, '\\"')}"`

await agent(`Create release.

Execute:
${createReleaseCmd}

Create the release.`, {
  label: 'Create Release'
})

log(`✅ Release published: ${releaseVersion}`)

log('')
log('═'.repeat(60))
log('🎉 RELEASE PUBLISHED')
log('═'.repeat(60))
log(`Version: ${releaseVersion}`)
log(`Commits: ${commits.total}`)
log(`Breaking: ${arbiterResult.breaking?.length || 0}`)
log(`Features: ${arbiterResult.features?.length || 0}`)
log(`Fixes: ${arbiterResult.fixes?.length || 0}`)
log('═'.repeat(60))

const result = {
  status: 'published',
  version: releaseVersion,
  commits: commits.total,
  breaking: arbiterResult.breaking?.length || 0,
  features: arbiterResult.features?.length || 0,
  fixes: arbiterResult.fixes?.length || 0,
  notes: releaseNotes
}

// Extract learnings
try {
  await workflow('ai-extract-learning', {
    workflow_name: 'code-release-notes',
    execution_data: result
  })
} catch (error) {
  log(`⚠️ Learning extraction failed: ${error.message}`)
}

return result
