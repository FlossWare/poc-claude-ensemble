export const meta = {
  name: 'code-release-notes-auto',
  description: 'Autonomous release notes generation - auto-publishes releases',
  whenToUse: 'When you want fully automated release generation without manual intervention',
  autonomous: true,
  phases: [
    { title: 'Detect Platform', detail: 'Identify GitHub/GitLab' },
    { title: 'Find Last Release', detail: 'Get previous release tag' },
    { title: 'Analyze Commits', detail: 'Categorize commits' },
    { title: 'Multi-AI Categorization', detail: 'Consensus categorization', model: 'opus' },
    { title: 'Impact Analysis', detail: 'Prioritize changes' },
    { title: 'Generate Notes', detail: 'Create release notes' },
    { title: 'Auto-Publish', detail: 'Tag and publish automatically' },
  ],
}

const VERSION = args?.version || args?.[0] || null

log('')
log('═'.repeat(60))
log('📦 AUTONOMOUS RELEASE NOTES GENERATOR')
log('═'.repeat(60))
log('This workflow auto-publishes releases without confirmation')
log('')
log('Features:')
log('  • Multi-AI commit categorization')
log('  • Impact analysis')
log('  • Breaking change detection')
log('  • Auto-publishes to platform (GitHub/GitLab)')
if (VERSION) {
  log(`  • Target version: ${VERSION}`)
}
log('═'.repeat(60))
log('')

log('🔄 Delegating to code-release-notes workflow with autonomous=true...')
log('')

// Call the base workflow with autonomous flag
const result = await workflow('code-release-notes', {
  autonomous: true,
  version: VERSION
})

log('')
log('═'.repeat(60))
log('✅ AUTONOMOUS RELEASE COMPLETE')
log('═'.repeat(60))
if (result.release_version) {
  log(`Version: ${result.release_version}`)
}
if (result.release_url) {
  log(`URL: ${result.release_url}`)
}
if (result.commits_analyzed) {
  log(`Commits: ${result.commits_analyzed}`)
}
if (result.breaking_changes) {
  log(`Breaking changes: ${result.breaking_changes}`)
}
log('═'.repeat(60))
log('')

return result
