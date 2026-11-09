import assert from 'node:assert/strict'
import test from 'node:test'
import { authorizePRMutation } from '../shared/pr-mutation-policy.js'

const context = {
  repository: 'FlossWare/claude-ensemble',
  baseBranch: 'main',
}

test('denies mutations by default', () => {
  const result = authorizePRMutation({ ...context, action: 'approve', env: {} })
  assert.equal(result.allowed, false)
})

test('allows an explicitly authorized action', () => {
  const env = {
    CLAUDE_ENSEMBLE_PR_MUTATIONS: 'comment,approve',
    CLAUDE_ENSEMBLE_PR_REPOSITORIES: 'FlossWare/claude-ensemble',
  }
  const result = authorizePRMutation({ ...context, action: 'approve', env })
  assert.equal(result.allowed, true)
})

test('denies actions outside the explicit allowlist', () => {
  const env = {
    CLAUDE_ENSEMBLE_PR_MUTATIONS: 'comment',
    CLAUDE_ENSEMBLE_PR_REPOSITORIES: 'FlossWare/claude-ensemble',
  }
  const result = authorizePRMutation({ ...context, action: 'approve', env })
  assert.equal(result.allowed, false)
})

test('denies repositories outside the explicit repository allowlist', () => {
  const env = {
    CLAUDE_ENSEMBLE_PR_MUTATIONS: 'approve',
    CLAUDE_ENSEMBLE_PR_REPOSITORIES: 'FlossWare/other-repo',
  }
  const result = authorizePRMutation({ ...context, action: 'approve', env })
  assert.equal(result.allowed, false)
})

test('denies unauthorized base branches', () => {
  const env = {
    CLAUDE_ENSEMBLE_PR_MUTATIONS: 'approve',
    CLAUDE_ENSEMBLE_PR_REPOSITORIES: 'FlossWare/claude-ensemble',
    CLAUDE_ENSEMBLE_PR_BASE_BRANCHES: 'release',
  }
  const result = authorizePRMutation({ ...context, action: 'approve', env })
  assert.equal(result.allowed, false)
})

test('rejects unknown mutation actions', () => {
  const env = { CLAUDE_ENSEMBLE_PR_MUTATIONS: 'delete' }
  const result = authorizePRMutation({ ...context, action: 'delete', env })
  assert.equal(result.allowed, false)
})
