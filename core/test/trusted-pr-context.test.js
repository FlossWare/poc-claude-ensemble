import assert from 'node:assert/strict'
import test from 'node:test'
import { getTrustedPRContext, parseRepositoryFromRemote } from '../shared/trusted-pr-context.js'

test('parses GitHub HTTPS remotes', () => {
  assert.deepEqual(parseRepositoryFromRemote('https://github.com/FlossWare/claude-ensemble.git'), {
    host: 'github.com', repository: 'FlossWare/claude-ensemble', platform: 'github',
  })
})

test('parses GitHub SSH remotes', () => {
  assert.deepEqual(parseRepositoryFromRemote('git@github.com:FlossWare/claude-ensemble.git'), {
    host: 'github.com', repository: 'FlossWare/claude-ensemble', platform: 'github',
  })
})

test('rejects unsupported remotes', () => {
  assert.throws(() => parseRepositoryFromRemote('https://example.com/FlossWare/claude-ensemble.git'), /Unsupported repository host/)
})

test('uses deterministic git and GitHub CLI metadata, not model output', () => {
  const calls = []
  const exec = (command, args) => {
    calls.push([command, args])
    if (command === 'git') return 'https://github.com/FlossWare/claude-ensemble.git\n'
    if (command === 'gh') return JSON.stringify({ baseRefName: 'release' })
    throw new Error('Unexpected command: ' + command)
  }
  assert.deepEqual(getTrustedPRContext(36, { exec }), {
    repository: 'FlossWare/claude-ensemble', baseBranch: 'release', platform: 'github',
  })
  assert.deepEqual(calls, [
    ['git', ['remote', 'get-url', 'origin']],
    ['gh', ['pr', 'view', '36', '--json', 'baseRefName']],
  ])
})

test('rejects missing base branch metadata', () => {
  const exec = (command) => command === 'git'
    ? 'https://github.com/FlossWare/claude-ensemble.git'
    : JSON.stringify({})
  assert.throws(() => getTrustedPRContext(36, { exec }), /baseRefName/)
})