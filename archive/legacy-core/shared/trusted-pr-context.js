import { execFileSync } from 'node:child_process'

export function parseRepositoryFromRemote(remoteUrl) {
  const value = String(remoteUrl || '').trim()
  const match = value.match(/(?:github\.com|gitlab\.com)[:/]([^/]+)\/([^/]+?)(?:\.git)?$/i)
  if (!match) {
    throw new Error('Unsupported or invalid origin remote: ' + value)
  }

  const host = value.match(/(?:https?:\/\/|git@)([^/:]+)/i)?.[1]?.toLowerCase()
  if (!['github.com', 'gitlab.com'].includes(host)) {
    throw new Error('Unsupported repository host: ' + host)
  }

  return {
    host,
    repository: match[1] + '/' + match[2],
    platform: host === 'github.com' ? 'github' : 'gitlab',
  }
}

function run(command, args, exec = execFileSync) {
  return exec(command, args, {
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()
}

export function getTrustedPRContext(prNumber, { exec = execFileSync } = {}) {
  if (!Number.isInteger(prNumber) || prNumber < 1) {
    throw new Error('PR number must be a positive integer')
  }

  const remoteUrl = run('git', ['remote', 'get-url', 'origin'], exec)
  const remote = parseRepositoryFromRemote(remoteUrl)

  if (remote.platform === 'github') {
    const raw = run('gh', [
      'pr', 'view', String(prNumber), '--json', 'baseRefName',
    ], exec)
    const metadata = JSON.parse(raw)

    if (typeof metadata.baseRefName !== 'string' || metadata.baseRefName.length === 0) {
      throw new Error('GitHub PR metadata did not contain baseRefName')
    }

    return { repository: remote.repository, baseBranch: metadata.baseRefName, platform: remote.platform }
  }

  const raw = run('glab', [
    'mr', 'view', String(prNumber), '--output', 'json',
  ], exec)
  const metadata = JSON.parse(raw)
  const baseBranch = metadata.target_branch || metadata.targetBranch

  if (typeof baseBranch !== 'string' || baseBranch.length === 0) {
    throw new Error('GitLab MR metadata did not contain target branch')
  }

  return { repository: remote.repository, baseBranch, platform: remote.platform }
}