/**
 * Deterministic authorization for autonomous pull-request mutations.
 *
 * Mutations are denied by default. An operator must explicitly authorize
 * actions through environment variables before an autonomous workflow can
 * change a pull request.
 *
 * CLAUDE_ENSEMBLE_PR_MUTATIONS:
 *   Comma-separated actions: comment,approve,request_changes,merge,close
 *
 * CLAUDE_ENSEMBLE_PR_REPOSITORIES:
 *   Optional comma-separated exact repository names (owner/name).
 *
 * CLAUDE_ENSEMBLE_PR_BASE_BRANCHES:
 *   Optional comma-separated exact base branches. Defaults to main,master.
 */

export const PR_MUTATION_ACTIONS = Object.freeze([
  'comment',
  'approve',
  'request_changes',
  'merge',
  'close',
])

const DEFAULT_BASE_BRANCHES = ['main', 'master']

function parseList(value) {
  return (value || '')
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
}

function configuredActions(env = process.env) {
  return new Set(parseList(env.CLAUDE_ENSEMBLE_PR_MUTATIONS))
}

function configuredRepositories(env = process.env) {
  return parseList(env.CLAUDE_ENSEMBLE_PR_REPOSITORIES)
}

function configuredBaseBranches(env = process.env) {
  const branches = parseList(env.CLAUDE_ENSEMBLE_PR_BASE_BRANCHES)
  return branches.length > 0 ? branches : DEFAULT_BASE_BRANCHES
}

export function authorizePRMutation({
  action,
  repository,
  baseBranch,
  env = process.env,
}) {
  if (!PR_MUTATION_ACTIONS.includes(action)) {
    return { allowed: false, reason: 'Unknown PR mutation action: ' + action }
  }

  if (!repository || !baseBranch) {
    return {
      allowed: false,
      reason: 'Repository and base branch are required for PR mutation authorization',
    }
  }

  if (!configuredActions(env).has(action)) {
    return {
      allowed: false,
      reason: 'Action "' + action + '" is not explicitly authorized',
    }
  }

  const repositories = configuredRepositories(env)
  if (repositories.length > 0 && !repositories.includes(repository)) {
    return {
      allowed: false,
      reason: 'Repository "' + repository + '" is not authorized',
    }
  }

  if (!configuredBaseBranches(env).includes(baseBranch)) {
    return {
      allowed: false,
      reason: 'Base branch "' + baseBranch + '" is not authorized',
    }
  }

  return {
    allowed: true,
    reason: 'Action "' + action + '" is authorized for ' + repository + ' -> ' + baseBranch,
  }
}
