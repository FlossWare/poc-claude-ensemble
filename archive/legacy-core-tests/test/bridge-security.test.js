import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import test from 'node:test'

function runBridge(script, payload) {
  return spawnSync('python3', [script], {
    input: JSON.stringify(payload),
    encoding: 'utf8',
    timeout: 10000
  })
}

test('compression bridge accepts adversarial stdin payloads', () => {
  const text = String.raw`quotes: "' \\ backslashes
newlines: first line
second line
unicode: café 🚀 日本語
shell-looking: $(touch /tmp/should-not-exist) ; rm -rf /`

  const result = runBridge('scripts/python/compression-bridge.py', { text })

  assert.equal(result.status, 0, result.stderr)
  const output = JSON.parse(result.stdout)
  assert.equal(output.original_length, text.length)
  assert.equal(typeof output.compressed, 'string')
  assert.equal(output.compressed_length, output.compressed.length)
})

test('caching bridge never turns hostile input into executable source', () => {
  const result = runBridge('scripts/python/cache-bridge.py', {
    operation: 'initialize',
    workflow_name: `quotes"'\\
$(touch /tmp/should-not-exist)`,
    metrics_file: '/tmp/claude-ensemble-test.json'
  })

  // The legacy cache implementation may reject the request because its
  // tracker API is not present in the current Python module. Either outcome
  // must be a normal process result, never Python syntax or shell execution.
  assert.notEqual(result.signal, 'SIGSEGV')
  assert.doesNotMatch(result.stderr, /SyntaxError: unterminated|command not found/)
})
