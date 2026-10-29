/**
 * Compression Bridge
 * Calls Python compression API from JavaScript
 * Reduces token usage via recursive text compression
 */

import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PYTHON = 'python3'
const COMPRESSION_BRIDGE = path.join(__dirname, '../scripts/python/compression-bridge.py')

function runPython(payload) {
  const result = execFileSync(PYTHON, [COMPRESSION_BRIDGE], {
    input: JSON.stringify(payload),
    encoding: 'utf-8',
    timeout: 5000,
    stdio: ['pipe', 'pipe', 'pipe']
  }).trim()

  return JSON.parse(result)
}

function compressDiff(diffText) {
  try {
    const output = runPython({
      text: diffText,
      task_type: 'code-review'
    })

    const ratio = output.original_length === 0
      ? 0
      : ((1 - output.compressed_length / output.original_length) * 100).toFixed(1)

    log(`[Compression] ${output.original_length} → ${output.compressed_length} bytes (${ratio}% reduction)`)
    return output.compressed
  } catch (err) {
    log(`⚠️ Compression unavailable: ${err.message} (using uncompressed)`)
    return diffText
  }
}

function compressContext(contextText) {
  try {
    return runPython({
      text: contextText,
      task_type: 'context'
    }).compressed
  } catch (err) {
    log('⚠️ Compression unavailable (using uncompressed)')
    return contextText
  }
}

export { compressDiff, compressContext }
