/**
 * Caching Bridge
 * Calls Python caching API from JavaScript
 * Tracks prompt cache hits/misses for token cost optimization
 */

import { execFileSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PYTHON = 'python3'
const CACHE_BRIDGE = path.join(__dirname, '../scripts/python/cache-bridge.py')

function runPython(payload, timeout) {
  const result = execFileSync(PYTHON, [CACHE_BRIDGE], {
    input: JSON.stringify(payload),
    encoding: 'utf-8',
    timeout,
    stdio: ['pipe', 'pipe', 'pipe']
  }).trim()

  return JSON.parse(result)
}

class CachingBridge {
  constructor(workflowName) {
    this.workflowName = workflowName
    this.metricsFile = path.join(
      os.homedir(),
      `.claude/cache-metrics-${workflowName}-${Date.now()}.json`
    )
    this.initializeMetrics()
  }

  initializeMetrics() {
    try {
      runPython({
        operation: 'initialize',
        workflow_name: this.workflowName,
        metrics_file: this.metricsFile
      }, 3000)

      log(`[Caching] Initialized metrics for ${this.workflowName}`)
    } catch (err) {
      log(`⚠️ Caching initialization failed: ${err.message}`)
    }
  }

  logCacheRequest(requestType, inputTokens, outputTokens, cacheHit = false, cacheTokens = 0) {
    try {
      runPython({
        operation: 'log',
        request_type: requestType,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cache_hit: cacheHit,
        cache_tokens: cacheTokens,
        metrics_file: this.metricsFile
      }, 3000)

      const status = cacheHit ? 'HIT' : 'MISS'
      log(`[Caching] ${requestType}: ${status} (${inputTokens}→${outputTokens} tokens)`)
    } catch (err) {
      // Silent fail - caching is optional
    }
  }

  generateReport() {
    try {
      return runPython({
        operation: 'report',
        workflow_name: this.workflowName
      }, 5000)
    } catch (err) {
      log(`⚠️ Could not generate cache report: ${err.message}`)
      return { cache_hits: 0, cache_misses: 0, total_tokens_saved: 0 }
    }
  }
}

export default CachingBridge
