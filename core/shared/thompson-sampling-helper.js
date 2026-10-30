/**
 * Shared Thompson Sampling model selection with graceful fallback
 *
 * Centralizes the pattern duplicated across ai-web-learn-fleet.js,
 * ai-pdf-deep-research.js, and code-review.js
 */

import { hotImport } from './hot-reload.js';

/**
 * Select worker models using Thompson Sampling with fallback
 *
 * @param {string} context - Context name for Thompson Sampling (e.g., 'web-research-fleet')
 * @param {string[]} allModels - Full pool of available models
 * @param {number} count - Number of models to select
 * @param {Function} log - Logging function
 * @returns {Promise<{models: string[], orchestrator: object|null}>}
 */
export async function selectWorkersWithFallback(context, allModels, count, log) {
  let orchestrator = null;
  let selectedModels = allModels.slice(0, count);  // Default fallback

  try {
    orchestrator = await hotImport('../orchestrator.js');
    selectedModels = await orchestrator.selectWorkers(context, {
      strategy: 'thompson',
      models: allModels,
      count: count
    });
    log(`Thompson Sampling selected models: ${selectedModels.join(', ')}`);
  } catch (e) {
    // Fallback: Use first N from allModels pool
    log(`⚠ DEGRADED MODE: Thompson Sampling unavailable (${e.message}). Falling back to first ${count} models: ${selectedModels.join(', ')}`);
  }

  return { models: selectedModels, orchestrator };
}
