/**
 * Load Similar Workflows Hook
 *
 * Pre-workflow context hook that loads similar past workflows and injects
 * learnings into worker prompts. Prevents model echo chambers through diversity checking.
 *
 * Usage:
 *   const context = await loadContext('firmware reverse engineering');
 *   if (context) {
 *     // context.contextPrompt - formatted history to inject into prompts
 *     // context.previousWorkflows - raw workflow data
 *     // context.excludeModels - models to avoid due to >70% dominance
 *     // context.diversityWarning - warning message if echo chamber detected
 *   }
 *
 * Created: 2026-07-01
 */

import { getWorkflowStorage } from '../shared/workflow-storage-adapter.cjs';
import {
  buildContextPrompt,
  analyzeModelDistribution,
  buildDiversityWarning
} from '../shared/context-prompt-builder.mjs';

/**
 * Load context from similar past workflows
 *
 * @param {string} taskDescription - Current task description
 * @param {Object} options - Configuration options
 * @param {number} options.limit - Max similar workflows to load (default 5)
 * @param {number} options.diversityThreshold - Echo chamber threshold (default 70%)
 * @returns {Promise<Object|null>} Context object or null if no matches
 */
async function loadContext(taskDescription, options = {}) {
  const {
    limit = 5,
    diversityThreshold = 70
  } = options;

  if (!taskDescription || typeof taskDescription !== 'string') {
    console.warn('loadContext: invalid taskDescription, must be non-empty string');
    return null;
  }

  try {
    const db = getWorkflowStorage();

    // Find similar workflows via embedding similarity
    const similarWorkflows = await db.findSimilarWorkflows(taskDescription, limit);

    if (!similarWorkflows || similarWorkflows.length === 0) {
      console.log('No similar past workflows found');
      return null;
    }

    console.log(`Found ${similarWorkflows.length} similar past workflows`);

    // Enrich workflows with models_used metadata
    for (const wf of similarWorkflows) {
      const modelsUsed = await db.getModelsUsed(wf.id);

      // Parse existing metadata
      const metadata = wf.metadata
        ? (typeof wf.metadata === 'string' ? JSON.parse(wf.metadata) : wf.metadata)
        : {};

      // Add models_used if not already present
      if (!metadata.models_used) {
        metadata.models_used = modelsUsed;
        wf.metadata = metadata;
      }
    }

    // Build context prompt from workflow history
    const contextPrompt = buildContextPrompt(similarWorkflows);

    // Analyze model distribution for diversity checking
    const modelStats = analyzeModelDistribution(similarWorkflows);

    // Check for echo chamber (>70% single model usage)
    const excludeModels = [];
    let diversityWarning = null;

    if (modelStats.dominantModel && parseFloat(modelStats.dominancePercent) >= diversityThreshold) {
      excludeModels.push(modelStats.dominantModel);
      diversityWarning = buildDiversityWarning(modelStats);

      console.warn(`⚠️ Echo chamber detected: ${modelStats.dominantModel} at ${modelStats.dominancePercent}%`);
      console.warn(`Recommending exclusion to improve diversity`);
    }

    return {
      previousWorkflows: similarWorkflows,
      contextPrompt,
      modelStats,
      excludeModels,
      diversityWarning,
      foundCount: similarWorkflows.length
    };

  } catch (err) {
    console.error('Error loading similar workflows:', err.message);
    return null;
  }
}

/**
 * Inject context into agent prompt
 *
 * @param {string} basePrompt - Original agent prompt
 * @param {Object} context - Context from loadContext()
 * @returns {string} Enhanced prompt with context
 */
function injectContext(basePrompt, context) {
  if (!context || !context.contextPrompt) {
    return basePrompt;
  }

  const parts = [basePrompt];

  // Add workflow history context
  parts.push('\n---\n');
  parts.push(context.contextPrompt);

  // Add diversity warning if present
  if (context.diversityWarning) {
    parts.push(context.diversityWarning);
  }

  return parts.join('\n');
}

export {
  loadContext,
  injectContext
};
