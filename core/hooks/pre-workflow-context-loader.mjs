#!/usr/bin/env node

/**
 * Pre-Workflow Context Loader
 *
 * Loads similar past workflows and injects their context into new workflow prompts.
 * Addresses the write-only analytics antipattern by actually querying historical data.
 *
 * Features:
 * - Semantic similarity search using PostgreSQL+pgvector
 * - Diversity checks to prevent model echo chambers (>70% threshold)
 * - Context formatting for readable worker prompts
 * - Metadata tracking for feedback loop analysis
 *
 * Created: 2026-06-30
 * Issue: ECC #192 (Cross-Session Context Inheritance)
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { getWorkflowStorage } from '../shared/workflow-storage-adapter.cjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Format similar workflow results into readable context for worker prompts
 *
 * @param {Array} similarWorkflows - Array of workflow execution objects from findSimilarWorkflows()
 * @returns {string} Formatted context string
 *
 * Example output:
 * ```
 * ## Context from Similar Past Workflows
 *
 * ### Previous workflow #1 (similarity: 0.92)
 * Task: "Research firmware reverse engineering for RAX-75"
 * Outcome: success (45.3s, 6 workers)
 * Key learnings:
 * - QEMU simulation requires OpenWrt buildroot for ARM64 targets
 * - GPL sources often incomplete, use binwalk extraction instead
 * Models used: opus (3), sonnet (2), haiku (1)
 *
 * ### Previous workflow #2 (similarity: 0.85)
 * ...
 * ```
 */
function formatContextForPrompt(similarWorkflows) {
  if (!similarWorkflows || similarWorkflows.length === 0) {
    return '';
  }

  const lines = ['## Context from Similar Past Workflows\n'];

  for (let i = 0; i < similarWorkflows.length; i++) {
    const wf = similarWorkflows[i];
    const similarity = (1 - (wf.distance || 0)).toFixed(2); // Convert distance to similarity score

    lines.push(`### Previous workflow #${i + 1} (similarity: ${similarity})`);
    lines.push(`Task: "${wf.task_description}"`);
    lines.push(`Outcome: ${wf.outcome} (${(wf.total_duration_ms / 1000).toFixed(1)}s, ${wf.total_workers} workers)`);

    // Extract metadata insights if available
    const metadata = typeof wf.metadata === 'string' ? JSON.parse(wf.metadata) : wf.metadata;
    if (metadata) {
      if (metadata.key_learnings && Array.isArray(metadata.key_learnings)) {
        lines.push('Key learnings:');
        for (const learning of metadata.key_learnings.slice(0, 3)) {
          lines.push(`- ${learning}`);
        }
      } else if (metadata.summary) {
        lines.push(`Summary: ${metadata.summary}`);
      }
    }

    // Note: Model distribution added later by getModelDistribution()
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check if similar workflows exhibit model echo chamber effect
 * Returns warning if >70% of workers used the same model
 *
 * @param {Array} similarWorkflows - Array of workflow execution objects
 * @param {Object} db - WorkflowStorageDB instance
 * @returns {Promise<Object>} Diversity analysis result
 *
 * Returns:
 * {
 *   has_echo_chamber: boolean,
 *   dominant_model: string | null,
 *   dominant_percentage: number,
 *   model_distribution: { opus: 12, sonnet: 8, haiku: 4, ... },
 *   recommendation: string | null
 * }
 */
async function diversityCheck(similarWorkflows, db) {
  if (!similarWorkflows || similarWorkflows.length === 0) {
    return {
      has_echo_chamber: false,
      dominant_model: null,
      dominant_percentage: 0,
      model_distribution: {},
      recommendation: null
    };
  }

  // Get model distribution across all similar workflows
  const workflowIds = similarWorkflows.map(wf => wf.id);
  const distribution = await db.getModelDistribution(workflowIds);

  // Calculate total workers and find dominant model
  const totalWorkers = Object.values(distribution).reduce((sum, count) => sum + count, 0);
  let dominantModel = null;
  let dominantCount = 0;

  for (const [model, count] of Object.entries(distribution)) {
    if (count > dominantCount) {
      dominantModel = model;
      dominantCount = count;
    }
  }

  const dominantPercentage = totalWorkers > 0 ? (dominantCount / totalWorkers) : 0;
  const hasEchoChamber = dominantPercentage > 0.70;

  return {
    has_echo_chamber: hasEchoChamber,
    dominant_model: dominantModel,
    dominant_percentage: parseFloat(dominantPercentage.toFixed(2)),
    model_distribution: distribution,
    recommendation: hasEchoChamber
      ? `WARNING: ${dominantModel} used in ${(dominantPercentage * 100).toFixed(0)}% of similar workflows. Consider using alternative models (${Object.keys(distribution).filter(m => m !== dominantModel).join(', ')}) to prevent echo chamber effects.`
      : null
  };
}

/**
 * Inject context from similar workflows into workflow metadata
 *
 * @param {string} taskDescription - New workflow task description
 * @param {Object} options - Configuration options
 * @param {number} options.limit - Max similar workflows to retrieve (default: 5)
 * @param {boolean} options.check_diversity - Enable diversity checks (default: true)
 * @param {boolean} options.include_learnings - Include workflow.learnings table data (default: true)
 * @returns {Promise<Object>} Enriched context object
 *
 * Returns:
 * {
 *   previous_similar_workflows: [...],
 *   formatted_context: "## Context from Similar Past Workflows...",
 *   diversity_analysis: { has_echo_chamber, dominant_model, ... },
 *   metadata: {
 *     context_used: true,
 *     context_source: [workflow_id_1, workflow_id_2, ...],
 *     context_count: 5,
 *     similarity_scores: [0.92, 0.85, ...]
 *   }
 * }
 */
async function injectContextIntoWorkflow(taskDescription, options = {}) {
  const {
    limit = 5,
    check_diversity = true,
    include_learnings = true
  } = options;

  const db = getWorkflowStorage();

  // Find similar workflows using pgvector semantic search
  const similarWorkflows = await db.findSimilarWorkflows(taskDescription, limit);

  if (similarWorkflows.length === 0) {
    return {
      previous_similar_workflows: [],
      formatted_context: '',
      diversity_analysis: null,
      metadata: {
        context_used: false,
        context_source: [],
        context_count: 0,
        similarity_scores: []
      }
    };
  }

  // Format context for prompt injection
  let formattedContext = formatContextForPrompt(similarWorkflows);

  // Run diversity check if enabled
  let diversityAnalysis = null;
  if (check_diversity) {
    diversityAnalysis = await diversityCheck(similarWorkflows, db);

    // Append model distribution to context
    if (Object.keys(diversityAnalysis.model_distribution).length > 0) {
      formattedContext += '\n**Model usage in similar workflows:**\n';
      for (const [model, count] of Object.entries(diversityAnalysis.model_distribution)) {
        formattedContext += `- ${model}: ${count} workers\n`;
      }

      if (diversityAnalysis.recommendation) {
        formattedContext += `\n⚠️  ${diversityAnalysis.recommendation}\n`;
      }
    }
  }

  // Load associated learnings if enabled
  if (include_learnings) {
    const workflowIds = similarWorkflows.map(wf => wf.id);
    const learnings = await db.getLearningSummary(workflowIds);

    if (learnings.length > 0) {
      formattedContext += '\n**Key learnings from similar workflows:**\n';
      for (const learning of learnings.slice(0, 5)) {
        formattedContext += `- ${learning.description} (importance: ${learning.importance})\n`;
      }
    }
  }

  // Build metadata for tracking
  const metadata = {
    context_used: true,
    context_source: similarWorkflows.map(wf => wf.workflow_id),
    context_count: similarWorkflows.length,
    similarity_scores: similarWorkflows.map(wf => parseFloat((1 - (wf.distance || 0)).toFixed(2)))
  };

  return {
    previous_similar_workflows: similarWorkflows,
    formatted_context: formattedContext,
    diversity_analysis: diversityAnalysis,
    metadata
  };
}

export {
  formatContextForPrompt,
  diversityCheck,
  injectContextIntoWorkflow
};
