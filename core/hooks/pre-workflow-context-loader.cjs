#!/usr/bin/env node
/**
 * Pre-workflow context loader hook (ECC #236)
 * Loads similar past workflows before execution
 *
 * Uses findSimilarWorkflows() from workflow-storage-adapter
 * to inject learnings from 1,168 stored executions
 */

const path = require('path');

module.exports = async function preWorkflowContextLoader(context) {
  const { workflowName, taskDescription } = context;

  // Skip if no task description
  if (!taskDescription || taskDescription.length < 10) {
    return { similarWorkflows: [] };
  }

  try {
    // Dynamically import workflow storage adapter
    const adapterPath = path.join(
      process.env.CLAUDE_PROJECT_DIR || process.cwd(),
      'shared/workflow-storage-adapter.cjs'
    );

    const { getWorkflowStorage } = require(adapterPath);
    const db = getWorkflowStorage();

    // Find similar past workflows
    console.log(`[CONTEXT LOADER] 🔍 Finding similar workflows for: "${taskDescription.substring(0, 50)}..."`);

    const similar = await db.findSimilarWorkflows(taskDescription, 5);

    if (similar && similar.length > 0) {
      console.log(`[CONTEXT LOADER] ✅ Found ${similar.length} similar workflows`);

      // Extract learnings
      const learnings = [];
      const failures = [];

      for (const wf of similar) {
        console.log(`  - ${wf.workflow_name}: ${wf.outcome} (similarity: ${(1 - wf.distance).toFixed(2)})`);

        if (wf.outcome === 'error') {
          failures.push({
            workflow: wf.workflow_name,
            task: wf.task_description,
            similarity: 1 - wf.distance
          });
        }
      }

      // Warn about common failure patterns
      if (failures.length > 0) {
        console.warn(`[CONTEXT LOADER] ⚠️  ${failures.length} similar workflows failed:`);
        failures.forEach(f => {
          console.warn(`     - ${f.workflow} (similarity: ${f.similarity.toFixed(2)})`);
        });
      }

      return {
        similarWorkflows: similar,
        learnings: learnings,
        failures: failures,
        contextInjected: true
      };
    } else {
      console.log(`[CONTEXT LOADER] ℹ️  No similar workflows found - this is new territory`);
      return { similarWorkflows: [], contextInjected: false };
    }

  } catch (err) {
    // Fail open - don't block workflow if context loading fails
    console.warn(`[CONTEXT LOADER] ⚠️  Context loading error: ${err.message}`);
    return { similarWorkflows: [], error: err.message };
  }
};
