/**
 * Example workflow demonstrating learning extraction integration
 * Shows best practices for calling ai-extract-learning and loading similar workflows
 */

export const meta = {
  name: 'example-workflow-with-learning',
  description: 'Example showing learning extraction and context loading integration',
  phases: [
    { title: 'Context', detail: 'Load similar past workflows' },
    { title: 'Execute', detail: 'Run main workflow logic' },
    { title: 'Learn', detail: 'Extract and store learnings' }
  ]
};

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

const taskDescription = args?.task || 'example task';

// ============================================================================
// CROSS-SESSION CONTEXT LOADING
// ============================================================================

phase('Context');
log(`📚 Loading context from similar past workflows...`);

const { loadContext, injectContext } = require('./load-similar-workflows.js');
const context = await loadContext(taskDescription, { limit: 5 });

let contextUsed = false;
if (context && context.foundCount > 0) {
  log(`✅ Found ${context.foundCount} similar workflows`);
  if (context.excludeModels.length > 0) {
    log(`⚠️ Diversity check: Recommend avoiding ${context.excludeModels.join(', ')}`);
  }
  contextUsed = true;
} else {
  log('No similar workflows found - proceeding without context');
}

// ============================================================================
// TASK EXECUTION
// ============================================================================

phase('Execute');
log(`🔧 Executing task: ${taskDescription}`);

// Simulate workflow execution
const startTime = Date.now();

// Example: Process some data
const result = {
  processed_items: 42,
  success_rate: 0.95,
  errors_encountered: ['minor timeout', 'rate limit'],
  approach: 'parallel processing with retry',
  files_modified: ['src/processor.js', 'src/utils.js']
};

const duration = Date.now() - startTime;
log(`✅ Task completed in ${duration}ms`);

// ============================================================================
// LEARNING EXTRACTION
// ============================================================================

phase('Learn');
log(`📚 Extracting learnings...`);

// Prepare execution data for learning extraction
const executionData = {
  task_description: taskDescription,
  processed_items: result.processed_items,
  success_rate: result.success_rate,
  errors_encountered: result.errors_encountered,
  approach: result.approach,
  files_modified: result.files_modified,
  duration_ms: duration,
  context_used: contextUsed,
  similar_workflows_found: context ? context.foundCount : 0
};

// Optional: Log execution to monitoring.execution_summary
let executionId = null;
try {
  const { getExecutionMonitor } = require('~/.claude/learning/postgres-adapter');
  const monitor = getExecutionMonitor();

  executionId = await monitor.logExecution({
    model: 'example-worker',
    workflow: meta.name,
    task_type: 'processing',
    quality_score: result.success_rate,
    input_tokens: 100,
    output_tokens: 200,
    cost_usd: 0.001,
    duration_ms: duration,
    outcome: 'success'
  });

  log(`📊 Logged execution #${executionId}`);
} catch (err) {
  log(`⚠️ Could not log execution: ${err.message}`);
}

// Extract learnings using ai-extract-learning workflow
const learningResult = await workflow('ai-extract-learning', {
  run_id: `example-${Date.now()}`,
  workflow_name: meta.name,
  execution_data: executionData,
  execution_id: executionId,
  quality_score: result.success_rate,
  strategy: 'base',  // or 'maximum-coverage', 'quantized', 'quintuple-verification'
  save_to_memory: false
});

if (learningResult.status === 'success') {
  log(`✅ Stored learning #${learningResult.learning_id}`);

  // Display key learnings
  const { learnings } = learningResult;

  if (learnings.recommendations?.length > 0) {
    log(`\n💡 Recommendations:`);
    learnings.recommendations.slice(0, 3).forEach((rec, idx) => {
      log(`   ${idx + 1}. ${rec}`);
    });
  }

  if (learnings.code_patterns?.tech_stack?.length > 0) {
    log(`\n🔧 Tech stack detected: ${learnings.code_patterns.tech_stack.join(', ')}`);
  }

  // Check for high-priority memory suggestions
  const highPriority = learnings.memory_suggestions?.filter(s => s.priority === 'high') || [];
  if (highPriority.length > 0) {
    log(`\n🔴 High-priority memory suggestions:`);
    highPriority.forEach(s => {
      log(`   [${s.type}] ${s.content}`);
    });
  }
} else {
  log(`⚠️ Learning extraction failed`);
}

// ============================================================================
// RETURN RESULT
// ============================================================================

log(`\n✅ Workflow complete`);

return {
  status: 'success',
  task: taskDescription,
  result,
  duration_ms: duration,
  learning_id: learningResult.learning_id,
  quality_score: result.success_rate,

  // Optional: skip automatic learning extraction (if hook is enabled)
  // skip_learning_extraction: true
};

// ============================================================================
// ALTERNATIVE: Manual Learning Extraction (without ai-extract-learning)
// ============================================================================

/**
 * If you want to manually construct and store learnings without calling
 * the ai-extract-learning workflow, you can do this:
 */
async function manualLearningExtraction() {
  const { storeLearnings } = require('~/.claude/learning/storage');

  const manualLearnings = {
    user_patterns: {
      preferences: ['Uses parallel processing'],
      expertise_level: { 'data-processing': 'intermediate' },
      workflow_usage: [`Runs ${meta.name} for batch tasks`]
    },
    code_patterns: {
      common_bugs: result.errors_encountered,
      architecture_insights: ['Parallel processing with retry logic'],
      tech_stack: ['Node.js'],
      quality_trends: [`${(result.success_rate * 100).toFixed(1)}% success rate`]
    },
    recommendations: [
      'Monitor rate limits to reduce errors',
      'Consider exponential backoff for retries'
    ],
    memory_suggestions: [
      {
        type: 'project',
        content: 'Workflow uses parallel processing pattern',
        priority: 'medium'
      }
    ]
  };

  const learningId = await storeLearnings(
    `manual-${Date.now()}`,
    manualLearnings,
    {
      workflowName: meta.name,
      strategy: 'manual',
      qualityScore: result.success_rate
    }
  );

  return learningId;
}

// To use manual extraction instead:
// const learningId = await manualLearningExtraction();
// log(`✅ Manually stored learning #${learningId}`);
