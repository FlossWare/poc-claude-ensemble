// Legacy post-task analysis adapter.
//
// DISABLED: the previous implementation ran from UserPromptSubmit and directly
// mutated Thompson priors via learning/post_task_analyzer.py. Re-enable only
// after it delegates to the canonical Learning service with stable outcome IDs
// and evidence-gated learner updates. Prompt hooks must never cause learning.

const postTaskAnalysis = {
  name: "post-task-analysis",
  description: "Disabled legacy adapter; use the canonical Learning service",
  event: "WorkflowComplete",
  config: {
    enabled: false,
    priority: 100,
    timeout: 30000,
    runInBackground: true
  },
  async execute() {
    return {
      success: false,
      skipped: true,
      reason: "disabled_until_learning_service_delegation"
    };
  }
};

export default postTaskAnalysis;
