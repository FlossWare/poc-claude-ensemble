/**
 * Legacy post-workflow learning hook.
 *
 * DISABLED: this implementation writes directly through the local learning
 * storage module, bypassing the canonical Learning service boundary and
 * lacks an idempotent outcome key. Do not register it. Workflow outcomes must
 * be submitted to the Learning service, which owns deduplication, evaluation,
 * and learner delegation.
 */

export async function onWorkflowComplete() {
  return {
    status: "skipped",
    reason: "disabled_until_learning_service_delegation"
  };
}

export default { onWorkflowComplete };
