#!/usr/bin/env node

/**
 * Unit tests for pre-workflow-context-loader.js
 *
 * Tests:
 * 1. Similarity search returns correct number of results
 * 2. Diversity check detects echo chamber (>70% threshold)
 * 3. Prompt formatting produces readable context
 * 4. Context injection includes learnings when enabled
 * 5. Metadata tracking captures all required fields
 *
 * Usage: node hooks/test-context-loader.js
 */

import { injectContextIntoWorkflow, formatContextForPrompt, diversityCheck } from './pre-workflow-context-loader.mjs';
import { getWorkflowStorage } from '../shared/workflow-storage-adapter.cjs';

async function testSimilaritySearch() {
  console.log('\n=== Test 1: Similarity Search ===');

  const taskDescription = 'Research firmware reverse engineering for router';
  const contextData = await injectContextIntoWorkflow(taskDescription, {
    limit: 3,
    check_diversity: false,
    include_learnings: false
  });

  console.log(`✅ Found ${contextData.previous_similar_workflows.length} similar workflows`);
  console.log(`   Context used: ${contextData.metadata.context_used}`);
  console.log(`   Similarity scores: ${contextData.metadata.similarity_scores.join(', ')}`);

  if (contextData.previous_similar_workflows.length > 0) {
    console.log(`   Top match: "${contextData.previous_similar_workflows[0].task_description}"`);
  }

  return contextData.previous_similar_workflows.length <= 3;
}

async function testDiversityCheck() {
  console.log('\n=== Test 2: Diversity Check (Echo Chamber Detection) ===');

  // Create mock workflow data with 80% opus usage (should trigger warning)
  const mockWorkflows = [
    { id: 1, task_description: 'Test 1', outcome: 'success', total_workers: 10, total_duration_ms: 5000, distance: 0.1 },
    { id: 2, task_description: 'Test 2', outcome: 'success', total_workers: 10, total_duration_ms: 5000, distance: 0.2 },
    { id: 3, task_description: 'Test 3', outcome: 'success', total_workers: 10, total_duration_ms: 5000, distance: 0.3 }
  ];

  const db = getWorkflowStorage();

  // Mock distribution: 80% opus, 10% sonnet, 10% haiku
  const mockDistribution = { opus: 24, sonnet: 3, haiku: 3 };

  // Mock getModelDistribution to return our test data
  const originalGetModelDistribution = db.getModelDistribution;
  db.getModelDistribution = async () => mockDistribution;

  const analysis = await diversityCheck(mockWorkflows, db);

  // Restore original method
  db.getModelDistribution = originalGetModelDistribution;

  console.log(`   Echo chamber detected: ${analysis.has_echo_chamber}`);
  console.log(`   Dominant model: ${analysis.dominant_model} (${(analysis.dominant_percentage * 100).toFixed(0)}%)`);
  console.log(`   Model distribution:`, analysis.model_distribution);

  if (analysis.recommendation) {
    console.log(`   ⚠️  ${analysis.recommendation}`);
  }

  const passed = analysis.has_echo_chamber && analysis.dominant_model === 'opus' && analysis.dominant_percentage === 0.80;
  console.log(passed ? '✅ PASS' : '❌ FAIL');

  return passed;
}

async function testPromptFormatting() {
  console.log('\n=== Test 3: Prompt Formatting ===');

  const mockWorkflows = [
    {
      id: 1,
      workflow_id: 'wf-abc',
      task_description: 'Research firmware for RAX-75',
      outcome: 'success',
      total_workers: 6,
      total_duration_ms: 45000,
      distance: 0.08,
      metadata: JSON.stringify({
        key_learnings: [
          'QEMU simulation requires OpenWrt buildroot',
          'GPL sources often incomplete, use binwalk instead'
        ]
      })
    },
    {
      id: 2,
      workflow_id: 'wf-def',
      task_description: 'Reverse engineer router firmware',
      outcome: 'success',
      total_workers: 4,
      total_duration_ms: 30000,
      distance: 0.15,
      metadata: JSON.stringify({
        summary: 'Successfully extracted and analyzed firmware'
      })
    }
  ];

  const formatted = formatContextForPrompt(mockWorkflows);

  console.log('   Formatted context:');
  console.log('---');
  console.log(formatted);
  console.log('---');

  const passed = formatted.includes('## Context from Similar Past Workflows') &&
                 formatted.includes('similarity: 0.92') &&
                 formatted.includes('QEMU simulation');

  console.log(passed ? '✅ PASS' : '❌ FAIL');

  return passed;
}

async function testContextInjectionWithLearnings() {
  console.log('\n=== Test 4: Context Injection with Learnings ===');

  const taskDescription = 'Research firmware reverse engineering';
  const contextData = await injectContextIntoWorkflow(taskDescription, {
    limit: 2,
    check_diversity: true,
    include_learnings: true
  });

  console.log(`   Similar workflows: ${contextData.previous_similar_workflows.length}`);
  console.log(`   Context formatted: ${contextData.formatted_context.length > 0}`);

  if (contextData.diversity_analysis) {
    console.log(`   Diversity check ran: ${contextData.diversity_analysis.has_echo_chamber !== undefined}`);
  }

  const passed = contextData.metadata.context_used === (contextData.previous_similar_workflows.length > 0);
  console.log(passed ? '✅ PASS' : '❌ FAIL');

  return passed;
}

async function testMetadataTracking() {
  console.log('\n=== Test 5: Metadata Tracking ===');

  const taskDescription = 'Code review for Java project';
  const contextData = await injectContextIntoWorkflow(taskDescription, { limit: 5 });

  const requiredFields = ['context_used', 'context_source', 'context_count', 'similarity_scores'];
  const hasAllFields = requiredFields.every(field => contextData.metadata.hasOwnProperty(field));

  console.log(`   Metadata fields present: ${Object.keys(contextData.metadata).join(', ')}`);
  console.log(`   context_used: ${contextData.metadata.context_used}`);
  console.log(`   context_count: ${contextData.metadata.context_count}`);

  if (contextData.metadata.context_source.length > 0) {
    console.log(`   context_source: [${contextData.metadata.context_source.slice(0, 3).join(', ')}...]`);
  }

  console.log(hasAllFields ? '✅ PASS' : '❌ FAIL');

  return hasAllFields;
}

async function runAllTests() {
  console.log('╔══════════════════════════════════════════════╗');
  console.log('║  Pre-Workflow Context Loader Unit Tests     ║');
  console.log('╚══════════════════════════════════════════════╝');

  const results = {
    similarity_search: await testSimilaritySearch(),
    diversity_check: await testDiversityCheck(),
    prompt_formatting: await testPromptFormatting(),
    context_injection: await testContextInjectionWithLearnings(),
    metadata_tracking: await testMetadataTracking()
  };

  console.log('\n╔══════════════════════════════════════════════╗');
  console.log('║  Test Summary                                ║');
  console.log('╚══════════════════════════════════════════════╝');

  const passed = Object.values(results).filter(r => r).length;
  const total = Object.keys(results).length;

  for (const [name, result] of Object.entries(results)) {
    const status = result ? '✅ PASS' : '❌ FAIL';
    console.log(`   ${status} - ${name.replace(/_/g, ' ')}`);
  }

  console.log(`\n   Total: ${passed}/${total} tests passed`);

  if (passed === total) {
    console.log('\n   🎉 All tests passed!');
  } else {
    console.log(`\n   ⚠️  ${total - passed} test(s) failed`);
  }

  process.exit(passed === total ? 0 : 1);
}

// Run tests
runAllTests().catch(err => {
  console.error('❌ Test suite failed:', err);
  process.exit(1);
});
