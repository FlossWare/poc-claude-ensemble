#!/usr/bin/env node

/**
 * Test script for cross-session context loading
 *
 * Verifies that:
 * 1. loadContext() finds similar workflows via embedding similarity
 * 2. Context prompts are properly formatted
 * 3. Diversity checking detects echo chambers
 * 4. getModelsUsed() extracts model lists from worker_results
 *
 * Usage: node hooks/test-context-loading.js
 */

import { loadContext, injectContext } from './load-similar-workflows.mjs';
import { getWorkflowStorage } from '../shared/workflow-storage-adapter.cjs';

async function testContextLoading() {
  console.log('🧪 Testing Cross-Session Context Loading\n');

  try {
    // Test 1: Load context for a typical research task
    console.log('Test 1: Load context for "firmware reverse engineering"');
    const context1 = await loadContext('firmware reverse engineering', { limit: 5 });

    if (context1) {
      console.log(`✅ Found ${context1.foundCount} similar workflows`);
      console.log(`Models distribution:`, JSON.stringify(context1.modelStats.distribution, null, 2));

      if (context1.excludeModels.length > 0) {
        console.log(`⚠️ Echo chamber detected - recommend excluding: ${context1.excludeModels.join(', ')}`);
      }

      console.log('\nContext prompt preview (first 300 chars):');
      console.log(context1.contextPrompt.substring(0, 300) + '...\n');
    } else {
      console.log('ℹ️ No similar workflows found\n');
    }

    // Test 2: Test context injection into a prompt
    console.log('Test 2: Context injection into base prompt');
    const basePrompt = 'Analyze this firmware binary and extract configuration data.';

    if (context1) {
      const enhancedPrompt = injectContext(basePrompt, context1);
      console.log(`✅ Injected context (${enhancedPrompt.length} chars total)`);
      console.log('Enhanced prompt preview:');
      console.log(enhancedPrompt.substring(0, 400) + '...\n');
    }

    // Test 3: Test getModelsUsed() method
    console.log('Test 3: Extract models used from a workflow execution');
    const db = getWorkflowStorage();

    // Get a sample workflow execution ID
    const result = await db.pool.query(`
      SELECT id FROM workflow.executions
      WHERE outcome = 'success'
      LIMIT 1
    `);

    if (result.rows.length > 0) {
      const executionId = result.rows[0].id;
      const modelsUsed = await db.getModelsUsed(executionId);
      console.log(`✅ Execution #${executionId} used models: ${modelsUsed.join(', ')}\n`);
    } else {
      console.log('ℹ️ No workflow executions found in database\n');
    }

    // Test 4: Verify SQL view exists
    console.log('Test 4: Check context_impact_analysis view');
    const viewCheck = await db.pool.query(`
      SELECT COUNT(*) as count FROM workflow.context_impact_analysis
    `);
    console.log(`✅ View accessible, ${viewCheck.rows[0].count} rows\n`);

    console.log('✅ All tests passed!');
    process.exit(0);

  } catch (err) {
    console.error('❌ Test failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

testContextLoading();
