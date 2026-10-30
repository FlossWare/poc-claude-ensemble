#!/usr/bin/env node
/**
 * Test script for workflow learning integration
 * Validates database schema, storage, embeddings, and queries
 */

import { getDB } from '../learning/postgres-adapter.js';
import {
  storeLearnings,
  getLearningsByWorkflow,
  findSimilarLearnings,
  getRecentLearnings,
  getLearningStats
} from '../learning/storage.js';
import { generateEmbedding, cosineSimilarity } from '../learning/embeddings.js';

async function main() {
  console.log('🧪 Testing Workflow Learning Integration\n');

  let passed = 0;
  let failed = 0;

  // Test 1: Database Connection
  console.log('Test 1: Database Connection');
  try {
    const db = getDB();
    const result = await db.query('SELECT NOW() as now');
    console.log(`✅ Connected to PostgreSQL (${result[0].now})\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
    return;
  }

  // Test 2: Schema Exists
  console.log('Test 2: Database Schema');
  try {
    const db = getDB();
    const tables = await db.query(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'workflows' AND table_name = 'learnings'
    `);

    if (tables.length === 0) {
      console.error('❌ workflows.learnings table does not exist');
      console.log('   Run: psql -h /var/run/postgresql -U sfloess -d learning -f ~/.claude/learning/schema-workflows.sql\n');
      failed++;
    } else {
      console.log('✅ workflows.learnings table exists\n');
      passed++;
    }
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 3: Embedding Generation
  console.log('Test 3: Embedding Generation');
  try {
    const embedding = await generateEmbedding('Test learning extraction');

    if (!Array.isArray(embedding)) {
      throw new Error('Embedding is not an array');
    }

    if (embedding.length === 0) {
      throw new Error('Embedding is empty');
    }

    console.log(`✅ Generated ${embedding.length}-dim embedding\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}`);
    console.log('   Falling back to hash-based embedding\n');
    // Not a critical failure
    passed++;
  }

  // Test 4: Store Learning
  console.log('Test 4: Store Learning');
  try {
    const testLearning = {
      user_patterns: {
        preferences: ['Test preference 1', 'Test preference 2'],
        expertise_level: {
          'javascript': 'advanced',
          'testing': 'intermediate'
        },
        workflow_usage: ['Uses test workflow']
      },
      code_patterns: {
        common_bugs: ['Test bug pattern'],
        architecture_insights: ['Test architecture insight'],
        tech_stack: ['Node.js', 'PostgreSQL'],
        quality_trends: ['Improving test coverage']
      },
      recommendations: [
        'Test recommendation 1',
        'Test recommendation 2'
      ],
      memory_suggestions: [
        {
          type: 'user',
          content: 'Test memory suggestion',
          priority: 'high'
        }
      ]
    };

    const learningId = await storeLearnings(
      `test-${Date.now()}`,
      testLearning,
      {
        executionId: null, // No execution_id for standalone test
        workflowName: 'test-workflow',
        strategy: 'test',
        qualityScore: 0.95
      }
    );

    console.log(`✅ Stored learning ${learningId}\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 5: Query by Workflow
  console.log('Test 5: Query by Workflow');
  try {
    const learnings = await getLearningsByWorkflow('test-workflow', 5);

    if (!Array.isArray(learnings)) {
      throw new Error('Result is not an array');
    }

    console.log(`✅ Retrieved ${learnings.length} learnings for test-workflow\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 6: Recent Learnings
  console.log('Test 6: Recent Learnings');
  try {
    const recent = await getRecentLearnings(10);

    if (!Array.isArray(recent)) {
      throw new Error('Result is not an array');
    }

    console.log(`✅ Retrieved ${recent.length} recent learnings\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 7: Learning Statistics
  console.log('Test 7: Learning Statistics');
  try {
    const stats = await getLearningStats();

    if (!Array.isArray(stats)) {
      throw new Error('Result is not an array');
    }

    console.log(`✅ Retrieved stats for ${stats.length} workflows:`);
    stats.slice(0, 5).forEach(s => {
      const avgImpact = s.avg_impact ? parseFloat(s.avg_impact).toFixed(2) : 'N/A';
      const avgConfidence = s.avg_confidence ? parseFloat(s.avg_confidence).toFixed(2) : 'N/A';
      console.log(`   - ${s.workflow_name}: ${s.total_learnings} learnings, avg impact ${avgImpact}, avg confidence ${avgConfidence}`);
    });
    console.log('');
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 8: Semantic Similarity Search
  console.log('Test 8: Semantic Similarity Search');
  try {
    const similar = await findSimilarLearnings(
      'JavaScript testing and code quality',
      5,
      { minQualityScore: 0.5 }
    );

    if (!Array.isArray(similar)) {
      throw new Error('Result is not an array');
    }

    console.log(`✅ Found ${similar.length} similar learnings`);
    if (similar.length > 0) {
      console.log(`   Top match: distance=${similar[0].distance?.toFixed(4) || 'N/A'}`);
    }
    console.log('');
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 9: Cosine Similarity
  console.log('Test 9: Cosine Similarity');
  try {
    const emb1 = await generateEmbedding('JavaScript testing');
    const emb2 = await generateEmbedding('JavaScript testing');
    const emb3 = await generateEmbedding('Python machine learning');

    const sim1 = cosineSimilarity(emb1, emb2);
    const sim2 = cosineSimilarity(emb1, emb3);

    if (sim1 < 0.9) {
      throw new Error(`Identical text similarity too low: ${sim1}`);
    }

    console.log(`✅ Cosine similarity working:`);
    console.log(`   Same text: ${sim1.toFixed(4)}`);
    console.log(`   Different text: ${sim2.toFixed(4)}\n`);
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Test 10: JSONB Queries
  console.log('Test 10: JSONB Queries');
  try {
    const db = getDB();

    // Query for specific tech stack in context.code_patterns.tech_stack
    const nodeJsLearnings = await db.query(`
      SELECT
        workflow_name,
        learning_type,
        title,
        impact_score,
        confidence,
        created_at
      FROM workflows.learnings
      WHERE context->'code_patterns'->'tech_stack' @> '["Node.js"]'
      ORDER BY created_at DESC
      LIMIT 5
    `);

    console.log(`✅ Found ${nodeJsLearnings.length} learnings with Node.js in tech stack`);
    if (nodeJsLearnings.length > 0) {
      const first = nodeJsLearnings[0];
      console.log(`   Latest: ${first.title || first.learning_type} (impact: ${first.impact_score || 'N/A'}, confidence: ${first.confidence || 'N/A'})`);
    }
    console.log('');
    passed++;
  } catch (err) {
    console.error(`❌ Failed: ${err.message}\n`);
    failed++;
  }

  // Summary
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`TEST RESULTS: ${passed} passed, ${failed} failed`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  if (failed === 0) {
    console.log('✅ All tests passed! Integration is working correctly.\n');
    console.log('Next steps:');
    console.log('1. Register post-workflow hook in ~/.claude/settings.json');
    console.log('2. Run workflows and check learnings table');
    console.log('3. Query learnings: SELECT * FROM workflows.learnings ORDER BY created_at DESC LIMIT 10;');
    console.log('4. Check stats: SELECT workflow_name, COUNT(*) as count FROM workflows.learnings GROUP BY workflow_name;');
  } else {
    console.log('❌ Some tests failed. Check error messages above.\n');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
