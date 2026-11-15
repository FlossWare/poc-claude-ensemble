/**
 * Direct learning extraction without workflow orchestration
 * Used by post-workflow hook for fast learning extraction
 */

/**
 * Extract learnings from workflow result
 * This is a simplified version that uses heuristics instead of AI models
 * For full AI-powered extraction, use the ai-extract-learning workflow
 *
 * @param {string} workflowName - Name of workflow
 * @param {object} executionData - Execution data
 * @returns {Promise<object>} Learning object
 */
async function extractLearningsFromResult(workflowName, executionData) {
  // For hook-based extraction, we use heuristic patterns
  // This is faster but less sophisticated than AI extraction

  const learnings = {
    user_patterns: {
      preferences: [],
      expertise_level: {},
      workflow_usage: []
    },
    code_patterns: {
      common_bugs: [],
      architecture_insights: [],
      tech_stack: [],
      quality_trends: []
    },
    recommendations: [],
    memory_suggestions: []
  };

  // Extract workflow-specific patterns
  switch (workflowName) {
    case 'code-solve':
      extractCodeSolvePatterns(executionData, learnings);
      break;

    case 'code-review':
      extractCodeReviewPatterns(executionData, learnings);
      break;

    case 'code-test':
      extractCodeTestPatterns(executionData, learnings);
      break;

    case 'code-security':
      extractSecurityPatterns(executionData, learnings);
      break;

    default:
      extractGenericPatterns(executionData, learnings);
  }

  // Only return if we found meaningful patterns
  if (hasLearnings(learnings)) {
    return learnings;
  }

  return null;
}

/**
 * Extract patterns from code-solve workflow
 */
function extractCodeSolvePatterns(data, learnings) {
  if (data.issue_title) {
    learnings.code_patterns.common_bugs.push(`Issue: ${data.issue_title}`);
  }

  if (data.fix_approach) {
    learnings.recommendations.push(`Successful fix approach: ${data.fix_approach}`);
  }

  if (data.files_changed?.length > 0) {
    const fileTypes = data.files_changed
      .map(f => f.split('.').pop())
      .filter((v, i, a) => a.indexOf(v) === i);

    learnings.code_patterns.tech_stack.push(...fileTypes.map(ext => `.${ext} files`));
  }

  if (data.consensus_score > 0.8) {
    learnings.recommendations.push('High consensus indicates clear fix strategy');
  }
}

/**
 * Extract patterns from code-review workflow
 */
function extractCodeReviewPatterns(data, learnings) {
  if (data.findings_count > 0) {
    learnings.code_patterns.quality_trends.push(
      `Found ${data.findings_count} issues in review`
    );
  }

  if (data.severity_breakdown) {
    Object.entries(data.severity_breakdown).forEach(([severity, count]) => {
      if (count > 0) {
        learnings.code_patterns.common_bugs.push(
          `${count} ${severity}-severity issues`
        );
      }
    });
  }

  if (data.categories?.length > 0) {
    data.categories.forEach(cat => {
      learnings.recommendations.push(`Review focus needed: ${cat}`);
    });
  }
}

/**
 * Extract patterns from code-test workflow
 */
function extractCodeTestPatterns(data, learnings) {
  if (data.pass_rate !== undefined) {
    if (data.pass_rate < 80) {
      learnings.code_patterns.quality_trends.push(
        `Low test pass rate: ${data.pass_rate}%`
      );
      learnings.recommendations.push('Improve test reliability');
    } else if (data.pass_rate >= 95) {
      learnings.code_patterns.quality_trends.push('High test reliability');
    }
  }

  if (data.failures?.length > 0) {
    const failedTests = data.failures.slice(0, 3).map(f => f.name || f);
    learnings.code_patterns.common_bugs.push(
      `Failing tests: ${failedTests.join(', ')}`
    );
  }

  if (data.build_status === 'failed') {
    learnings.recommendations.push('Fix build failures before deployment');
  }
}

/**
 * Extract patterns from code-security workflow
 */
function extractSecurityPatterns(data, learnings) {
  if (data.vulnerabilities_count > 0) {
    learnings.code_patterns.common_bugs.push(
      `Found ${data.vulnerabilities_count} security vulnerabilities`
    );
  }

  if (data.vuln_categories?.length > 0) {
    data.vuln_categories.forEach(cat => {
      learnings.memory_suggestions.push({
        type: 'project',
        content: `Security issue category: ${cat}`,
        priority: 'high'
      });
    });
  }

  if (data.max_severity === 'critical' || data.max_severity === 'high') {
    learnings.recommendations.push('Address high-severity security issues immediately');
  }
}

/**
 * Extract generic patterns from any workflow
 */
function extractGenericPatterns(data, learnings) {
  // Quality score indicates workflow success
  if (data.quality_score !== undefined) {
    if (data.quality_score > 0.8) {
      learnings.code_patterns.quality_trends.push('High-quality execution');
    } else if (data.quality_score < 0.5) {
      learnings.recommendations.push('Review workflow configuration');
    }
  }

  // Track model performance
  if (data.model) {
    learnings.user_patterns.workflow_usage.push(
      `Used ${data.model} for task`
    );
  }

  // Duration insights
  if (data.duration_ms > 60000) {
    learnings.recommendations.push('Consider optimizing for performance');
  }
}

/**
 * Check if learnings object has any meaningful content
 */
function hasLearnings(learnings) {
  return (
    learnings.user_patterns.preferences.length > 0 ||
    Object.keys(learnings.user_patterns.expertise_level).length > 0 ||
    learnings.user_patterns.workflow_usage.length > 0 ||
    learnings.code_patterns.common_bugs.length > 0 ||
    learnings.code_patterns.architecture_insights.length > 0 ||
    learnings.code_patterns.tech_stack.length > 0 ||
    learnings.code_patterns.quality_trends.length > 0 ||
    learnings.recommendations.length > 0 ||
    learnings.memory_suggestions.length > 0
  );
}

module.exports = {
  extractLearningsFromResult
};
