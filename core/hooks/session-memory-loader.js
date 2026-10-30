#!/usr/bin/env node
/**
 * Session Memory Loader - Auto-load project memory on session start
 *
 * Automatically displays the MEMORY.md index at every Claude Code session start
 * so the session knows what context is available without asking.
 *
 * This is non-blocking and informational only.
 */

const fs = require('fs');
const path = require('path');

try {
  const memoryDir = path.join(process.env.HOME || '', '.claude/projects/memory');
  const memoryIndexPath = path.join(memoryDir, 'MEMORY.md');

  if (!fs.existsSync(memoryIndexPath)) {
    process.exit(0);
  }

  const memoryIndex = fs.readFileSync(memoryIndexPath, 'utf8');

  // Display memory index at session start
  console.error('\n📚 Project Memory Index:');
  console.error('─'.repeat(60));
  console.error(memoryIndex);
  console.error('─'.repeat(60));
  console.error('Use: mem_search <keyword> | mem_read <name> | Read tool on .md files\n');

  process.exit(0);
} catch (err) {
  // Silently fail - memory not available
  process.exit(0);
}
