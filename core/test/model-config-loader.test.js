import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import {
  DEFAULT_MODEL_CONFIG_PATH,
  ModelConfigValidationError,
  getSkillArbiter,
  getSkillModels,
  loadUserModelConfig
} from '../shared/model-config-loader.js'

function withTempConfig(content, callback) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-ensemble-'))
  const configPath = path.join(tempDir, 'models.yaml')

  try {
    fs.writeFileSync(configPath, content)
    return callback(configPath)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
}

test('default model config path uses the current home directory', () => {
  assert.equal(
    DEFAULT_MODEL_CONFIG_PATH,
    path.join(os.homedir(), '.claude', 'toolkit-models.yaml')
  )
})

test('explicit config path loads a valid models and skill_defaults configuration', () => {
  withTempConfig(
    `models:
  test-model:
    id: test-model
    available: true
skill_defaults:
  test-skill:
    models: [test-model]
    arbiter: test-model
`,
    configPath => {
      const config = loadUserModelConfig(configPath)
      assert.deepEqual(getSkillModels(config, 'test-skill'), ['test-model'])
      assert.equal(getSkillArbiter(config, 'test-skill'), 'test-model')
    }
  )
})

test('unavailable configured models are filtered from skill routing', () => {
  withTempConfig(
    `models:
  available-model:
    id: available-model
    available: true
  unavailable-model:
    id: unavailable-model
    available: false
skill_defaults:
  test-skill:
    models: [available-model, unavailable-model]
    arbiter: available-model
`,
    configPath => {
      const config = loadUserModelConfig(configPath)
      assert.deepEqual(getSkillModels(config, 'test-skill'), ['available-model'])
    }
  )
})

test('invalid skill model references fail with a useful error', () => {
  withTempConfig(
    `models:
  test-model:
    id: test-model
    available: true
skill_defaults:
  test-skill:
    models: [missing-model]
    arbiter: test-model
`,
    configPath => {
      assert.throws(
        () => loadUserModelConfig(configPath),
        error =>
          error instanceof ModelConfigValidationError &&
          error.message.includes(
            "skill_defaults.test-skill.models references unknown model 'missing-model'"
          )
      )
    }
  )
})

test('unavailable arbiter references fail validation', () => {
  withTempConfig(
    `models:
  worker-model:
    id: worker-model
    available: true
  arbiter-model:
    id: arbiter-model
    available: false
skill_defaults:
  test-skill:
    models: [worker-model]
    arbiter: arbiter-model
`,
    configPath => {
      assert.throws(
        () => loadUserModelConfig(configPath),
        error =>
          error instanceof ModelConfigValidationError &&
          error.message.includes(
            "skill_defaults.test-skill.arbiter references unavailable model 'arbiter-model'"
          )
      )
    }
  )
})

test('missing config returns the documented fallback', () => {
  const missingPath = path.join(os.tmpdir(), 'claude-ensemble-missing-models.yaml')
  assert.equal(loadUserModelConfig(missingPath), null)
})
