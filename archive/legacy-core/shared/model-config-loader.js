/**
 * Model Configuration Loader
 * Reads ~/.claude/toolkit-models.yaml to get available models
 * and validates the shared models/skill_defaults schema.
 */

import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import yaml from 'js-yaml'

const DEFAULT_MODEL_CONFIG_PATH = path.join(
  os.homedir(),
  '.claude',
  'toolkit-models.yaml'
)

class ModelConfigValidationError extends Error {
  constructor(message) {
    super(message)
    this.name = 'ModelConfigValidationError'
  }
}

function assertObject(value, pathName) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ModelConfigValidationError(
      `Invalid configuration: ${pathName} must be an object`
    )
  }
}

function validateModelConfig(config) {
  assertObject(config, 'root')
  assertObject(config.models, 'models')
  assertObject(config.skill_defaults, 'skill_defaults')

  for (const [modelName, model] of Object.entries(config.models)) {
    assertObject(model, `models.${modelName}`)
    if (typeof model.available !== 'boolean') {
      throw new ModelConfigValidationError(
        `Invalid configuration: models.${modelName}.available must be a boolean`
      )
    }
  }

  for (const [skillName, skillConfig] of Object.entries(config.skill_defaults)) {
    assertObject(skillConfig, `skill_defaults.${skillName}`)

    if (!Array.isArray(skillConfig.models) || skillConfig.models.length === 0) {
      throw new ModelConfigValidationError(
        `Invalid configuration: skill_defaults.${skillName}.models must be a non-empty array`
      )
    }

    const duplicateModels = skillConfig.models.filter(
      (modelName, index) => skillConfig.models.indexOf(modelName) !== index
    )
    if (duplicateModels.length > 0) {
      throw new ModelConfigValidationError(
        `Invalid configuration: skill_defaults.${skillName}.models contains duplicate model names: ${[...new Set(duplicateModels)].join(', ')}`
      )
    }

    for (const modelName of skillConfig.models) {
      if (typeof modelName !== 'string' || !config.models[modelName]) {
        throw new ModelConfigValidationError(
          `Invalid configuration: skill_defaults.${skillName}.models references unknown model '${modelName}'`
        )
      }
    }

    if (typeof skillConfig.arbiter !== 'string' || skillConfig.arbiter.length === 0) {
      throw new ModelConfigValidationError(
        `Invalid configuration: skill_defaults.${skillName}.arbiter must name a model`
      )
    }

    if (!config.models[skillConfig.arbiter]) {
      throw new ModelConfigValidationError(
        `Invalid configuration: skill_defaults.${skillName}.arbiter references unknown model '${skillConfig.arbiter}'`
      )
    }

    if (config.models[skillConfig.arbiter].available !== true) {
      throw new ModelConfigValidationError(
        `Invalid configuration: skill_defaults.${skillName}.arbiter references unavailable model '${skillConfig.arbiter}'`
      )
    }
  }

  return config
}

function loadUserModelConfig(configPath = DEFAULT_MODEL_CONFIG_PATH) {
  try {
    if (!fs.existsSync(configPath)) {
      console.warn(
        `[ModelConfig] Config not found at ${configPath}, using defaults`
      )
      return null
    }

    const content = fs.readFileSync(configPath, 'utf8')
    const config = yaml.load(content)
    const validatedConfig = validateModelConfig(config)

    console.log(`[ModelConfig] Loaded from ${configPath}`)
    return validatedConfig
  } catch (err) {
    if (err instanceof ModelConfigValidationError) {
      throw err
    }

    throw new ModelConfigValidationError(
      `Invalid configuration at ${configPath}: ${err.message}`
    )
  }
}

function getAvailableModels(config) {
  if (!config || !config.models) return null

  return Object.entries(config.models)
    .filter(([name, model]) => model.available === true)
    .map(([name, model]) => ({
      name,
      id: model.id,
      provider: model.provider,
      pricing: model.pricing,
      capabilities: model.capabilities || []
    }))
}

function getModelPricing(config, modelId) {
  if (!config || !config.models) return null

  for (const [name, model] of Object.entries(config.models)) {
    if (model.id === modelId && model.available) {
      return {
        model: modelId,
        input_per_m: model.pricing?.input_per_m || 0,
        output_per_m: model.pricing?.output_per_m || 0
      }
    }
  }

  return null
}

function getSkillModels(config, skillName) {
  if (!config || !config.skill_defaults || !config.skill_defaults[skillName]) {
    return null
  }

  const skillConfig = config.skill_defaults[skillName]
  const availableModels = getAvailableModels(config)

  if (!availableModels) return skillConfig.models

  return skillConfig.models.filter(modelName => {
    const model = config.models[modelName]
    return model && model.available === true
  })
}

function getSkillArbiter(config, skillName) {
  if (!config || !config.skill_defaults || !config.skill_defaults[skillName]) {
    return null
  }

  return config.skill_defaults[skillName].arbiter
}

function calculateCostFromConfig(config, model, inputTokens, outputTokens) {
  if (!config || !config.models) return 0

  for (const [name, modelCfg] of Object.entries(config.models)) {
    if ((modelCfg.id === model || name === model) && modelCfg.available) {
      const pricing = modelCfg.pricing
      const inputCost = (inputTokens / 1_000_000) * (pricing.input_per_m || 0)
      const outputCost = (outputTokens / 1_000_000) * (pricing.output_per_m || 0)
      return inputCost + outputCost
    }
  }

  return 0
}

export {
  DEFAULT_MODEL_CONFIG_PATH,
  ModelConfigValidationError,
  validateModelConfig,
  loadUserModelConfig,
  getAvailableModels,
  getModelPricing,
  getSkillModels,
  getSkillArbiter,
  calculateCostFromConfig
}
