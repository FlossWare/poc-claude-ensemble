import test from 'node:test'
import assert from 'node:assert/strict'

const productionModules = [
  '../shared/caching-bridge.js',
  '../shared/compression-bridge.js',
  '../shared/model-config-loader.js',
  '../tools/code-pr-review.js',
  '../tools/code-doc.js',
]

for (const modulePath of productionModules) {
  test(`imports ${modulePath}`, async () => {
    await assert.doesNotReject(() => import(modulePath))
  })
}
