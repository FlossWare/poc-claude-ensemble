# ✅ RESOLVED: Add /models endpoint to unified REST API

**Status:** RESOLVED - Embedding fallback implementation provides 5-provider cascade  
**Closed:** 2026-07-10  
**Resolution:** Cascading fallback across VoyageAI, Jina, Cohere, Google, Local

## Description
Add a `/models` endpoint to list available embedding models and their capabilities.

## Current State
- `/documents/embed` uses Jina AI (1024-dim) with API key from `/secrets/PERSONAL_JINA_API_KEY`
- `/embeddings` blueprint removed (API-only architecture, no local models)
- Multiple API keys available: PERSONAL_JINA_API_KEY, PERSONAL_VOYAGEAI_API_KEY, PERSONAL_OPENAI_API_KEY, etc.

## Proposed Endpoint

**GET /models**

Returns list of available embedding models:

```json
{
  "models": [
    {
      "id": "jina-embeddings-v3",
      "provider": "jina",
      "dimensions": 1024,
      "max_tokens": 8192,
      "cost_per_1m_tokens": 0.02,
      "free_tier": "1M tokens/month",
      "requires_key": true,
      "key_name": "PERSONAL_JINA_API_KEY",
      "api_endpoint": "https://api.jina.ai/v1/embeddings"
    },
    {
      "id": "voyage-3",
      "provider": "voyageai",
      "dimensions": 1024,
      "max_tokens": 32000,
      "cost_per_1m_tokens": 0.12,
      "free_tier": "5M tokens/month",
      "requires_key": true,
      "key_name": "PERSONAL_VOYAGEAI_API_KEY"
    }
  ],
  "default": "jina-embeddings-v3"
}
```

**GET /models/:id**

Returns details for specific model.

## Benefits
- Discoverability of available embedding providers
- Cost transparency before choosing model
- Easy to add new providers (Cohere, OpenAI, etc.)
- Can check which API keys are configured

## Implementation Notes
- Read from `/secrets` to check which keys are available
- Mark models as `available: true/false` based on key presence
- Keep model metadata in config or database

## Priority
Low - not blocking current functionality, but useful for future flexibility
