# Memory Schema Compatibility

Memory Service persists execution-aware records containing a canonical `ExecutionContext`. The serialized context has an explicit schema version so persisted records can evolve without silently changing meaning.

## Version

The current `ExecutionContext` schema version is **1**.

Every context produced by `ExecutionContext.to_dict()` includes:

```json
{
  "schema_version": 1
}
```

The version describes the serialized context shape, not the runtime implementation version.

## Reader compatibility

### Legacy records

Records without `schema_version` are treated as **version 0** legacy records.

Version 0 is a compatibility shape, not a separately maintained modern schema. The reader accepts the legacy shape and normalizes it into the current in-memory representation. Re-serializing the result emits the current version.

### Current records

Version 1 records are read normally.

Unknown fields are ignored. This permits additive fields to be introduced without requiring every older reader to understand them.

### Newer records

A reader must reject a record whose `schema_version` is greater than the highest version it understands.

It must not silently discard or reinterpret a newer schema. The caller can then decide whether to preserve, migrate, retry, or quarantine the record.

Invalid version values, including booleans, non-integers, and negative integers, are rejected.

## Migration policy

Schema changes should prefer additive, backward-compatible fields.

When a change cannot be read safely by an older reader:

1. Increment the schema version.
2. Define the migration or compatibility boundary explicitly.
3. Update readers and tests before producing the new version in normal operation.
4. Reject newer versions rather than guessing at their meaning.

There is currently no automatic in-place migration of persisted JSONL records. Reading a legacy record and re-serializing it through the current contract is the supported normalization path.

## Compatibility contract

The contract is intentionally asymmetric:

- **Older data:** read when the current reader can interpret it.
- **Unknown fields:** ignore when the schema version is supported.
- **Newer schema versions:** reject explicitly.
- **Invalid schema versions:** reject explicitly.
- **Writer:** emits exactly the current schema version.

This keeps Memory persistence forward-safe without turning the service into a migration framework before one is actually needed.
