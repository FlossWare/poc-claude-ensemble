# Patches for code that runs on the live system

`memory-service-session-events-ledger.patch` changes the memory service (core runtime, not in
this repository) so that the `session_events` store is written to the ledger root and each row
carries `schema_version`, `producer`, and `kind`. Apply it to the release copy of
`memory-service/memory_service.py`. The patch is a unified diff against the pre-ledger copy.
