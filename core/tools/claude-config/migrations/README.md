# Migrations

Claude configuration integration versions use explicit migrations.

The current integration is 0.2. A future release that changes the manifest or managed configuration should add a migration script named:

  FROM-to-TO.sh

Migrations must be idempotent and must not overwrite foreign configuration.
