# Security Policy

## Scope

This repository is a proof of concept. The HTTP gateway defaults to loopback-only.
LAN access is opt-in and requires an explicit client CIDR allowlist and
`POC_CLAUDE_ENSEMBLE_AUTH_TOKEN`. Set `POC_CLAUDE_ENSEMBLE_ALLOWED_HOSTS` to a
comma-separated list of trusted DNS hostnames when clients access the gateway
by hostname; otherwise Host validation may reject those requests. Host and
Origin validation are applied at the gateway boundary.

Do not expose the gateway directly to the public internet or configure router
port forwarding for it. The gateway's HTTP transport is not encrypted; use it
only on a trusted network or behind a properly configured TLS-terminating
boundary. Keep the host firewall restricted to the intended client networks.

The services may persist prompts, source material, model outputs, operational
events, identifiers, and diagnostic metadata. Treat their data directories,
ledger files, logs, and backups as sensitive. Do not commit credentials,
tokens, private source material, or production data.

## Reporting a vulnerability

Please do not report suspected vulnerabilities in public issues or discussions.

If GitHub private vulnerability reporting is enabled for this repository, use
the repository's **Report a vulnerability** action to submit a private
advisory. If that action is unavailable, contact a repository maintainer
privately through the FlossWare GitHub organization and request a secure
reporting channel before sharing technical details.

Include the affected commit or branch, impacted component, reproduction steps,
and potential impact. Redact tokens, credentials, personal data, and customer
content from reports.

## Supported versions

Security fixes are applied to the current default branch. This POC does not
provide a formal long-term support policy or production security guarantee.
