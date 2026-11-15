# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public GitHub issues or pull
requests.

Use GitHub's **Report a vulnerability** workflow in the repository's
**Security** tab when private vulnerability reporting is available. If that
workflow is unavailable, contact the repository maintainer through their
GitHub profile and request a private disclosure channel. Include the affected
commit or release, reproduction steps, impact, and any relevant logs with
secrets and personal data removed.

Please allow maintainers reasonable time to investigate and prepare a fix
before publicly disclosing a vulnerability. Do not include credentials,
tokens, private documents, or production ledger contents in a report.

## Supported code

This is a proof-of-concept repository. Security fixes are made against the
default branch; there is no promise of maintenance for archived code or old
snapshots. The former implementation under `archive/legacy-core/` is
historical reference material and is not part of the active service surface.

## Deployment threat model

The POC is intended for local development and trusted, operator-controlled
environments:

- Services bind to loopback by default. Do not assume that loopback binding
  provides authentication or protects against other local processes running
  as the same user.
- The gateway can optionally bind to a home LAN interface when explicitly
  configured with an allowed-network list. That mode is intended only for a
  trusted private LAN with a host firewall restricting access to the intended
  subnet.
- The POC is not designed to be exposed directly to the public internet.
  Do not port-forward it, publish it behind an unrestricted reverse proxy, or
  treat the optional LAN allowlist as a substitute for authentication.
- There is no general multi-user identity or authorization boundary. Anyone
  who can reach an exposed service may be able to invoke its available API.
- Memory and ledger data are stored locally. Protect the host, filesystem,
  backups, and any imported documents as sensitive data. Do not attach real
  secrets or private production data to public bug reports.

If the deployment is changed to accept untrusted network clients, add an
appropriate authentication, authorization, transport-security, and request
hardening layer before exposing it.
