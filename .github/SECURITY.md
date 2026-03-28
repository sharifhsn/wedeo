# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in wedeo, please report it by
[opening a private security advisory](https://github.com/sharifhsn/wedeo/security/advisories/new).

Do not open a public issue for security vulnerabilities.

## Scope

wedeo is a media decoder. Security-relevant issues include:
- Buffer overflows or out-of-bounds reads from malformed input
- Panics or crashes from fuzzed/malicious media files
- Memory safety issues (any `unsafe` block)

## Response

I'll acknowledge reports within 72 hours and aim to fix confirmed
vulnerabilities within 2 weeks.
