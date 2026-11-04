# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private reporting: **Security → Report a vulnerability** on this repository
(<https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability>).
Include affected version/commit, reproduction steps and the impact you see.

You can expect an acknowledgement within a few days. Fixes are released as soon as practical and
credited in the advisory unless you prefer otherwise.

## Scope

In scope: authentication/authorisation, CSRF, image delivery (path handling), ingestion
(ffmpeg/Pillow handling of untrusted files), export, dependency and CI/CD configuration.

Out of scope: vulnerabilities in your own OIDC provider, reverse proxy or host configuration.

## Data

This repository contains **code only**. If you ever find real apartment photos, databases,
tokens or credentials in the repository or its history, report it the same way.
