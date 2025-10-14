# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in CoRAG, please report it by emailing
mhd.ayham.joumran@studium.uni-hamburg.de.

**Please do not report security vulnerabilities through public GitHub issues.**

When reporting a vulnerability, please include:

* A description of the vulnerability
* Steps to reproduce the issue
* Potential impact
* Suggested fix (if available)

We will acknowledge receipt of your vulnerability report within 48 hours and
will send you regular updates about our progress. If the issue is confirmed,
we will release a patch as soon as possible.

## Security Best Practices

When using CoRAG:

* Never commit API keys or credentials to version control
* Use environment variables or secure secret management for sensitive data
* Keep dependencies up to date
* Review and sanitize any user-provided input
* Use HTTPS for API endpoints in production
* Implement rate limiting on public-facing services
