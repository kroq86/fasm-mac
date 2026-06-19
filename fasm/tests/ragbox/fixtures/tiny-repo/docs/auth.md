# Authentication

This document describes JWT authentication and login middleware.

The auth middleware validates bearer tokens on every HTTP request.
Login endpoints issue signed JWT access tokens with configurable expiry.
Session refresh uses a separate refresh token stored in an HttpOnly cookie.

Authorization checks run after authentication and map roles to route policies.
Failed auth returns 401; failed authorization returns 403.

## JWT details

Tokens use HS256 by default. Rotate signing keys on a schedule.
Include `sub`, `iat`, and `exp` claims in every access token payload.

## Middleware chain

Request -> auth middleware -> authorization -> handler.

Repeat filler for chunk overlap testing: auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth auth
