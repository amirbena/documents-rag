"""CORS policy for frontend integration (Phase 2.10).

`allow_origins` is the only thing an operator controls, via `CORS_ALLOW_ORIGINS`
(`Settings.cors_allow_origins_list`) — empty by default, so no cross-origin request is permitted
until a frontend origin is named explicitly. Everything else below is a fixed, non-speculative
policy, not configuration surface:

- `allow_credentials=False`: this backend has no cookie/session/bearer-token authentication, so
  there is no concrete requirement for credentialed cross-origin requests. Keeping it False also
  keeps a wildcard `CORS_ALLOW_ORIGINS=*` (if an operator ever sets one) safe — the CORS spec
  forbids combining a wildcard origin with credentialed requests, and Starlette's own
  `CORSMiddleware` reflects a literal `*` back as `Access-Control-Allow-Origin: *` (never a
  specific origin, never `Access-Control-Allow-Credentials`) exactly when credentials are off.
- `allow_methods`: exactly the HTTP verbs this API's routes use today (GET/POST/DELETE) — no
  speculative PUT/PATCH for endpoints that don't exist.
- `expose_headers=[CORRELATION_ID_HEADER]`: makes the already-echoed `X-Correlation-ID` response
  header (see `app/core/correlation.py`) readable by a browser-based frontend's own JS — without
  this, browsers hide all non-safelisted response headers from cross-origin script access
  regardless of the header being present on the wire.
"""

from typing import TypedDict

from app.core.config import Settings
from app.core.correlation import CORRELATION_ID_HEADER

CORS_ALLOW_METHODS = ["GET", "POST", "DELETE"]
CORS_EXPOSE_HEADERS = [CORRELATION_ID_HEADER]


class CorsMiddlewareKwargs(TypedDict):
    """Precisely-typed kwargs for `CORSMiddleware`, so they can be passed with `**` under mypy."""

    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    expose_headers: list[str]


def cors_middleware_kwargs(settings: Settings) -> CorsMiddlewareKwargs:
    """Return the kwargs for `CORSMiddleware` given `settings` — see module docstring for policy."""
    return {
        "allow_origins": settings.cors_allow_origins_list,
        "allow_credentials": False,
        "allow_methods": CORS_ALLOW_METHODS,
        "expose_headers": CORS_EXPOSE_HEADERS,
    }
