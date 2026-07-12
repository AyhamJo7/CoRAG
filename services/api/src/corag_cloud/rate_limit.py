"""Per-tenant fixed-window rate limiting for the expensive ask paths.

In-process by design: the stack runs as a single API instance on the Pi.
If the API ever scales horizontally this moves to Postgres or Redis.
"""

import threading
import time
from uuid import UUID

from fastapi import HTTPException

WINDOW_SECONDS = 60.0


class FixedWindowLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._windows: dict[UUID, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, tenant_id: UUID) -> None:
        """Raise 429 when the tenant exceeds the per-minute ask limit."""
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            window_start, count = self._windows.get(tenant_id, (now, 0))
            if now - window_start >= WINDOW_SECONDS:
                window_start, count = now, 0
            if count >= self.limit:
                raise HTTPException(
                    status_code=429,
                    detail="Too many questions at once — try again in a minute",
                )
            self._windows[tenant_id] = (window_start, count + 1)
            # Keep the map from growing unboundedly.
            if len(self._windows) > 10_000:
                cutoff = now - WINDOW_SECONDS
                self._windows = {
                    k: v for k, v in self._windows.items() if v[0] > cutoff
                }
