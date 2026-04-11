from __future__ import annotations

import threading
from collections import defaultdict


class LockManager:
    """Per-series_id lock manager using threading.Lock."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._meta_lock = threading.Lock()

    def get_lock(self, series_id: str) -> threading.Lock:
        with self._meta_lock:
            return self._locks[series_id]
