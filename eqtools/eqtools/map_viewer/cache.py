"""Thread-safe, bounded caches for parsed viewer payloads."""

from collections import OrderedDict
from threading import RLock


class ParsedLayerCache:
    """Small catalog-owned LRU for detached parsed layer payloads.

    The cache only drops in-memory values. It never deletes, writes, or
    invalidates a scientific source file.
    """

    def __init__(self, max_entries=12):
        max_entries = int(max_entries)
        if max_entries < 1:
            raise ValueError("max_entries must be positive.")
        self.max_entries = max_entries
        self._values = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        """Return a cached payload and mark it as recently used."""

        with self._lock:
            value = self._values.pop(key, None)
            if value is not None:
                self._values[key] = value
            return value

    def put(self, key, value):
        """Store one payload and evict only the oldest in-memory entries."""

        with self._lock:
            self._values.pop(key, None)
            self._values[key] = value
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def clear(self):
        """Drop all parsed in-memory payloads."""

        with self._lock:
            self._values.clear()

    def __len__(self):
        with self._lock:
            return len(self._values)


__all__ = ["ParsedLayerCache"]
