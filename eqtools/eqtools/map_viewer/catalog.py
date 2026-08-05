"""Layer catalog and parsed-payload cache for the ECAT map viewer."""

from threading import RLock

from .cache import ParsedLayerCache
from .loaders import LOADER_VERSION, load_layer, source_fingerprint


class LayerCatalog:
    """Read-only layer declarations plus lazy payload loading.

    Layer declarations are immutable. Parsed payloads are shared read-only
    across sessions, while visibility, filters, and viewport remain browser
    state.
    """

    def __init__(self, layers, *, cache=None):
        layers = tuple(layers)
        self._layers = {layer.id: layer for layer in layers}
        if len(self._layers) != len(layers):
            raise ValueError("LayerCatalog requires unique layer ids.")
        self.cache = cache or ParsedLayerCache()
        self._load_counts = {layer_id: 0 for layer_id in self._layers}
        self._load_lock = RLock()

    @property
    def layers(self):
        """Ordered immutable layer declarations."""

        return tuple(self._layers.values())

    def get(self, layer_id):
        """Return one declaration by stable layer id."""

        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise KeyError(f"Unknown viewer layer id: {layer_id!r}.") from exc

    def load(self, layer_id):
        """Load once per source fingerprint and return a cached payload."""

        with self._load_lock:
            spec = self.get(layer_id)
            fingerprint = source_fingerprint(spec)
            key = (
                spec.id,
                fingerprint,
                spec.kind,
                spec.variable,
                spec.mask,
                spec.format,
                spec.data_type,
                LOADER_VERSION,
            )
            payload = self.cache.get(key)
            if payload is not None:
                return payload
            payload = load_layer(spec)
            self.cache.put(key, payload)
            self._load_counts[layer_id] += 1
            return payload

    def load_count(self, layer_id):
        """Number of real parses, excluding cache hits."""

        self.get(layer_id)
        return self._load_counts[layer_id]


__all__ = ["LayerCatalog"]
