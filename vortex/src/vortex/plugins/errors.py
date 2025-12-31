"""Error types for Vortex plugin system."""


class PluginError(Exception):
    """Base error for plugin operations."""


class ManifestError(PluginError):
    """Raised when a plugin manifest is invalid or missing required data."""


class SchemaValidationError(PluginError):
    """Raised when input/output payloads fail schema validation."""


class PolicyViolationError(PluginError):
    """Raised when a plugin violates resource or lane policy constraints."""


class PluginLoadError(PluginError):
    """Raised when a plugin module cannot be loaded or instantiated."""


class PluginExecutionError(PluginError):
    """Raised when a plugin fails to execute or exceeds latency budgets."""
