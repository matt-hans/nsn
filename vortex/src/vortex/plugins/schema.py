"""Schema validation for plugin inputs and outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vortex.plugins.errors import SchemaValidationError
from vortex.plugins.types import JsonSchema


def validate_schema(schema: JsonSchema, payload: Mapping[str, Any], *, context: str) -> None:
    """Validate payload against a minimal JSON schema subset.

    Supported schema fields:
    - type: "object"
    - properties: mapping of field -> {"type": ...}
    - required: list of required fields
    - items: for arrays, schema of array items
    """
    schema_type = schema.get("type")
    if schema_type != "object":
        raise SchemaValidationError(f"{context} schema must declare type 'object'")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise SchemaValidationError(f"{context} schema 'properties' must be a mapping")

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise SchemaValidationError(f"{context} schema 'required' must be a list")

    for field in required:
        if field not in payload:
            raise SchemaValidationError(f"{context} missing required field '{field}'")

    for field_name, value in payload.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            continue
        _validate_field(field_name, value, field_schema, context=context)


def _validate_field(
    field: str, value: Any, field_schema: Mapping[str, Any], *, context: str
) -> None:
    field_type = field_schema.get("type")
    if field_type is None:
        return

    if field_type == "string":
        if not isinstance(value, str):
            raise SchemaValidationError(
                f"{context} field '{field}' must be string, got {type(value).__name__}"
            )
    elif field_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaValidationError(
                f"{context} field '{field}' must be integer, got {type(value).__name__}"
            )
    elif field_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaValidationError(
                f"{context} field '{field}' must be number, got {type(value).__name__}"
            )
    elif field_type == "boolean":
        if not isinstance(value, bool):
            raise SchemaValidationError(
                f"{context} field '{field}' must be boolean, got {type(value).__name__}"
            )
    elif field_type == "object":
        if not isinstance(value, Mapping):
            raise SchemaValidationError(
                f"{context} field '{field}' must be object, got {type(value).__name__}"
            )
    elif field_type == "array":
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"{context} field '{field}' must be array, got {type(value).__name__}"
            )
        items_schema = field_schema.get("items")
        if isinstance(items_schema, Mapping):
            for idx, item in enumerate(value):
                _validate_field(f"{field}[{idx}]", item, items_schema, context=context)
    else:
        raise SchemaValidationError(
            f"{context} field '{field}' has unsupported type '{field_type}'"
        )
