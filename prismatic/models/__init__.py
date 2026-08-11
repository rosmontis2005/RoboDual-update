"""Prismatic model package with lazy imports."""

import importlib

_LOAD_EXPORTS = {
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
    "load_vla",
}

_MATERIALIZE_EXPORTS = {
    "get_llm_backbone_and_tokenizer",
    "get_vision_backbone_and_transform",
    "get_vlm",
}

def __getattr__(name):
    if name in _LOAD_EXPORTS:
        module = importlib.import_module(".load", __name__)
        return getattr(module, name)

    if name in _MATERIALIZE_EXPORTS:
        module = importlib.import_module(".materialize", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = sorted(_LOAD_EXPORTS | _MATERIALIZE_EXPORTS)
