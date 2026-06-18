"""Utils module initialization"""
from .model_wrapper import wrap_model_call, inject_preferences_to_prompt

__all__ = [
    "wrap_model_call",
    "inject_preferences_to_prompt",
]
