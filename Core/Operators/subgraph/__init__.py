"""Lazy public exports: importing a graph operator does not load LLM/vector stacks."""

from importlib import import_module

_EXPORTS = {
    'subgraph_khop_paths': 'Core.Operators.subgraph.khop_paths',
    'subgraph_steiner_tree': 'Core.Operators.subgraph.steiner_tree',
    'subgraph_agent_path': 'Core.Operators.subgraph.agent_path',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
