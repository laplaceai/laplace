"""Static checks and a restricted namespace for language-model-written forecasting programs.

This is defence in depth for research runs on our own machines, where every program comes from our own mutation
calls: an AST allow-list (imports limited to ``math`` and ``numpy``, no private attributes, no I/O, no dynamic code),
restricted builtins, and execution in a separate, resource-limited process (see ``evaluate``). Production runs
execute programs in isolated containers without network access.
"""

from __future__ import annotations

import ast
import builtins
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np

MAX_SOURCE_BYTES = 16_000
MAX_LINES = 300
ALLOWED_MODULES: dict[str, ModuleType] = {"math": math, "numpy": np}
FORBIDDEN_NAMES = frozenset(
    {
        "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec", "exit", "getattr", "globals",
        "help", "input", "locals", "memoryview", "open", "print", "quit", "setattr", "super", "type", "vars",
    }
)  # fmt: skip
# numpy entry points that read or write files, load code, reach outside the process, or are non-deterministic.
FORBIDDEN_NUMPY_ATTRIBUTES = frozenset(
    {
        "ctypeslib", "DataSource", "distutils", "f2py", "fromfile", "fromregex", "genfromtxt", "lib", "load",
        "loadtxt", "memmap", "random", "save", "savetxt", "savez", "savez_compressed", "testing", "tofile",
    }
)  # fmt: skip
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
        "map", "max", "min", "pow", "range", "reversed", "round", "set", "slice", "sorted", "sum", "tuple", "zip",
        "ArithmeticError", "Exception", "FloatingPointError", "IndexError", "KeyError", "OverflowError",
        "TypeError", "ValueError", "ZeroDivisionError",
    )
}  # fmt: skip


class ProgramRejected(ValueError):
    """The source failed static checks or does not define ``forecast(ctx)``."""


def _numpy_aliases(tree: ast.Module) -> set[str]:
    aliases = {"numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "numpy"}
    return aliases


def check(source: str) -> ast.Module:
    """Parse and statically check ``source``; raise ``ProgramRejected`` with the reason."""
    if len(source.encode()) > MAX_SOURCE_BYTES or source.count("\n") > MAX_LINES:
        raise ProgramRejected("program too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramRejected(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    numpy_names = _numpy_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in ALLOWED_MODULES]
            if bad:
                raise ProgramRejected(f"import of {', '.join(bad)} is not allowed (only math and numpy)")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_MODULES or node.level:
                raise ProgramRejected(f"import from {node.module} is not allowed (only math and numpy)")
            if node.module == "numpy" and any(a.name in FORBIDDEN_NUMPY_ATTRIBUTES for a in node.names):
                raise ProgramRejected("that numpy import is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ProgramRejected(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ProgramRejected(f"access to private attribute {node.attr} is not allowed")
            if node.attr in FORBIDDEN_NUMPY_ATTRIBUTES and _rooted_in(node.value, numpy_names):
                raise ProgramRejected(f"numpy.{node.attr} is not allowed")
        elif isinstance(node, ast.ClassDef | ast.Global | ast.Nonlocal | ast.AsyncFunctionDef | ast.Await):
            raise ProgramRejected(f"{type(node).__name__} is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forecast" for n in tree.body):
        raise ProgramRejected("no top-level forecast(ctx) function")
    return tree


def _rooted_in(node: ast.expr, names: set[str]) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in names


def _restricted_import(
    name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0
) -> ModuleType:
    if level or name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name} is not allowed")
    return ALLOWED_MODULES[name]


def load(source: str) -> Callable[[Any], Any]:
    """Check ``source`` and return its ``forecast`` function, bound to a restricted namespace."""
    tree = check(source)
    namespace: dict[str, Any] = {"__builtins__": {**SAFE_BUILTINS, "__import__": _restricted_import}}
    exec(compile(tree, "<program>", "exec"), namespace)
    fn = namespace.get("forecast")
    if not callable(fn):
        raise ProgramRejected("forecast is not callable")
    return fn  # type: ignore[no-any-return]
