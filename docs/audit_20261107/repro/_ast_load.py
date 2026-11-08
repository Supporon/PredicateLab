"""Load selected top-level functions / class methods from an audited source file
without importing the module (the audited module imports qlib at top level).

Only the requested definitions are compiled; nothing else in the file runs.
The SHA-256 of the source file is returned so every result is tied to the
exact audited bytes.
"""
from __future__ import annotations

import ast
import hashlib
import types
from pathlib import Path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_functions(path: str | Path, names: list[str], namespace: dict) -> dict:
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise KeyError(f"not found in {path}: {sorted(missing)}")
    mod = ast.Module(body=wanted, type_ignores=[])
    code = compile(mod, str(path), "exec")
    exec(code, namespace)
    return {n: namespace[n] for n in names}


def load_methods(path: str | Path, class_name: str, names: list[str], namespace: dict) -> dict:
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    wanted = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise KeyError(f"not found in {class_name}: {sorted(missing)}")
    mod = ast.Module(body=wanted, type_ignores=[])
    exec(compile(mod, str(path), "exec"), namespace)
    return {n: namespace[n] for n in names}
