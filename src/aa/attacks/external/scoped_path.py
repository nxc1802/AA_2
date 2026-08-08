import sys
import os
from contextlib import contextmanager


@contextmanager
def scoped_sys_path(path: str):
    """
    Temporarily prepends path to sys.path and isolates generic top-level modules
    (e.g., 'utils', 'models', 'attacks') in sys.modules to prevent cross-adapter collisions.
    """
    old_modules = {}
    colliding_names = ["utils", "models", "attacks", "parsers", "datasets", "deepfool"]

    for name in colliding_names:
        if name in sys.modules:
            old_modules[name] = sys.modules.pop(name)

    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True

    try:
        yield
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)
        for name in colliding_names:
            if name in sys.modules:
                sys.modules.pop(name)
        for name, mod in old_modules.items():
            sys.modules[name] = mod
