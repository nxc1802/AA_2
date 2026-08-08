import sys
from contextlib import contextmanager


@contextmanager
def scoped_sys_path(path: str):
    """Temporarily prepends path to sys.path during execution block."""
    if path not in sys.path:
        sys.path.insert(0, path)
        try:
            yield
        finally:
            if path in sys.path:
                sys.path.remove(path)
    else:
        yield
