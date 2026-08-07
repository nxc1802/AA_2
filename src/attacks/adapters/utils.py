import sys
import os
import contextlib

@contextlib.contextmanager
def scoped_sys_path(target_dir: str):
    """
    Context manager to temporarily place target_dir at sys.path[0]
    and purge conflicting generic module names (e.g. 'utils', 'mask', 'models')
    from sys.modules to prevent cross-repo import collisions.
    """
    abs_dir = os.path.abspath(target_dir)
    conflicting_modules = ['utils', 'mask', 'models', 'linear_solver', 'deepfool', 'parsers']
    
    # Save original state of conflicting modules
    saved_modules = {}
    for mod in conflicting_modules:
        if mod in sys.modules:
            saved_modules[mod] = sys.modules.pop(mod)
            
    sys.path.insert(0, abs_dir)
    try:
        yield
    finally:
        if abs_dir in sys.path:
            sys.path.remove(abs_dir)
        # Restore original modules
        for mod in conflicting_modules:
            if mod in sys.modules:
                del sys.modules[mod]
            if mod in saved_modules:
                sys.modules[mod] = saved_modules[mod]
