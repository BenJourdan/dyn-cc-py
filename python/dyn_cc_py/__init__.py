"""
Python frontend for the dyn-cc Rust extension.

The compiled extension module (built with maturin) is named ``dyn_cc_py``.
This package re-exports the native functions and is the place to add
any pure-Python helpers or ergonomics.
"""

from importlib import metadata

# Import everything from the native extension.
from .dyn_cc_py import *  


def __getattr__(name: str):
    """Expose the package version via dyn_cc_py.__version__."""
    if name == "__version__":
        try:
            return metadata.version("dyn-cc-py")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def splat():
    print("splat")


__all__ = [name for name in globals() if not name.startswith("_")]
