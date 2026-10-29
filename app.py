"""Backward-compatible imports for the core application.

The production Docker entry point is `enterprise_main:app`. This module
re-exports the canonical core implementation so `uvicorn app:app` and the
older test/import surface do not keep a divergent lexical-only retriever.
"""
from main import *  # noqa: F401,F403
