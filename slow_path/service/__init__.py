"""Lightweight package initializer for slow_path.service.

Avoid importing heavy modules (like the FastAPI app) at package import time,
so that importing submodules (e.g., ``from slow_path.service.worker import Worker``)
does not instantiate a global Worker or load a model inadvertently.

If you need the HTTP service API (ServiceClient, enqueue_infer, etc.), import
them directly from ``slow_path.service.api`` to trigger FastAPI and background
worker initialization explicitly.
"""

from .worker import get_local_cache

__all__ = ["get_local_cache"]
