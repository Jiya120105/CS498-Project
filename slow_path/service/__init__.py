"""Public exports for the slow_path.service package.

A stable API so other modules can import the slow-path service

Public symbols:
 - ServiceClient: synchronous in-process client (uses TestClient)
 - enqueue_infer: async helper to enqueue a Job directly
 - trigger_tick: the ASGI endpoint and other helpers remain available via
	 import from .api if deeper access is required.
"""

from .api import ServiceClient, enqueue_infer, trigger_tick, trigger_tick as trigger_tick_endpoint
from .worker import get_local_cache

__all__ = ["ServiceClient", "enqueue_infer", "trigger_tick", "trigger_tick_endpoint", "get_local_cache"]
