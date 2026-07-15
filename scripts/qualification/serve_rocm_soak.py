#!/usr/bin/env python3
"""Compatibility launcher for the committed ROCm development-soak workload."""

from serve_development_soak import *  # noqa: F401,F403
from serve_development_soak import main


if __name__ == "__main__":
    raise SystemExit(main())
