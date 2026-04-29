"""Early runtime compatibility tweaks for Streamlit Cloud."""

from __future__ import annotations

import os


# Chroma's OpenTelemetry dependency chain can break on newer protobuf runtimes.
# Force the Python implementation before other imports happen.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
