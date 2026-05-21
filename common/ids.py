"""ID and version generators for pipeline entities."""

import uuid
from datetime import datetime, timezone


def new_molecule_id() -> str:
    return f"mol_{uuid.uuid4().hex[:12]}"


def new_calculation_id() -> str:
    return f"dft_{uuid.uuid4().hex[:12]}"


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def new_model_version() -> str:
    """Time-based version tag, e.g. v20260517120000."""
    return f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def new_benchmark_id() -> str:
    return f"bench_{uuid.uuid4().hex[:12]}"
