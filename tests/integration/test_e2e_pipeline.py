"""End-to-end test of the main pipeline:

  archive upload  →  DFT (PySCF)  →  Redis counter reaches threshold  →
  training task auto-enqueued  →  new model published  →  density-service
  serves the new version.

The test is skipped unless:

* The Nginx gateway is reachable at ``E2E_BASE_URL`` (default ``http://localhost``).
* ``TRAINING_BATCH_MIN_SAMPLES`` is small enough for a manual run
  (default cap ``E2E_MAX_THRESHOLD=10``).

To run::

    # 1) recreate stack with a low trigger threshold (so we don't need 1000 DFT runs):
    TRAINING_BATCH_MIN_SAMPLES=3 docker compose up -d --force-recreate

    # 2) run the test from the conda env (it talks to the host-side Nginx + Redis):
    TRAINING_BATCH_MIN_SAMPLES=3 \
      pytest tests/integration/test_e2e_pipeline.py -q -s

Timings can be tuned via ``E2E_DFT_TIMEOUT_SEC`` and ``E2E_TRAIN_TIMEOUT_SEC``.
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import httpx
import pytest

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost")
THRESHOLD = int(os.environ.get("TRAINING_BATCH_MIN_SAMPLES", "1000"))
MAX_THRESHOLD = int(os.environ.get("E2E_MAX_THRESHOLD", "10"))
DFT_TIMEOUT_SEC = int(os.environ.get("E2E_DFT_TIMEOUT_SEC", "240"))
TRAIN_TIMEOUT_SEC = int(os.environ.get("E2E_TRAIN_TIMEOUT_SEC", "300"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
COUNTER_KEY = os.environ.get(
    "TRAINING_BATCH_COUNTER_KEY", "dft:completed_since_last_train"
)


# Small, real molecules safe for PySCF + sto-3g (5 distinct ones cover thresholds 1..5).
MOLECULES: dict[str, str] = {
    "water.xyz":   "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
    "h2.xyz":      "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
    "n2.xyz":      "2\nn2\nN 0.0 0.0 0.0\nN 0.0 0.0 1.10\n",
    "hf.xyz":      "2\nhf\nF 0.0 0.0 0.0\nH 0.0 0.0 0.92\n",
    "co.xyz":      "2\nco\nC 0.0 0.0 0.0\nO 0.0 0.0 1.13\n",
}


def _gateway_reachable() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/api/data/health", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _gateway_reachable(),
        reason=f"Gateway not reachable at {BASE_URL} (run `docker compose up -d`)",
    ),
    pytest.mark.skipif(
        THRESHOLD > MAX_THRESHOLD,
        reason=(
            f"TRAINING_BATCH_MIN_SAMPLES={THRESHOLD} > {MAX_THRESHOLD}; "
            "recreate stack with `TRAINING_BATCH_MIN_SAMPLES=3 docker compose up -d "
            "--force-recreate` (and export the same value when running pytest)"
        ),
    ),
]


def _build_archive(n_molecules: int) -> tuple[bytes, list[str]]:
    items = list(MOLECULES.items())
    if n_molecules > len(items):
        # pad by repeating molecules — each one still produces a fresh manifest
        items = (items * (n_molecules // len(items) + 1))[:n_molecules]
    else:
        items = items[:n_molecules]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, (name, content) in enumerate(items):
            # disambiguate filenames so all entries are valid zip records
            zf.writestr(f"{i:03d}_{name}", content.encode("utf-8"))
    return buf.getvalue(), [name for name, _ in items]


def _reset_redis_counter() -> None:
    import redis as redis_lib

    redis_lib.Redis.from_url(REDIS_URL, decode_responses=True).delete(COUNTER_KEY)


def _current_counter() -> int:
    import redis as redis_lib

    raw = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True).get(COUNTER_KEY)
    return int(raw or 0)


def _get_active_version() -> str | None:
    r = httpx.get(f"{BASE_URL}/api/density/models/active", timeout=5)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["manifest"]["version"]


def _wait_for_dft_completion(mol_ids: list[str], timeout_sec: int) -> None:
    pending = set(mol_ids)
    deadline = time.time() + timeout_sec
    while pending and time.time() < deadline:
        for mid in list(pending):
            r = httpx.get(f"{BASE_URL}/api/data/molecules/{mid}", timeout=5)
            r.raise_for_status()
            status = r.json()["manifest"]["status"]
            if status == "dft_completed":
                pending.discard(mid)
            elif status == "dft_failed":
                raise AssertionError(f"DFT failed for {mid}")
        if pending:
            time.sleep(2)
    if pending:
        raise AssertionError(
            f"DFT did not finish within {timeout_sec}s for {len(pending)} molecules: "
            f"{sorted(pending)}"
        )


def _wait_for_new_active_version(initial: str | None, timeout_sec: int) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        version = _get_active_version()
        if version and version != initial:
            return version
        time.sleep(1)
    raise AssertionError(
        f"no new active model published within {timeout_sec}s "
        f"(still {initial!r}); check `docker compose logs training-worker`"
    )


def test_archive_upload_to_model_update() -> None:
    """One run of the full happy path; each step asserts the previous side-effect."""
    pytest.importorskip("redis")

    # 0. Clean Redis counter so the threshold is reached exactly by our batch.
    _reset_redis_counter()
    initial_version = _get_active_version()
    print(f"\n[e2e] initial active version: {initial_version!r}")

    # 1. Upload archive with exactly THRESHOLD molecules.
    n = max(THRESHOLD, 1)
    archive_bytes, entry_names = _build_archive(n)
    print(f"[e2e] uploading archive: {n} molecules ({entry_names})")
    r = httpx.post(
        f"{BASE_URL}/api/data/molecules/batch",
        files={"file": ("e2e.zip", archive_bytes, "application/zip")},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    assert body["queued"] == n, body
    mol_ids = [it["molecule_id"] for it in body["items"] if it["status"] == "queued"]
    assert len(mol_ids) == n

    # 2. Wait for all DFT calculations to complete (PySCF, real SCF).
    print(f"[e2e] waiting up to {DFT_TIMEOUT_SEC}s for DFT…")
    _wait_for_dft_completion(mol_ids, DFT_TIMEOUT_SEC)
    print(f"[e2e] all {n} DFT calculations completed")

    # 3. The completion counter should have triggered training automatically.
    #    After triggering the script resets the counter to 0.
    counter_after = _current_counter()
    assert counter_after < THRESHOLD, (
        f"counter={counter_after} did not reset; trigger did not fire"
    )

    # 4. Training task runs and publishes a new active model pointer.
    print(f"[e2e] waiting up to {TRAIN_TIMEOUT_SEC}s for new active model…")
    new_version = _wait_for_new_active_version(initial_version, TRAIN_TIMEOUT_SEC)
    print(f"[e2e] new active version: {new_version}")
    assert new_version != initial_version

    # 5. Density-service picks up the new model (force cache refresh to make
    #    the test deterministic even if MODEL_CACHE_TTL_SEC is large).
    httpx.post(f"{BASE_URL}/api/density/models/cache/invalidate", timeout=5).raise_for_status()
    serving = httpx.get(f"{BASE_URL}/api/density/models/serving", timeout=5).json()
    print(f"[e2e] serving: version={serving['manifest']['version']} "
          f"selection={serving['selection']} metric={serving['metric']} "
          f"versions_considered={serving['versions_considered']}")
    assert serving["versions_considered"] >= 1

    # 6. Inference end-to-end through both prediction modes.
    water_xyz = MOLECULES["water.xyz"].encode()

    pred_mace = httpx.post(
        f"{BASE_URL}/api/density/predict/mace",
        files={"file": ("water.xyz", water_xyz, "chemical/x-xyz")},
        timeout=15,
    )
    pred_mace.raise_for_status()
    mb = pred_mace.json()
    assert mb["model"] == "mace"
    assert mb["scf_iterations"] == 0
    assert mb["details"]["cache_selection"] in {"best_loss", "train_loss", "active_pointer"}
    print(f"[e2e] predict/mace ok: version={mb['details']['model_version']} "
          f"shape={mb['shape']} wall={mb['wall_time_sec']}s")

    pred_dft = httpx.post(
        f"{BASE_URL}/api/density/predict/dft",
        files={"file": ("water.xyz", water_xyz, "chemical/x-xyz")},
        timeout=120,  # real PySCF is slower than mock
    )
    pred_dft.raise_for_status()
    db = pred_dft.json()
    assert db["model"] == "dft"
    assert db["scf_iterations"] >= 1
    assert db["shape"][0] == db["shape"][1] > 0
    print(f"[e2e] predict/dft  ok: iters={db['scf_iterations']} "
          f"shape={db['shape']} wall={db['wall_time_sec']}s")

    # 7. Honesty check: the MACE forward pass must produce a DM in the same
    #    AO basis as PySCF (sto-3g for water → 7×7). If shapes differ, MACE
    #    is returning a placeholder block and inference is not real.
    assert mb["shape"] == db["shape"], (
        f"MACE forward DM shape {mb['shape']} != PySCF DM shape {db['shape']}; "
        "the inference path is not running a real graph2mat forward"
    )
