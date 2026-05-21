"""Shared environment settings for services and workers."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MinIO / S3
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_secure: bool = False
    minio_bucket: str = "dft-workflow"
    minio_region: str = "us-east-1"

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"

    # Batch training policy: training is enqueued after every N completed DFT
    # runs, counted via a Redis INCR counter (not per-molecule).
    training_batch_min_samples: int = 1000
    training_batch_counter_key: str = "dft:completed_since_last_train"

    # Default density-predictor name. Storage layout uses this as a directory
    # prefix (``models/{default_model_name}/{version}/...``) and clients reach
    # it via ``/api/density/predict/{default_model_name}``.
    default_model_name: str = "mace"

    # In-process TTL of the active model in the density service (seconds).
    # ``0`` disables caching (every request reloads from MinIO).
    model_cache_ttl_sec: int = 300

    # Training backend: ``mace`` (real graph2mat + mace-torch + MSE) or
    # ``mock`` (random tensor; used by unit tests and quick smoke runs).
    training_engine: str = "mace"

    # DFT backend: ``pyscf`` (real SCF) or ``mock`` (synthetic DM, tests only).
    dft_engine: str = "pyscf"
    dft_default_method: str = "rks"
    dft_default_basis: str = "sto-3g"
    dft_max_scf_cycles: int = 100
    dft_scf_conv_tol: float = 1e-9

    log_level: str = "INFO"

    # --- Object key layout (single bucket, path-style prefixes) ---

    @property
    def molecules_raw_prefix(self) -> str:
        return "molecules/raw"

    @property
    def molecules_manifests_prefix(self) -> str:
        return "molecules/manifests"

    @property
    def dft_artifacts_prefix(self) -> str:
        return "dft/artifacts"

    @property
    def dft_manifests_prefix(self) -> str:
        return "dft/manifests"

    @property
    def models_prefix(self) -> str:
        return "models"

    @property
    def models_active_prefix(self) -> str:
        return "models/active"

    @property
    def inference_artifacts_prefix(self) -> str:
        return "inference/artifacts"

    @property
    def benchmarks_prefix(self) -> str:
        return "benchmarks"

    @property
    def jobs_prefix(self) -> str:
        return "jobs"

    def molecule_raw_key(self, molecule_id: str) -> str:
        return f"{self.molecules_raw_prefix}/{molecule_id}.xyz"

    def molecule_manifest_key(self, molecule_id: str) -> str:
        return f"{self.molecules_manifests_prefix}/{molecule_id}.json"

    def dft_artifact_key(self, molecule_id: str, calculation_id: str) -> str:
        return f"{self.dft_artifacts_prefix}/{molecule_id}/{calculation_id}.npz"

    def dft_manifest_key(self, calculation_id: str) -> str:
        return f"{self.dft_manifests_prefix}/{calculation_id}.json"

    def model_dir_prefix(self, model_name: str, version: str) -> str:
        return f"{self.models_prefix}/{model_name}/{version}"

    def model_weights_key(self, model_name: str, version: str) -> str:
        return f"{self.model_dir_prefix(model_name, version)}/model.pt"

    def model_config_key(self, model_name: str, version: str) -> str:
        return f"{self.model_dir_prefix(model_name, version)}/config.json"

    def model_metrics_key(self, model_name: str, version: str) -> str:
        return f"{self.model_dir_prefix(model_name, version)}/metrics.json"

    def model_manifest_key(self, model_name: str, version: str) -> str:
        return f"{self.model_dir_prefix(model_name, version)}/manifest.json"

    def active_model_key(self, model_name: str) -> str:
        return f"{self.models_active_prefix}/{model_name}.json"

    def inference_artifact_key(self, request_id: str) -> str:
        return f"{self.inference_artifacts_prefix}/{request_id}.npz"

    def benchmark_key(self, benchmark_id: str) -> str:
        return f"{self.benchmarks_prefix}/{benchmark_id}.json"

    def job_key(self, job_id: str) -> str:
        return f"{self.jobs_prefix}/{job_id}.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
