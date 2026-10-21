from dataclasses import dataclass, asdict
from pathlib import Path
import os

from agent.errors import AgentError

ROOT = Path(__file__).resolve().parents[1]


def integer(name, default, low, high):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise AgentError("INVALID_CONFIG", f"{name} must be an integer") from exc
    if not low <= value <= high:
        raise AgentError("INVALID_CONFIG", f"{name} must be between {low} and {high}")
    return value


def boolean(name, default):
    value = os.environ.get(name, "1" if default else "0").lower()
    if value not in {"1", "0", "true", "false", "yes", "no"}:
        raise AgentError("INVALID_CONFIG", f"{name} must be a boolean")
    return value in {"1", "true", "yes"}


@dataclass(frozen=True)
class Config:
    llm_mode: str = "live"
    time_budget_seconds: int = 10800
    max_workers: int = 2
    train_cpu_threads: int = 3
    train_memory_mb: int = 10240
    experiment_timeout_seconds: int = 600
    max_research_experiments: int = 4
    llm_max_calls: int = 24
    llm_timeout_seconds: int = 90
    llm_max_output_tokens: int = 4096
    max_context_tokens: int = 16000
    log_level: str = "INFO"
    log_model_io: bool = True
    work_dir: Path = Path("runs")
    seed: int = 42

    @classmethod
    def from_env(cls, *, llm_mode=None, work_dir=None):
        mode = llm_mode or os.environ.get("LLM_MODE", "live")
        if mode not in {"live", "mock"}:
            raise AgentError("INVALID_CONFIG", "LLM_MODE must be live or mock")
        cfg = cls(
            llm_mode=mode,
            time_budget_seconds=integer("TIME_BUDGET_SECONDS", 10800, 5, 10800),
            max_workers=integer("MAX_WORKERS", 2, 1, 2),
            train_cpu_threads=integer("TRAIN_CPU_THREADS", 3, 1, 3),
            train_memory_mb=integer("TRAIN_MEMORY_MB", 10240, 256, 10240),
            experiment_timeout_seconds=integer("EXPERIMENT_TIMEOUT_SECONDS", 600, 1, 3600),
            max_research_experiments=integer("MAX_RESEARCH_EXPERIMENTS", 4, 0, 100),
            llm_max_calls=integer("LLM_MAX_CALLS", 24, 1, 1000),
            llm_timeout_seconds=integer("LLM_TIMEOUT_SECONDS", 90, 1, 600),
            llm_max_output_tokens=integer("LLM_MAX_OUTPUT_TOKENS", 4096, 128, 16000),
            max_context_tokens=integer("MAX_CONTEXT_TOKENS", 16000, 1000, 32768),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            log_model_io=boolean("LOG_MODEL_IO", True),
            work_dir=Path(work_dir or os.environ.get("WORK_DIR", "runs")).resolve(),
            seed=integer("SEED", 42, 0, 2**31 - 1),
        )
        if cfg.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise AgentError("INVALID_CONFIG", "Unsupported LOG_LEVEL")
        return cfg

    def public_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
