from dataclasses import dataclass, asdict
from pathlib import Path
import math
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


def real(name, default, low, high):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise AgentError("INVALID_CONFIG", f"{name} must be a number") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise AgentError("INVALID_CONFIG", f"{name} must be between {low} and {high}")
    return value


def optional_limit(name, default, low, high):
    value = os.environ.get(name)
    if value is not None and value.strip().lower() in {'0', 'none', 'unlimited'}:
        return None
    return integer(name, default, low, high) if value is not None or default is not None else None


@dataclass(frozen=True)
class Config:
    profile: str = 'development'
    llm_mode: str = "live"
    time_budget_seconds: int = 10800
    max_workers: int = 2
    train_cpu_threads: int = 3
    train_memory_mb: int = 10240
    experiment_timeout_seconds: int = 600
    max_research_experiments: int = 12  # Legacy configuration metadata, never a stopping condition.
    research_series_size: int = 4
    catalog_warmup_experiments: int = 4
    research_mode: str = "adaptive"
    llm_max_calls: int | None = 32
    llm_timeout_seconds: int = 600
    llm_operation_timeout_seconds: int = 1200
    llm_max_output_tokens: int = 16384
    llm_bootstrap_output_tokens: int = 4096
    llm_max_output_tokens_limit: int = 32768
    llm_max_total_output_tokens: int | None = 262144
    llm_research_tool_turns: int = 8
    feature_research_enabled: bool = True
    feature_seed_budget_fraction: float = .15
    feature_seed_timeout_seconds: int = 120
    llm_request_max_bytes: int = 196608
    llm_context_window_tokens: int = 262144
    llm_reasoning_protocol: str = 'compatible'
    llm_reasoning_effort: str | None = None
    llm_reasoning_field: str = 'reasoning_content'
    llm_reasoning_history_bytes: int = 32768
    llm_progress_seconds: int = 15
    max_context_tokens: int = 16000
    log_level: str = "INFO"
    log_model_io: bool = True
    console_format: str = "human"
    work_dir: Path = Path("runs")
    seed: int = 42
    graph_enabled: bool = True
    graph_peers: int = 2
    graph_train_workers: int = 1
    graph_max_live_branches: int = 3
    graph_max_total_branches: int = 15
    graph_plateau_window: int = 4
    graph_plateau_seconds: int = 900
    graph_refinement_enabled: bool = True
    graph_search_fraction: float = 0.85
    graph_plateau_absolute_epsilon: float = 0.001
    graph_plateau_relative_epsilon: float = 0.001
    graph_exploit_steps: int = 3
    graph_min_effect: float = 0.001
    graph_context_bytes: int = 48000
    graph_max_artifact_bytes: int = 8000000000
    graph_min_free_disk_bytes: int = 2000000000
    model_snapshot_max_bytes: int = 16777216
    model_snapshot_total_bytes: int = 67108864
    worker_log_max_bytes: int = 1048576

    @classmethod
    def from_env(cls, *, llm_mode=None, work_dir=None, profile=None):
        profile = profile or os.environ.get('AUTOML_PROFILE', 'platform')
        if profile not in {'development', 'platform'}:
            raise AgentError('INVALID_CONFIG', 'AUTOML_PROFILE must be development or platform')
        platform = profile == 'platform'
        mode = llm_mode or os.environ.get("LLM_MODE", "live")
        if mode not in {"live", "mock"}:
            raise AgentError("INVALID_CONFIG", "LLM_MODE must be live or mock")
        cfg = cls(
            profile=profile,
            llm_mode=mode,
            time_budget_seconds=integer("TIME_BUDGET_SECONDS", 10800, 5, 10800),
            max_workers=integer("MAX_WORKERS", 2, 1, 2),
            train_cpu_threads=integer("TRAIN_CPU_THREADS", 3, 1, 3),
            train_memory_mb=integer("TRAIN_MEMORY_MB", 10240, 256, 10240),
            experiment_timeout_seconds=integer("EXPERIMENT_TIMEOUT_SECONDS", 600, 1, 3600),
            max_research_experiments=integer("MAX_RESEARCH_EXPERIMENTS", 12, 0, 100),
            research_series_size=integer("RESEARCH_SERIES_SIZE", 4, 1, 16),
            catalog_warmup_experiments=integer("CATALOG_WARMUP_EXPERIMENTS", 4, 0, 20),
            research_mode=os.environ.get("RESEARCH_MODE", "adaptive"),
            llm_max_calls=optional_limit("LLM_MAX_CALLS", None if platform else 32, 1, 1000000),
            llm_timeout_seconds=integer("LLM_TIMEOUT_SECONDS", 600, 1, 1800),
            llm_operation_timeout_seconds=integer("LLM_OPERATION_TIMEOUT_SECONDS", 1200, 1, 3600),
            llm_max_output_tokens=integer("LLM_MAX_OUTPUT_TOKENS", 16384, 128, 131072),
            llm_bootstrap_output_tokens=integer("LLM_BOOTSTRAP_OUTPUT_TOKENS", 4096, 128, 131072),
            llm_max_output_tokens_limit=integer("LLM_MAX_OUTPUT_TOKENS_LIMIT", 32768, 128, 131072),
            llm_max_total_output_tokens=optional_limit("LLM_MAX_TOTAL_OUTPUT_TOKENS", None if platform else 262144, 1024, 1000000000),
            llm_research_tool_turns=integer('LLM_RESEARCH_TOOL_TURNS', 8, 0, 32),
            feature_research_enabled=boolean('FEATURE_RESEARCH_ENABLED', True),
            feature_seed_budget_fraction=real('FEATURE_SEED_BUDGET_FRACTION', .15, 0, .5),
            feature_seed_timeout_seconds=integer('FEATURE_SEED_TIMEOUT_SECONDS', 120, 1, 600),
            llm_request_max_bytes=integer('LLM_REQUEST_MAX_BYTES', 196608, 16000, 2000000),
            llm_context_window_tokens=integer('LLM_CONTEXT_WINDOW_TOKENS', 262144, 4096, 1000000),
            llm_reasoning_protocol=os.environ.get('LLM_REASONING_PROTOCOL', 'compatible'),
            llm_reasoning_effort=os.environ.get('LLM_REASONING_EFFORT') or None,
            llm_reasoning_field=os.environ.get('LLM_REASONING_FIELD', 'reasoning_content'),
            llm_reasoning_history_bytes=integer('LLM_REASONING_HISTORY_BYTES', 32768, 0, 262144),
            llm_progress_seconds=integer("LLM_PROGRESS_SECONDS", 15, 1, 60),
            max_context_tokens=integer("MAX_CONTEXT_TOKENS", 16000, 1000, 32768),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            log_model_io=boolean("LOG_MODEL_IO", True),
            console_format=os.environ.get("CONSOLE_LOG_FORMAT", "human"),
            work_dir=Path(work_dir or os.environ.get("WORK_DIR", "runs")).resolve(),
            seed=integer("SEED", 42, 0, 2**31 - 1),
            graph_enabled=boolean("GRAPH_ENABLED", True),
            graph_peers=integer("GRAPH_PEERS", min(2, integer("MAX_WORKERS", 2, 1, 2)), 1, 2),
            graph_train_workers=integer("GRAPH_TRAIN_WORKERS", 1, 1, 2),
            graph_max_live_branches=integer("GRAPH_MAX_LIVE_BRANCHES", 3, 1, 31),
            graph_max_total_branches=integer("GRAPH_MAX_TOTAL_BRANCHES", 15, 1, 255),
            graph_plateau_window=integer("GRAPH_PLATEAU_WINDOW", 4, 1, 20),
            graph_plateau_seconds=integer("GRAPH_PLATEAU_SECONDS", 900, 0, 10800),
            graph_refinement_enabled=boolean("GRAPH_REFINEMENT_ENABLED", not platform),
            graph_search_fraction=real("GRAPH_SEARCH_FRACTION", 0.85, 0.8, 0.9),
            graph_plateau_absolute_epsilon=real("GRAPH_PLATEAU_ABSOLUTE_EPSILON", 0.001, 0, 1000000),
            graph_plateau_relative_epsilon=real("GRAPH_PLATEAU_RELATIVE_EPSILON", 0.001, 0, 1),
            graph_exploit_steps=integer("GRAPH_EXPLOIT_STEPS", 3, 1, 8),
            graph_min_effect=real("GRAPH_MIN_EFFECT", 0.001, 0, 1000000),
            graph_context_bytes=integer("GRAPH_CONTEXT_BYTES", 48000, 3000, 2000000),
            graph_max_artifact_bytes=integer("GRAPH_MAX_ARTIFACT_BYTES", 8000000000, 1000000, 100000000000),
            graph_min_free_disk_bytes=integer("GRAPH_MIN_FREE_DISK_BYTES", 2000000000, 0, 100000000000),
            model_snapshot_max_bytes=integer("MODEL_SNAPSHOT_MAX_BYTES", 16777216, 0, 100000000000),
            model_snapshot_total_bytes=integer("MODEL_SNAPSHOT_TOTAL_BYTES", 67108864, 0, 100000000000),
            worker_log_max_bytes=integer("WORKER_LOG_MAX_BYTES", 1048576, 256, 8388608),
        )
        if cfg.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise AgentError("INVALID_CONFIG", "Unsupported LOG_LEVEL")
        if cfg.llm_reasoning_protocol not in {'compatible', 'qwen_chat_template'}:
            raise AgentError('INVALID_CONFIG', 'LLM_REASONING_PROTOCOL must be compatible or qwen_chat_template')
        if cfg.llm_reasoning_effort not in {None, 'low', 'medium', 'xhigh'}:
            raise AgentError('INVALID_CONFIG', 'LLM_REASONING_EFFORT must be low, medium or xhigh')
        if cfg.llm_reasoning_field not in {'reasoning_content', 'reasoning'}:
            raise AgentError('INVALID_CONFIG', 'LLM_REASONING_FIELD must be reasoning_content or reasoning')
        if cfg.llm_reasoning_effort is not None and cfg.llm_reasoning_protocol == 'compatible':
            raise AgentError('INVALID_CONFIG', 'Reasoning controls require an explicitly selected provider protocol; proxy support is not inferred')
        if cfg.research_mode not in {"adaptive", "catalog", "code"} or cfg.console_format not in {"human", "json"}:
            raise AgentError("INVALID_CONFIG", "Unsupported RESEARCH_MODE or CONSOLE_LOG_FORMAT")
        if cfg.graph_peers > cfg.max_workers or cfg.graph_train_workers > cfg.max_workers:
            raise AgentError("INVALID_CONFIG", "GRAPH_PEERS/GRAPH_TRAIN_WORKERS must not exceed MAX_WORKERS")
        if cfg.graph_max_live_branches > cfg.graph_max_total_branches:
            raise AgentError("INVALID_CONFIG", "Live graph limit must not exceed total graph limit")
        if max(cfg.llm_max_output_tokens, cfg.llm_bootstrap_output_tokens) > cfg.llm_max_output_tokens_limit:
            raise AgentError("INVALID_CONFIG", "Initial output token limits must not exceed LLM_MAX_OUTPUT_TOKENS_LIMIT")
        return cfg

    def public_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
