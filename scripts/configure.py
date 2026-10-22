"""Apply recommended non-secret settings while preserving API credentials and paths."""
from pathlib import Path
import argparse
import os
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from agent.config import Config


def recommended_settings(profile="standard"):
    if profile not in {"standard", "extended"}:
        raise ValueError("Unknown research profile")
    config = Config()
    keys = ("research_mode", "catalog_warmup_experiments", "research_series_size", "llm_max_calls",
            "llm_timeout_seconds", "llm_operation_timeout_seconds", "llm_bootstrap_output_tokens",
            "llm_max_output_tokens", "llm_max_output_tokens_limit", "llm_max_total_output_tokens", "llm_progress_seconds",
            "graph_plateau_window", "graph_plateau_seconds", "graph_plateau_absolute_epsilon", "graph_plateau_relative_epsilon", "graph_exploit_steps", "graph_search_fraction",
            "model_snapshot_max_bytes", "model_snapshot_total_bytes", "worker_log_max_bytes")
    settings = {**{key.upper():str(getattr(config,key)) for key in keys}, "CONSOLE_LOG_FORMAT":"human", "LOG_LEVEL":"INFO", "LOG_MODEL_IO":"1"}
    settings['GRAPH_REFINEMENT_ENABLED'] = '1'
    if profile == "extended":
        settings.update(LLM_MAX_CALLS="96")
    return settings


def configure(path, apply=False, profile="standard"):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Choose an existing regular .env file; create it from .env.example if needed")
    settings = recommended_settings(profile)
    original = path.read_text(encoding='utf-8')
    if profile == 'extended' and re.search(r'^\s*(?:export\s+)?LLM_MAX_TOTAL_OUTPUT_TOKENS\s*=', original, re.MULTILINE):
        settings.pop('LLM_MAX_TOTAL_OUTPUT_TOKENS')
    lines, seen = [], set()
    for line in original.splitlines(keepends=True):
        match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=',line)
        key = match[1] if match else None
        if key in settings:
            if key not in seen:
                lines.append(key+'='+settings[key]+'\n')
                seen.add(key)
        else:
            lines.append(line)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    lines.extend(key+'='+value+'\n' for key,value in settings.items() if key not in seen)
    if apply:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary,0o600)
                stream.write(''.join(lines))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,default=Path('.env'))
    parser.add_argument('--profile', choices=['standard','extended'], default='standard', help='Extended raises only the LLM call ceiling; both profiles search by available time')
    parser.add_argument('--apply',action='store_true',help='Write settings atomically; otherwise show only proposed non-secret values')
    args = parser.parse_args()
    settings = configure(args.env_file,args.apply,args.profile)
    print(('Applied' if args.apply else 'Proposed')+' non-secret settings:')
    for key,value in settings.items():
        print(key+'='+value)
    print('API credentials, LLM_MODE and task/work paths are preserved. Process environment still takes precedence over .env.')


if __name__ == '__main__':
    main()
