import ast
import hashlib
from pathlib import Path
import re

from agent.errors import AgentError

ALLOWED_IMPORTS = {"agent.ml", "agent.feature_engineering", "agent.learned_features", "agent.specialists", "agent.ensemble", "numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm", "xgboost", "math"}
BLOCKED_CALLS = {"exec", "eval", "open", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr", "delattr", "vars", "breakpoint"}
BLOCKED_ATTRIBUTES = {'os', 'sys', 'environ', 'load', 'save', 'loadtxt', 'savetxt', 'genfromtxt', 'memmap',
                      'fromfile', 'tofile', 'ctypes', 'f2py', 'system', 'popen', 'read_pickle', 'to_pickle', 'eval', 'query'}


def validate_code(content):
    if len(content.encode("utf-8")) > 262144:
        raise AgentError("INVALID_CODE", "Generated code exceeds 256 KiB")
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise AgentError("INVALID_CODE", f"Invalid Python syntax: {exc}") from exc
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "make_pipeline"]
    if len(functions) != 1 or len(functions[0].args.args) != 2:
        raise AgentError("INVALID_CODE", "Expected one make_pipeline(task_spec, params) function")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
                continue
            except (ValueError, TypeError):
                pass
        raise AgentError("INVALID_CODE", "Top-level executable statements are not allowed")
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.level:
                raise AgentError("INVALID_CODE", "Relative imports are not allowed")
        for module in modules:
            if not any(module == x or module.startswith(x + ".") for x in ALLOWED_IMPORTS):
                raise AgentError("INVALID_CODE", f"Import not allowed: {module}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise AgentError("INVALID_CODE", f"Call not allowed: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise AgentError("INVALID_CODE", "Dunder attribute access is not allowed")
        if isinstance(node, ast.Attribute) and (node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith('read_')
                or node.attr in {'to_csv', 'to_json', 'to_excel', 'to_parquet', 'to_feather', 'to_sql', 'to_hdf', 'to_clipboard', 'to_stata'}):
            raise AgentError('INVALID_CODE', 'Generated pipelines receive data from Runner; filesystem/dynamic IO is not allowed: '+node.attr)
    return tree


def write_code(payload, artifacts):
    if Path(payload["filename"]).name != payload["filename"] or not payload["filename"].endswith(".py"):
        raise AgentError("PATH_OUTSIDE_WORKSPACE", "Code filename must be a plain .py filename")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", payload["researcher_id"]):
        raise AgentError("INVALID_CODE", "Invalid researcher id")
    validate_code(payload["content"])
    digest = hashlib.sha256(payload["content"].encode()).hexdigest()
    art = artifacts.write_text(f"code/{payload['researcher_id']}/{digest}.py", payload["content"], "python_source", "text/x-python")
    return {"artifact": art, "syntax_valid": True, "entrypoint": "make_pipeline"}
