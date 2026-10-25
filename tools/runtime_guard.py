"""Worker-side IO guard for explicit generated pipelines; not an OS sandbox."""
from pathlib import Path
import os
import site
import sys
import sysconfig
import threading


def prepare_pipeline_imports(code_path, base_recipe_id=None, parameters=None):
    """Initialize declared allowed libraries before sealing generated execution.

    Only package imports run here, never the generated module or its functions.
    Native libraries must load before threadpool limits are configured as well.
    """
    import ast
    import importlib
    from tools.code import validate_code

    tree = validate_code(Path(code_path).read_text(encoding='utf-8'))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    if isinstance(base_recipe_id, str):
        if base_recipe_id.startswith('catboost-'):
            modules.add('catboost')
        elif base_recipe_id.startswith('linear-'):
            modules.add('sklearn.linear_model')
        elif base_recipe_id.startswith(('hist-gradient-', 'random-forest-', 'extra-trees-')):
            modules.add('sklearn.ensemble')
    def inspect_parameters(params):
        recipe=params.get('base_recipe_id','')
        if recipe.startswith('catboost-'):modules.add('catboost')
        if recipe.startswith('lightgbm-'):modules.add('lightgbm')
        if recipe.startswith('tabular-net-'):modules.update({'torch','torch._dynamo'})
        source=params.get('representation_source')
        if source:
            from agent.specialists import validate_feature_source
            feature_tree=validate_feature_source(source)
            for node in ast.walk(feature_tree):
                if isinstance(node,ast.Import):modules.update(a.name for a in node.names)
                elif isinstance(node,ast.ImportFrom):modules.add(node.module)
        for member in params.get('members',[]):inspect_parameters(member)
    inspect_parameters(parameters or {})
    for name in sorted(modules):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Leave absent optional packages to the generated code's own import
            # handling. A required absent import still fails there normally.
            if exc.name != name.split('.')[0]:
                raise
    if any(m=='torch' for m in modules):
        import torch
        torch.optim.AdamW(torch.nn.Linear(1,1).parameters())


class PipelineGuard:
    def __init__(self, code_path, trusted_thread=None):
        import agent.ml
        self.active = False
        self.trusted_thread = trusted_thread
        self.code = Path(code_path).resolve()
        # A venv's packages need not live under base_prefix (e.g. /usr).
        # Trust installation directories, not the whole venv/project directory.
        roots = [sys.base_prefix, *site.getsitepackages()]
        roots.extend(sysconfig.get_path(name) for name in ('stdlib', 'platstdlib', 'purelib', 'platlib'))
        if site.ENABLE_USER_SITE:
            roots.append(site.getusersitepackages())
        self.libraries = tuple(dict.fromkeys(Path(root).resolve() for root in roots if root))
        self.runtime = Path(__file__).resolve().parents[1]
        sys.addaudithook(self.check)

    def check(self, event, args):
        if not self.active:
            return
        if threading.current_thread() is self.trusted_thread:
            return
        if event.startswith(('socket.', 'subprocess.', 'ctypes.')) or event in {'os.system', 'os.exec', 'os.posix_spawn', 'os.fork'}:
            raise PermissionError('Generated pipeline cannot start processes or use network/native loaders')
        if event == 'open':
            filename, mode, flags = args
            if isinstance(filename, int):
                return
            # Resolve in the hook without opening files. The generated source is
            # the sole readable run artifact; datasets/job files remain sealed.
            path = Path(os.fsdecode(filename)).resolve()
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            owned_source = (any(path.is_relative_to(self.runtime / name) for name in ('agent', 'tools'))
                            and path.suffix in {'.py', '.pyc'})
            resource_info = str(path) in {'/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                '/sys/fs/cgroup/cpu/cpu.cfs_period_us', '/sys/fs/cgroup/cpuset.cpus.effective', '/proc/cpuinfo'}
            readable = path == self.code or any(path.is_relative_to(root) for root in self.libraries) or owned_source or resource_info
            if writing or not readable:
                raise PermissionError('Generated pipeline may read installed code only; data is supplied by Runner; blocked path: '+str(path))
        if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate', 'os.chmod'}:
            raise PermissionError('Generated pipeline cannot mutate files')
