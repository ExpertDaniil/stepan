"""Worker-side IO guard for explicit generated pipelines; not an OS sandbox."""
from pathlib import Path
import os
import site
import sys
import sysconfig
import threading


def prepare_pipeline_imports(code_path, base_recipe_id=None):
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
        elif base_recipe_id.startswith('lightgbm-'):
            modules.add('lightgbm')
    for name in sorted(modules):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Leave absent optional packages to the generated code's own import
            # handling. A required absent import still fails there normally.
            if exc.name != name.split('.')[0]:
                raise
    preload_native(modules)


# ctypes resolves C symbols lazily: the first call of each LGBM_*/XG* function
# raises a ctypes.dlsym audit event, which the sealed guard must refuse. Resolve
# the installed library's own declared API (never generated code) beforehand.
NATIVE_CTYPES_PACKAGES = {'lightgbm': ('lightgbm.basic', 'LGBM_'), 'xgboost': ('xgboost.core', 'XG')}


def _resolve_ctypes_api(package, owner, prefix):
    import ast
    import importlib
    module = importlib.import_module(owner)
    library = getattr(module, '_LIB', None)
    if library is None:
        return 0
    root = Path(importlib.import_module(package).__file__).resolve().parent
    names = set()
    for source in root.rglob('*.py'):
        try:
            tree = ast.parse(source.read_text(encoding='utf-8'))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        names.update(node.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                     and node.value.id == '_LIB' and node.attr.startswith(prefix))
    resolved = 0
    for name in sorted(names):
        try:
            getattr(library, name)
            resolved += 1
        except AttributeError:
            pass  # Optional C APIs keep their normal installed-version error.
    return resolved


def preload_native(modules):
    """Load native ctypes APIs and sklearn's threadpool controller before sealing."""
    import importlib
    roots = {name.split('.')[0] for name in modules if name}
    for package, (owner, prefix) in NATIVE_CTYPES_PACKAGES.items():
        if package not in roots:
            continue
        try:
            _resolve_ctypes_api(package, owner, prefix)
        except ModuleNotFoundError:
            pass
    try:
        # Its first construction uses ctypes.dlopen/dlsym; later calls reuse it.
        importlib.import_module('sklearn.utils.parallel')._get_threadpool_controller()
    except (ModuleNotFoundError, AttributeError):
        pass


# Buffer helpers used by already loaded native libraries; loaders stay blocked.
CTYPES_BUFFER_EVENTS = frozenset({'ctypes.create_string_buffer', 'ctypes.create_unicode_buffer', 'ctypes.addressof',
                                  'ctypes.cdata', 'ctypes.cdata/buffer', 'ctypes.string_at', 'ctypes.wstring_at'})


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
        if (event.startswith(('socket.', 'subprocess.')) or
                (event.startswith('ctypes.') and event not in CTYPES_BUFFER_EVENTS) or
                event in {'os.system', 'os.exec', 'os.posix_spawn', 'os.fork'}):
            raise PermissionError('Generated pipeline blocked audit event '+event+
                                  ': processes, network and new native loaders are forbidden')
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
