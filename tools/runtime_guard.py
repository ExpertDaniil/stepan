"""Worker-side IO guard for explicit generated pipelines; not an OS sandbox."""
from pathlib import Path
import os
import site
import sys
import sysconfig
import threading


# Owned recipe wrappers import their backend lazily inside fit/predict, so a
# recipe ID (not an import statement) is what reveals the native dependency.
RECIPE_MODULES = (('catboost-', ('catboost',)), ('lightgbm-', ('lightgbm',)), ('strong-', ('lightgbm',)),
                  ('linear-', ('sklearn.linear_model',)),
                  (('hist-gradient-', 'hist-native-', 'random-forest-', 'extra-trees-'), ('sklearn.ensemble',)))
OWNED_MODULES = {'agent.strong': ('lightgbm',)}


def recipe_modules(value):
    """Backends of every recipe ID nested in parameters or source literals."""
    found = set()
    if isinstance(value, str):
        for prefix, names in RECIPE_MODULES:
            if value.startswith(prefix):
                found.update(names)
    elif isinstance(value, dict):
        for item in value.values():
            found |= recipe_modules(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= recipe_modules(item)
    return found


def prepare_pipeline_imports(code_path, parameters=None):
    """Initialize declared allowed libraries before sealing generated execution.

    Only package imports run here, never the generated module or its functions.
    Native libraries must load before threadpool limits are configured as well.
    ``parameters`` is the plan hyperparameter mapping (or a base recipe ID).
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
        elif isinstance(node, ast.Constant):
            modules |= recipe_modules(node.value)
    modules |= recipe_modules(parameters)
    for name in list(modules):
        modules.update(OWNED_MODULES.get(name, ()))
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


class GuardViolation(PermissionError):
    """A sealed-execution refusal; ``origin`` says whose call reached the guard."""

    def __init__(self, message, origin):
        super().__init__(message)
        self.origin = origin


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
            self.refuse('Generated pipeline blocked audit event '+event+
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
                self.refuse('Generated pipeline may read installed code only; data is supplied by Runner; blocked path: '+str(path))
        if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate', 'os.chmod'}:
            self.refuse('Generated pipeline cannot mutate files')

    def origin(self):
        """Nearest non-library caller: the generated source or an owned wrapper.

        An owned wrapper (agent/, tools/) reaching the guard is an execution
        environment defect, not a property of the candidate's hypothesis.
        """
        frame = sys._getframe(2)
        guard_file = Path(__file__).resolve()
        while frame is not None:
            path = Path(frame.f_code.co_filename).resolve()
            if path == self.code:
                return 'generated_source'
            if path != guard_file and any(path.is_relative_to(self.runtime/name) for name in ('agent', 'tools')):
                return 'owned_runtime'
            frame = frame.f_back
        return 'installed_library'

    def refuse(self, message):
        origin = self.origin()
        raise GuardViolation(message+' (origin: '+origin+')', origin)

