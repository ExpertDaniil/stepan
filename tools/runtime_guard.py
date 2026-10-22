"""Worker-side IO guard for explicit generated pipelines; not an OS sandbox."""
from pathlib import Path
import os
import sys
import threading


class PipelineGuard:
    def __init__(self, code_path):
        self.active = False
        self.code = Path(code_path).resolve()
        self.library = Path(sys.base_prefix).resolve()
        self.runtime = Path(__file__).resolve().parents[1]
        sys.addaudithook(self.check)

    def check(self, event, args):
        if not self.active:
            return
        if threading.current_thread().name == 'worker-watchdog':
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
            readable = path == self.code or path.is_relative_to(self.library) or owned_source or resource_info
            if writing or not readable:
                raise PermissionError('Generated pipeline may read installed code only; data is supplied by Runner')
        if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate', 'os.chmod'}:
            raise PermissionError('Generated pipeline cannot mutate files')
