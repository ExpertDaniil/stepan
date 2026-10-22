"""Provider-neutral CI entry point, using stdlib unittest."""
from pathlib import Path
import argparse
import compileall
import re
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def case_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from case_ids(item)
        else:
            yield item.id()


def parallel_check(jobs):
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
    cases = list(case_ids(suite))
    if len(cases) != len(set(cases)):
        raise ValueError('Duplicate discovered test IDs')
    started = time.monotonic()
    total, failed = 0, False
    with tempfile.TemporaryDirectory(prefix='stepan-check-') as temp:
        children = []
        for number in range(min(jobs, len(cases))):
            group = cases[number::jobs]
            log = Path(temp) / f'group-{number+1}.log'
            with log.open('w') as stream:
                process = subprocess.Popen([sys.executable, '-m', 'unittest', *group, '-v'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            children.append((process, log))
            print(f'Test group {number+1}: {len(group)} cases; log={log}', flush=True)
        for number, (process, log) in enumerate(children, 1):
            code = process.wait()
            output = log.read_text()
            counts = re.findall(r'Ran (\d+) tests? in ', output)
            total += int(counts[-1]) if counts else 0
            failed = failed or code != 0
            print(f'\nTest group {number}: exit={code}\n{output}', flush=True)
    failed = failed or total != len(cases)
    print(f'\nRan {total} tests in {time.monotonic()-started:.3f}s (parallel jobs={jobs})\n')
    print('FAILED (see group results)' if failed else 'OK')
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, choices=range(1, 4), default=1, help='Independent test processes; default preserves sequential execution')
    args = parser.parse_args()
    if not compileall.compile_dir(ROOT / "agent", quiet=1) or not compileall.compile_dir(ROOT / "tools", quiet=1):
        raise SystemExit(1)
    if args.jobs > 1:
        raise SystemExit(parallel_check(args.jobs))
    raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT))
