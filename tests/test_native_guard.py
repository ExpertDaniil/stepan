"""Sealed generated pipelines can train ctypes-based boosters, but still cannot load new natives."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = '''
import sys
from tools.runtime_guard import PipelineGuard, prepare_pipeline_imports
prepare_pipeline_imports(sys.argv[1])
guard = PipelineGuard(sys.argv[1]); guard.active = True
import numpy as np
X = np.random.RandomState(0).normal(size=(200, 4)); y = X[:, 0]*2 + X[:, 1]
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
for model in (LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1), XGBRegressor(n_estimators=20, n_jobs=1)):
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()
print('boosters_trained')
import ctypes, ctypes.util
try:
    ctypes.CDLL(ctypes.util.find_library('c'))
except PermissionError:
    print('new_loader_blocked')
guard.active = False
'''


class NativeGuardTests(unittest.TestCase):
    def test_lightgbm_and_xgboost_train_inside_the_sealed_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'candidate.py'
            source.write_text('import lightgbm\nimport xgboost\n\ndef make_pipeline(task_spec, params):\n    return None\n')
            done = subprocess.run([sys.executable, '-c', SCRIPT, str(source)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        self.assertIn('boosters_trained', done.stdout)
        self.assertIn('new_loader_blocked', done.stdout)


if __name__ == '__main__':
    unittest.main()
