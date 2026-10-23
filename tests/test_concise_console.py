"""Verify the terminal communicates measured facts without losing the full log."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.config import Config
from agent.errors import AgentError
from agent.observability.logging import RunLogger
from scripts.configure import recommended_settings


class ConciseConsoleTests(unittest.TestCase):
    def test_meaningful_run_is_readable_and_full_records_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(temp, 'console-unit', console_format='concise', secrets=['unit-private-secret'])
            output = io.StringIO()
            try:
                with redirect_stderr(output):
                    log.event('contract.completed', contract_id='C10', large_detail={'raw': 'keep this'})
                    log.event('artifact.registered', path='private-artifact')
                    log.event('run.state_changed', state='baseline')
                    log.event('research.hypothesis_proposed', researcher_id='researcher-1',
                              hypothesis='Проверить сглаживание редких значений', recipe_id='catboost-regressor-v1',
                              hyperparameters={'depth': 6, 'iterations': 300})
                    log.event('search.admission', experiment_id='exp-1', allowed=True,
                              estimated_seconds=12, remaining_seconds=150, reserve_seconds=30)
                    log.event('job.started', experiment_id='exp-1', phase='train', recipe_id='catboost-regressor-v1',
                              hyperparameters={'depth': 6}, cpu_threads=3, memory_limit_mb=10240, timeout_seconds=60)
                    log.event('experiment.evaluated', experiment_id='exp-1', metric_id='rmse', score=1.25,
                              improvement_over_baseline=-.25, direction='minimize', diagnostics_artifact_id='full-details')
                    log.event('submission.published', experiment_id='exp-1', output_path='/task/submission.csv',
                              score=1.25, champion_version=2)
                    log.event('search.proposal_replaced', work_order_id='work-2', reason='predicted_duration')
                    log.event('graph.branch_state_changed', branch_id='branch-2', state='paused', reason='Проверка альтернативы')
                    log.event('search.finished', stop_reason='time_reserve',
                              budget={'wall_seconds_remaining': 30, 'finalization_reserve_seconds': 29})
                    log.event('run.failed', level='ERROR', code='UNIT_FAILURE', message='unit-private-secret failed',
                              traceback='full traceback belongs in events.jsonl')
            finally:
                log.close()
            text = output.getvalue()
            for phrase in ('первый проверенный ответ', 'явная гипотеза', '"depth": 6', 'прогноз 12 с',
                           'осталось 150 с, запас 30 с', 'Потоки: 3', 'rmse = 1.25',
                           'улучшение относительно исходного решения -0.25', 'Ответ проверен и сохранён',
                           '/task/submission.csv', 'Предложение заменено', 'превышает остаток времени',
                           'ветви branch-2', 'приостановлена', 'запас времени', '[REDACTED]'):
                self.assertIn(phrase, text)
            for hidden in ('contract.completed', 'artifact.registered', 'private-artifact',
                           'full traceback belongs', 'unit-private-secret'):
                self.assertNotIn(hidden, text)
            records = [json.loads(line) for line in (Path(temp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 12)
            self.assertEqual(records[0]['large_detail'], {'raw': 'keep this'})
            self.assertEqual(records[-1]['traceback'], 'full traceback belongs in events.jsonl')
            self.assertNotIn('unit-private-secret', json.dumps(records))

    def test_waiting_is_reported_once_per_minute_and_only_in_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(temp, 'waiting', console_format='concise')
            output = io.StringIO()
            try:
                with redirect_stderr(output):
                    for elapsed in (15, 45, 60, 75, 120):
                        log.event('llm.waiting', call_no=1, elapsed_seconds=elapsed, run_seconds_remaining=500-elapsed)
            finally:
                log.close()
            self.assertEqual(output.getvalue().count('Ответ языковой модели:'), 2)
            self.assertEqual(len((Path(temp)/'events.jsonl').read_text().splitlines()), 5)

    def test_successful_worker_exit_is_not_presented_as_a_verified_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(temp, 'worker', console_format='concise')
            output = io.StringIO()
            try:
                with redirect_stderr(output):
                    log.event('job.finished', state='succeeded', experiment_id='exp-1')
                    log.event('submission.refit_deferred', experiment_id='exp-2', estimated_seconds=100, remaining_seconds=20)
                    log.event('research.analysis_completed', kind='numeric_correlations', columns=['x'], rows=72,
                              scope='fit', artifact_id='analysis-1')
            finally:
                log.close()
            text = output.getvalue()
            self.assertNotIn('Ответ проверен и сохранён', text)
            self.assertIn('Предыдущий проверенный ответ сохранён', text)
            self.assertIn('строк 72, область fit', text)

    def test_routine_scheduling_is_quiet_but_branch_switch_has_explicit_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(temp, 'decisions', console_format='concise')
            output = io.StringIO()
            try:
                with redirect_stderr(output):
                    log.event('search.decision', action='CONTINUE', branch_id='a',
                              reason='Round-robin work on equal peers', parameters={})
                    log.event('search.decision', action='BACKTRACK', branch_id='a',
                              reason='Measured plateau; resume a saved alternative', parameters={'target_branch_id': 'b'})
            finally:
                log.close()
            self.assertEqual(output.getvalue().count('Решение:'), 1)
            self.assertIn('измерено плато', output.getvalue())
            self.assertIn('target_branch_id', output.getvalue())

    def test_json_and_default_human_remain_explicitly_available(self):
        for format in ('json', 'human'):
            with self.subTest(format=format), tempfile.TemporaryDirectory() as temp:
                log = RunLogger(temp, format, console_format=format)
                output = io.StringIO()
                try:
                    with redirect_stderr(output):
                        log.event('contract.completed', contract_id='C01', detail={'x': 1})
                finally:
                    log.close()
                if format == 'json':
                    self.assertEqual(json.loads(output.getvalue())['detail'], {'x': 1})
                else:
                    self.assertIn('contract.completed', output.getvalue())
                    self.assertIn('contract_id=C01', output.getvalue())

    def test_default_and_configuration_support_all_three_formats(self):
        self.assertEqual(Config().console_format, 'concise')
        self.assertEqual(recommended_settings()['CONSOLE_LOG_FORMAT'], 'concise')
        for format in ('concise', 'human', 'json'):
            with self.subTest(format=format), patch.dict(os.environ, {'CONSOLE_LOG_FORMAT': format}):
                self.assertEqual(Config.from_env().console_format, format)
        with patch.dict(os.environ, {'CONSOLE_LOG_FORMAT': 'unknown'}):
            with self.assertRaises(AgentError):
                Config.from_env()


if __name__ == '__main__':
    unittest.main()
