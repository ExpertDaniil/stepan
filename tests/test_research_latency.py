from copy import deepcopy
from dataclasses import replace
import json
import unittest

from agent.context import output_schemas
from agent.llm.gateway import Gateway
from scripts.analyze_run import job_occupancy
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tools.analysis import AnalysisService


class ResearchLatencyTests(unittest.TestCase):
    def test_zero_tool_wall_budget_preserves_final_and_json_repair(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
                         llm_max_calls=None, llm_max_total_output_tokens=None)
        try:
            good = action('catalog')
            requests = []
            class Transport:
                def complete(self, request, timeout, context):
                    requests.append(deepcopy(request))
                    return {'choices': [{'message': {'content': json.dumps(good)[:-1] if len(requests) == 1 else json.dumps(good)},
                                         'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
            g = h.g
            config = replace(g.config, llm_research_tool_seconds=0)
            gateway = Gateway(config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
            gateway.analysis = AnalysisService(g.spec, g.validation, g.artifacts, g.budget, g.log)
            result = gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': h.issue()['work_order']['context_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['analyze_data'],
                'max_output_tokens': 1024, 'deadline_seconds': 900, 'attempt_limit': 10})
            self.assertEqual(result['parsed_payload'], good)
            self.assertEqual(len(requests), 2)
            self.assertTrue(all('tools' not in r for r in requests))
            self.assertFalse(g.budget.llm_dialogues)
        finally:
            h.close()

    def test_overlapping_jobs_are_not_double_counted_and_open_tail_is_busy(self):
        def event(second, name, job=None):
            return {'timestamp': '2026-01-01T00:00:'+str(second).zfill(2)+'+00:00', 'event': name, 'job_id': job}
        report = job_occupancy([event(0, 'run.started'), event(5, 'job.started', 'a'),
            event(7, 'job.started', 'b'), event(10, 'job.finished', 'a'), event(12, 'job.finished', 'b'),
            event(15, 'job.started', 'c'), event(20, 'job.progress', 'c')])
        self.assertEqual(report['job_present_seconds'], 12)
        self.assertEqual(report['no_train_or_refit_seconds'], 8)
        self.assertEqual(report['unfinished_jobs'], 1)
