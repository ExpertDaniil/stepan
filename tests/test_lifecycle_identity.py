"""Identity across research, time estimation and final publication transitions."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.context import output_schemas
from agent.llm.gateway import Gateway
from agent.orchestrator import Orchestrator
from agent.submission import Submission
from tests.graph_helpers import GraphHarness
from tests.helpers import Harness
from tests.test_adaptive_research import action, SOURCE


class LifecycleIdentityTests(unittest.TestCase):
    def test_custom_base_duration_probe_keeps_source_identity_and_reads_old_contexts(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800)
        try:
            reply, payload = h.submit(h.issue(), {'alpha': 1.0}, recipe='python-pipeline-v1', source=SOURCE)
            h.commit(reply, 7)
            g = h.g
            context_id = h.issue()['work_order']['context_id']
            context = g.artifacts.read_json(context_id)
            user = json.loads(context['messages'][1]['content'])
            source_id = user['base_result']['code_artifact_id']
            self.assertEqual(source_id, payload['proposal']['plan']['code_artifact_id'])
            g.budget.phase_deadline = time.monotonic()+9000
            class Transport:
                def complete(self, request, timeout, context):
                    return {'choices': [{'message': {'content': json.dumps(action('stop', hyperparameters={}))},
                                         'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
            args = {'purpose': 'research_proposal', 'context_artifact_id': context_id,
                    'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': [],
                    'max_output_tokens': 1024, 'deadline_seconds': 900, 'attempt_limit': 1}
            for legacy in (False, True):
                if legacy:
                    user['base_result'].pop('code_artifact_id')
                    context['messages'][1]['content'] = json.dumps(user)
                    args['context_artifact_id'] = g.artifacts.write_json('contexts/old-base.json', context)['artifact_id']
                with self.subTest(legacy=legacy), patch.object(g.budget.duration, 'estimate', wraps=g.budget.duration.estimate) as estimate:
                    result = gateway.complete(args)
                    self.assertEqual(result['parsed_payload']['action'], 'stop')
                    probes = [call.args[0] for call in estimate.call_args_list if call.args[0]['model']['family'] == 'custom']
                    self.assertTrue(probes)
                    self.assertTrue(all(p['code_artifact_id'] == ('unresolved-source' if legacy else source_id) for p in probes))
        finally:
            h.close()

    def test_final_confirmation_reselects_previously_published_model_with_fresh_transaction(self):
        h = Harness()
        try:
            weak, strong = h.evaluation(.6), h.evaluation(.8)
            selected = [weak['experiment_id']]
            submission = Submission(h.spec, h.artifacts, h.store, h.log)
            candidates = {weak['experiment_id']: h.candidate('B'), strong['experiment_id']: h.candidate('A')}
            jobs, outcomes = {}, {}
            # Training is simulated here; bus validation, output validation,
            # idempotency, champion CAS and actual CSV publication are real.
            def start_refit(payload, spec, experiment):
                job_id = 'job-'+payload['submission_job_id']
                job = SimpleNamespace(job_id=job_id)
                jobs[job_id] = job
                outcomes[job_id] = {'state': 'succeeded', 'output_artifact_ids': [candidates[experiment['experiment_id']]['artifact_id']]}
                return {'job_id': job_id, 'state': 'accepted', 'completion_contract': 'C15'}
            def lease(owner, phase, resources):
                value = deepcopy(h.schemas.operations['C19']['request_example']['payload']['lease'])
                value['lease_id'] = 'lease-'+owner
                return {'granted': True, 'lease': value}
            budget = SimpleNamespace(set_final_plan=lambda plan: None, remaining=lambda: 1000,
                                     output_reserve=lambda: 2, duration=SimpleNamespace(estimate=lambda *args: 1))
            controller = SimpleNamespace(final_candidate=lambda: selected[0], failed_refits=set(), store=h.store,
                budget=budget, submission=submission, spec=h.spec, config=h.config, log=h.log, artifacts=h.artifacts,
                resources=lambda *args: {}, lease=lease, runner=SimpleNamespace(jobs=jobs, start_refit=start_refit),
                wait_jobs=lambda batch: {job.job_id: outcomes[job.job_id] for job in batch}, update_final_reserve=lambda: None,
                call=lambda cid, value, handler, key=None: h.bus.call(cid, value, handler, key=key)['data'])
            Orchestrator._finalize_best(controller)
            selected[0] = strong['experiment_id']
            Orchestrator._finalize_best(controller)
            self.assertEqual(h.store.champion()['champion_version'], 2)
            confirmation = {'chosen_experiment_id': weak['experiment_id'], 'expected_champion_version': 2,
                            'prior_champion_experiment_id': strong['experiment_id']}
            art = h.artifacts.write_json('validation/final-confirmed.json', confirmation, 'final_confirmation')
            submission.final_confirmation = {**confirmation, 'artifact_id': art['artifact_id']}
            selected[0] = weak['experiment_id']
            Orchestrator._finalize_best(controller)
            self.assertEqual(h.store.champion()['experiment_id'], weak['experiment_id'])
            self.assertEqual(h.store.champion()['champion_version'], 3)
            self.assertEqual(len(jobs), 3)
            self.assertTrue(submission.final_confirmation['consumed'])
            self.assertEqual(h.artifacts.resolve(candidates[weak['experiment_id']]['artifact_id']).read_bytes(),
                             (h.task.parent/'submission.csv').read_bytes())
        finally:
            h.close()
