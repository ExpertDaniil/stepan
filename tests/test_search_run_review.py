from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.catalog import Catalog
from agent.config import Config
from agent.evaluation import error_diagnostics
from agent.graph.controller import warmup_ticket, lineage_for
from agent.runner import Runner
from agent.worker import save_snapshot
from scripts.analyze_run import inspect_run
from tests.graph_helpers import GraphHarness


class SearchRunReviewTests(unittest.TestCase):
    def assess(self, h, branch=None):
        branch = branch or h.root
        return h.g.invoke('G07', {'branch_id': branch, 'expected_branch_revision': h.g.branch(branch)['branch_revision']})['assessment']

    def test_fork_seeds_measured_own_models_and_keeps_causal_checkpoint(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork()
            cp = deepcopy(h.g.checkpoint_record(fork['checkpoint_id']))
            _, improved, _, _ = h.result('researcher-1', {'alpha': 7}, 7)
            failed = h.issue();bad, _ = h.submit(failed, {'alpha': 99})
            h.g.store.apply_event(h.train_event(bad, state='failed'), 'train')
            best = h.g.store.best()
            result = h.g.invoke('G02', fork)
            accept = result['accept_branch']['branch_id'];reject = result['reject_branch']['branch_id']
            own = next(x for x in h.g.heads(reject) if x['researcher_id'] == 'researcher-1')
            self.assertEqual(own['experiment_id'], improved['lead_submission']['experiment_id'])
            self.assertEqual(h.g.checkpoint_record(fork['checkpoint_id']), cp)
            self.assertEqual(h.g.store.best(), best)
            self.assertTrue(all(h.g.head_satisfies(x, result['accept_branch']['constraints']) for x in h.g.heads(accept)))
            seeds = [json.loads(x[0])['record'] for x in h.g.db.execute("SELECT record_json FROM g_events WHERE json_extract(record_json,'$.record_schema')='GBranchSeed'")]
            self.assertEqual(len(seeds), 4)
            for seed in seeds:
                h.g.schemas.validate(seed, 'GBranchSeed')
                self.assertLess(seed['selection_cut'], h.g.cut())
            exported = h.g.invoke('G11', {'task_spec_id': h.g.spec['task_spec_id'], 'validation_id': h.g.validation['validation_id'], 'selection_policy_id': 'same-protocol-primary-v1', 'scope': 'final'})
            graph = h.g.artifacts.read_json(exported['graph_artifact']['artifact_id'])
            self.assertEqual(len(graph['branch_seeds']), 4)
            h.g.schemas.validate(graph, 'GGraphExport')
        finally:
            h.close()

    def test_warmup_bases_precede_feature_variants(self):
        catalog = Catalog('regression', Config())
        choices = catalog.warmup_choices()[:4]
        self.assertNotEqual(choices[0][0], choices[1][0])
        self.assertEqual([p.get('target_transform', 'identity') for _, p in choices[:2]], ['identity', 'identity'])
        self.assertTrue(any(p.get('target_transform') == 'shifted_log1p' for _, p in choices[2:]))

    def test_planned_warmup_is_reused_as_the_controlled_transfer_pilot(self):
        h = GraphHarness()
        try:
            first = h.issue('researcher-1');base, _ = h.submit(first, {}, recipe='hist-gradient-regressor-v1');h.commit(base, 50)
            h.result('researcher-2', {'alpha': 2}, 40)
            source = h.issue('researcher-1');candidate, _ = h.submit(source, {'target_transform': 'shifted_log1p'}, recipe='hist-gradient-regressor-v1');h.commit(candidate, 45)
            h.g.offer_component(candidate['lead_submission']['experiment_id'])
            h.controller.graph = h.g
            h.controller.call = lambda cid, payload, fn: h.g.bus.call(cid, payload, fn)['data']
            params = h.g.catalog.normalize('linear-regressor-v1', {'alpha': 2, 'target_transform': 'shifted_log1p'})
            plan, ticket = warmup_ticket(h.controller, h.root, 'researcher-2', 'linear-regressor-v1', params)
            self.assertEqual(ticket['work_order']['action'], 'transfer')
            self.assertEqual(plan['model']['hyperparameters'], params)
            self.assertEqual(h.g.budget.calls, 0)
            reply = h.g.invoke('G04', {'work_order_id': ticket['work_order']['work_order_id'],
                'proposal': {'proposal_id': plan['plan_id'], 'researcher_id': plan['researcher_id'], 'plan': plan, 'decision': 'propose', 'stop_reason': None},
                'manifest': h.g.constructor.build(plan), 'lineage': lineage_for(ticket['work_order'])})
            h.commit(reply, 35);h.g.finish_pilot(ticket['work_order']['work_order_id'], reply['lead_submission']['experiment_id'])
            offer = json.loads(h.g.db.execute('SELECT record_json FROM g_idea_candidates').fetchone()[0])
            self.assertEqual(offer['state'], 'confirmed')
        finally:
            h.close()

    def test_seed_is_an_explicit_memory_exception_without_sibling_history(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork()
            _, own, _, _ = h.result('researcher-1', {'alpha': 7}, 7)
            result = h.g.invoke('G02', fork)
            reject = result['reject_branch']['branch_id'];accept = result['accept_branch']['branch_id']
            h.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0, 'action': 'resume', 'reason': 'fixture'})
            h.result('researcher-1', {'alpha': 8}, 6, branch=reject)
            own_history = h.g.visible_history(reject, 'researcher-1')
            self.assertIn(own['lead_submission']['experiment_id'], own_history)
            _, sibling, _, _ = h.result('researcher-2', {'alpha': 19, 'target_transform': 'shifted_log1p'}, 1, branch=accept)
            self.assertNotIn(sibling['lead_submission']['experiment_id'], h.g.visible_history(reject))
        finally:
            h.close()

    def test_four_consecutive_distinct_trials_required_for_plateau(self):
        h = GraphHarness(graph_peers=1)
        try:
            h.result('researcher-1', {'alpha': 1}, 100)
            for alpha in [2, 3, 4]:
                h.result('researcher-1', {'alpha': alpha}, 101)
                report = self.assess(h)
                self.assertNotEqual(report['status'], 'plateau')
            h.result('researcher-1', {'alpha': 5}, 101)
            report = self.assess(h)
            self.assertEqual(report['status'], 'plateau')
            self.assertEqual(report['consecutive_no_gain'], 4)
            self.assertEqual(report['required_no_gain'], 4)
        finally:
            h.close()

    def test_small_gains_accumulate_and_reset_the_streak(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 100)
            for i, score in enumerate([99.97, 99.94, 99.91, 99.88], start=2):
                _, reply, _, outcome = h.result('researcher-1', {'alpha': i}, score)
                self.assertEqual(outcome['head_results'][0]['disposition'], 'advanced')
            report = self.assess(h)
            self.assertNotEqual(report['status'], 'plateau')
            self.assertEqual(report['consecutive_no_gain'], 0)
            self.assertEqual(h.g.heads(h.root)[0]['experiment_id'], reply['lead_submission']['experiment_id'])
        finally:
            h.close()

    def test_duplicates_seed_replicas_and_failures_do_not_create_a_plateau(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 100)
            h.result('researcher-1', {'alpha': 2}, 101)
            for seed in [43, 44, 45]:
                ticket = h.issue();reply, _ = h.submit(ticket, {'alpha': 2}, seed=seed);h.commit(reply, 101)
            t = h.issue();reply, _ = h.submit(t, {'alpha': 3})
            event = h.train_event(reply, state='failed');h.g.store.apply_event(event, 'train')
            report = self.assess(h)
            self.assertNotEqual(report['status'], 'plateau')
            self.assertEqual(report['consecutive_no_gain'], 1)
        finally:
            h.close()

    def test_classification_uses_the_same_streak_with_maximizing_scores(self):
        h = GraphHarness('classification', graph_peers=1)
        try:
            h.result('researcher-1', {'C': 1}, .8)
            for i in range(2, 6):
                h.result('researcher-1', {'C': i}, .8+i*.00001)
            report = self.assess(h)
            self.assertEqual(report['status'], 'plateau')
            self.assertEqual(report['consecutive_no_gain'], 4)
        finally:
            h.close()

    def test_archived_late_evaluations_do_not_count_toward_plateau(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 100)
            for alpha in range(2, 6):
                ticket = h.issue();reply, _ = h.submit(ticket, {'alpha': alpha})
                revision = h.g.branch(h.root)['branch_revision']
                h.g.invoke('G08', {'branch_id': h.root, 'expected_branch_revision': revision, 'action': 'pause', 'reason': 'fixture pause'})
                h.g.invoke('G08', {'branch_id': h.root, 'expected_branch_revision': revision+1, 'action': 'resume', 'reason': 'fixture resume'})
                result = h.commit(reply, 101)
                self.assertTrue(result['head_results'][0]['disposition'].startswith('archived'))
            report = self.assess(h)
            self.assertNotEqual(report['status'], 'plateau')
            self.assertEqual(report['consecutive_no_gain'], 0)
        finally:
            h.close()

    def test_each_equal_peer_gets_three_exploit_trials_for_one_explore(self):
        h = GraphHarness()
        try:
            for peer in ['researcher-1', 'researcher-2']:
                assignments = []
                for i in range(4):
                    assignments.append(h.g.research_assignment(h.root, peer))
                    h.result(peer, {'alpha': i+1+(10 if peer.endswith('2') else 0)}, 10+i)
                self.assertEqual(assignments.count('exploit_incumbent'), 3)
                self.assertEqual(assignments.count('explore_alternative'), 1)
        finally:
            h.close()

    def test_diagnostics_stay_bounded_and_do_not_export_individual_labels(self):
        import numpy as np
        import pandas as pd
        data = pd.DataFrame({'id': range(2000), 'feature': np.arange(2000)%7, 'label': ['class_'+str(i) for i in range(2000)]})
        task = {'task_kind': 'multiclass_classification', 'excluded_features': ['id'], 'target_column': 'label'}
        result = error_diagnostics(data, task, list(range(1000)), list(range(1000, 2000)), data['label'].iloc[1000:], np.array(['class_1000']*1000))
        self.assertLess(len(json.dumps(result).encode()), 4000)
        self.assertLessEqual(len(result['class_summary']['worst_classes']), 6)
        self.assertLessEqual(len(result['hard_groups']), 4)
        self.assertTrue(all(g['feature'] != 'id' for g in result['hard_groups']))
        self.assertNotIn('prediction', result)

    def test_regression_group_diagnostics_reveal_measured_error_and_reach_memory(self):
        import numpy as np
        import pandas as pd
        h = GraphHarness()
        try:
            data = pd.DataFrame({'x': list(range(100)), 'label': np.zeros(100)})
            diagnostics = error_diagnostics(data, {'task_kind': 'regression', 'excluded_features': [], 'target_column': 'label'}, list(range(60)), list(range(60, 100)), np.zeros(40), np.ones(40)*5)
            self.assertTrue(diagnostics['hard_groups'])
            self.assertEqual(diagnostics['hard_groups'][0]['rmse'], 5)
            _, reply, _, _ = h.result('researcher-1', {'alpha': 1}, 10)
            ev = h.g.store.experiment_evaluation(reply['lead_submission']['experiment_id'])
            artifact = h.g.artifacts.write_json('diagnostics/review.json', diagnostics, 'diagnostics')
            with h.g.store.transaction():
                ev['diagnostics_artifact_id'] = artifact['artifact_id']
                h.g.db.execute('UPDATE evaluations SET record=? WHERE evaluation_id=?', (json.dumps(ev), ev['evaluation_id']))
            memory = h.g.memory_record(reply['lead_submission']['experiment_id'])
            self.assertEqual(memory['diagnostics']['hard_groups'][0]['rmse'], 5)
            h.g.schemas.validate(memory, 'GExperimentMemory')
        finally:
            h.close()

    def test_optional_snapshot_limit_keeps_predictions_and_removes_partial_file(self):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42);X = rng.normal(size=(500, 8));y = rng.integers(0, 8, 500)
        model = RandomForestClassifier(n_estimators=12, n_jobs=1, random_state=42).fit(X, y)
        before = model.predict(X)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'pipeline.joblib'
            report = save_snapshot(model, path, 512)
            self.assertEqual(report['status'], 'size_limit')
            self.assertFalse(path.exists());self.assertFalse(path.with_suffix('.joblib.tmp').exists())
            np.testing.assert_array_equal(model.predict(X), before)
            report = save_snapshot(model, path, 16777216)
            self.assertEqual(report['status'], 'saved')
            self.assertLessEqual(path.stat().st_size, 16777216)

    def test_snapshot_budget_reserves_concurrent_workers_and_has_a_total_ceiling(self):
        h = GraphHarness()
        try:
            cfg = replace(h.g.config, model_snapshot_max_bytes=100, model_snapshot_total_bytes=150)
            runner = Runner(h.g.artifacts, cfg, h.g.budget, h.g.log)
            self.assertEqual(runner.snapshot_allowance(), 100)
            runner.model_snapshot_reservations['first'] = 100
            self.assertEqual(runner.snapshot_allowance(), 50)
            runner.model_snapshot_reservations['second'] = 50
            self.assertEqual(runner.snapshot_allowance(), 0)
            runner.model_snapshot_reservations.clear()
            (h.h.run/'saved.joblib').write_bytes(b'x'*140)
            self.assertEqual(runner.snapshot_allowance(), 10)
        finally:
            h.close()

    def test_worker_logs_are_limited_in_utf8_bytes_and_stream_is_still_drained(self):
        h = GraphHarness()
        try:
            runner = Runner(h.g.artifacts, replace(h.g.config, worker_log_max_bytes=512), h.g.budget, h.g.log)
            stream = io.StringIO('я🙂'*2000+'unit-private-secret')
            target = h.h.run/'unicode.log'
            runner._drain(stream, target)
            self.assertLessEqual(target.stat().st_size, 512)
            content = target.read_text(encoding='utf-8')
            self.assertIn('OUTPUT LIMIT REACHED', content)
            self.assertNotIn('unit-private-secret', content)
            self.assertTrue(stream.closed)
        finally:
            h.close()

    def test_storage_report_counts_models_without_loading_them(self):
        h = GraphHarness()
        try:
            (h.h.run/'untrusted.joblib').write_bytes(b'not a pickle'*100)
            folder = h.h.run/'exports';folder.mkdir()
            (folder/'graph-events.jsonl').write_text('{}\n')
            h.g.log.event('run.finished', state='completed')
            reports = inspect_run(h.h.run)
            self.assertEqual(len(reports), 1)
            report = reports[0]
            self.assertEqual(report['storage']['by_category']['model_snapshots'], 1200)
            self.assertTrue(report['storage']['largest_files'])
        finally:
            h.close()
