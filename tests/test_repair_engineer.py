"""Protect research intent while isolating interface-only repair."""
from copy import deepcopy
import json
import unittest

from agent.llm.repair_engineer import prepare, accept, messages
from tests.helpers import Harness


class EngineerContractTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.original = {'action': 'features', 'recipe_id': 'python-pipeline-v1',
                         'hyperparameters': {'alpha': 2},
                         'feature_program': [{'op': 'target_mean', 'name': 'f', 'columns': ['x']}],
                         'slice_remedies': ['report#one'], 'hypothesis': 'Keep the chosen experiment',
                         'code': None, 'tuning': None, 'feature_steps': [], 'evidence_ids': ['e1']}
        self.base = {'experiment_id': 'pinned-base', 'recipe_id': 'python-pipeline-v1',
                     'hyperparameters': {'base_recipe_id': 'linear-regressor-v1'}}

    def tearDown(self):
        self.h.close()

    def packet(self):
        return prepare(json.dumps(self.original), 'INVALID_RESEARCH_ACTION: features needs an owned recipe', {}, self.base)

    def reply(self, candidate):
        return json.dumps({'status': 'repaired', 'candidate': candidate, 'reason': 'Fix recipe interface only'})

    def test_repairs_owned_recipe_without_changing_experiment(self):
        p = self.packet()
        candidate = {**self.original, 'recipe_id': 'linear-regressor-v1'}
        self.assertEqual(accept(self.reply(candidate), p, self.h.bus.schemas._check), candidate)
        self.assertEqual(p['base_experiment_id'], 'pinned-base')
        self.assertEqual(self.original['recipe_id'], 'python-pipeline-v1')
        self.assertEqual(len(messages(p)), 2)

    def test_rejects_changes_to_every_scientific_field_and_omissions(self):
        p = self.packet()
        changes = {'action': 'catalog', 'hyperparameters': {'alpha': 3}, 'feature_program': [],
                   'slice_remedies': [], 'hypothesis': 'Another experiment', 'evidence_ids': [],
                   'code': 'new source', 'unknown': 1, 'recipe_id': 'catboost-regressor-v1'}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                accept(self.reply({**self.original, key: value}), p, self.h.bus.schemas._check)
        candidate = {**self.original, 'recipe_id': 'linear-regressor-v1'}
        del candidate['hyperparameters']
        with self.assertRaises(ValueError):
            accept(self.reply(candidate), p, self.h.bus.schemas._check)

    def test_conflicting_remedies_and_unreadable_json_use_existing_path(self):
        self.assertIsNone(prepare(json.dumps(self.original), 'INVALID_SLICE_REMEDY: 1..4 corrections', {}, self.base))
        self.assertIsNone(prepare('{broken', 'bad json', {}, self.base))
        without_base = {**self.original, 'code': None, 'tuning': None, 'feature_steps': []}
        self.assertIsNone(prepare(json.dumps(without_base), 'bad recipe', {}, None))

    def test_nonempty_code_and_parameter_search_are_never_discarded(self):
        original = {**self.original, 'code': 'def make_pipeline(): pass', 'tuning': {'search_space': {'a': 1}}}
        p = prepare(json.dumps(original), 'incompatible fields', {}, self.base)
        self.assertNotIn('code', p['allowed_edits'])
        self.assertNotIn('tuning', p['allowed_edits'])

    def test_large_source_is_skipped_without_truncation(self):
        value = {**self.original, 'code': 'x'*40000}
        self.assertIsNone(prepare(json.dumps(value), 'bad interface', {}, self.base))

    def test_refusal_is_explicit_and_does_not_invent_a_candidate(self):
        p = self.packet()
        content = json.dumps({'status': 'needs_researcher', 'candidate': None, 'reason': 'Ambiguous intention'})
        self.assertIsNone(accept(content, p, self.h.bus.schemas._check))
        with self.assertRaises(ValueError):
            accept(self.reply(self.original), p, self.h.bus.schemas._check)
