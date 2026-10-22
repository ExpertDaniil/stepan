import json
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.artifacts import Artifacts
from agent.config import ROOT
from agent.errors import AgentError
from agent.preprocessing import preprocessing_policy
from tools.knowledge import KnowledgeService


class Log:
    def event(self, *args, **kwargs):
        pass


class Catalog:
    def pipeline_facts(self, base=None):
        return {'recipe_id': 'catboost-regressor-v1', 'status': 'owned_policy',
                'preprocessing': preprocessing_policy('catboost-regressor-v1')}


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = Artifacts(self.root/'run', 'knowledge-test', Log())
        self.spec = {'task_kind': 'regression', 'target_column': 'target', 'excluded_features': ['id'],
                     'metric': {'metric_id': 'rmse'}}
        self.profile = {'train_dtypes': {'category': 'object', 'value': 'float64'},
                        'train_missing_fraction': {'value': .4}}
        self.service = KnowledgeService(self.spec, self.profile, Catalog(), self.artifacts, Log())

    def test_complete_example_native_nan_warning_and_cache_isolation(self):
        query = {'query': '', 'ids': ['method-missing-flags']}
        facts = Catalog().pipeline_facts()
        first = self.service.search(query, facts)
        row = first['results'][0]
        self.assertIn('def make_pipeline(task_spec, params):', row['code_example']['source'])
        self.assertEqual(row['code_example']['verification'], 'not_executed')
        self.assertTrue(any('does not median-impute' in s for s in row['current_task_cautions']))
        first['results'][0]['code_example']['source'] = 'mutated'
        second = self.service.search(query, facts)
        self.assertNotEqual(second['results'][0]['code_example']['source'], 'mutated')
        saved = self.artifacts.read_json(second['artifact_id'])
        self.assertEqual(saved['reply']['scope'], 'offline_external_experience')

    def test_seed_has_both_method_and_source_content(self):
        reply = self.service.seed()
        self.assertEqual({r['kind'] for r in reply['results']}, {'method', 'writeup'})
        source = next(r for r in reply['results'] if r['kind'] == 'writeup')
        self.assertTrue(source['excerpt'])
        self.assertIn('competition_url', source)
        self.assertNotIn('text', source)

    def test_wrong_task_source_and_forged_ids_are_rejected(self):
        original = [json.loads(line) for line in (ROOT/'recipes/knowledge/sources.jsonl').read_text().splitlines()]
        otto = next(r['id'] for r in original if r['source_record'] == 396)
        for ident in (otto, 'kg-invented'):
            with self.assertRaises(AgentError):
                self.service.search({'query': '', 'ids': [ident]})

    def test_source_read_offsets_do_not_change_the_original(self):
        doc = next(d for d in self.service.documents.values() if d['kind'] == 'writeup' and len(d['text']) > 6000)
        response = self.service.search({'query': '', 'ids': [doc['id']], 'offset': 5000})
        row = response['results'][0]
        self.assertEqual(row['excerpt'], doc['text'][5000:10000])
        self.assertEqual(row['excerpt_offset'], 5000)
        self.assertEqual(row['source_sha256'], doc['source_sha256'])

    def test_tampered_corpus_fails_closed_as_a_knowledge_source(self):
        copy = self.root/'corpus'
        shutil.copytree(ROOT/'recipes/knowledge', copy)
        (copy/'cards.json').write_text('[]', encoding='utf-8')
        with self.assertRaises(AgentError) as caught:
            KnowledgeService(self.spec, self.profile, Catalog(), self.artifacts, Log(), root=copy)
        self.assertEqual(caught.exception.code, 'KNOWLEDGE_UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
