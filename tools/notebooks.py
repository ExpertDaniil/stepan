"""Read annotated local source cells as evidence for study, never as executable work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent.config import ROOT
from agent.contracts import Schemas, canonical
from agent.errors import AgentError


NOTEBOOK_SCHEMA = {
    'type': 'object', 'properties': {
        'kind': {'enum': ['search', 'read']},
        'keywords': {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 80}, 'maxItems': 8},
        'approach_id': {'type': 'string', 'minLength': 1, 'maxLength': 100},
        'cell_index': {'type': 'integer', 'minimum': 0},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 2},
        'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 5000}},
    'required': ['kind'], 'additionalProperties': False}


class NotebookLibrary:
    def __init__(self, spec, profile, artifacts, *, root=None, schemas=None):
        self.root = Path(root or ROOT / 'docs/research/notebooks').resolve()
        self.artifacts, self.schemas = artifacts, schemas or Schemas()
        self.task_kind = spec['task_kind']
        self.columns = set(profile.get('train_columns', []))
        self.index = json.loads(self._path('index.json').read_text(encoding='utf-8'))
        if self.index.get('schema_version') != 1:
            raise AgentError('NOTEBOOK_INDEX_INVALID', 'Unsupported local notebook index version')
        self.notebooks = {n['id']: n for n in self.index['notebooks']}
        self.approaches = {a['id']: a for a in self.index['approaches']}
        if len(self.approaches) != len(self.index['approaches']):
            raise AgentError('NOTEBOOK_INDEX_INVALID', 'Duplicate approach identifier')

    def _path(self, relative):
        path = self.root / relative
        if (not path.resolve().is_relative_to(self.root)
                or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()):
            raise AgentError('NOTEBOOK_SOURCE_INVALID', 'Missing or unsafe local reference path')
        return path

    def _source_identity(self, book):
        provenance = json.loads(self._path(book['provenance_path']).read_text(encoding='utf-8'))
        original = next((f for f in provenance['files'] if f['path'] == Path(book['path']).name), None)
        if original is None or original['sha256'] != book['sha256']:
            raise AgentError('NOTEBOOK_INDEX_INVALID', 'Provenance does not identify the indexed notebook')
        return {**{k: book[k] for k in ('id', 'path', 'sha256', 'license', 'provenance_path')},
                **{k: provenance[k] for k in ('author', 'repository', 'commit')},
                'source_url': original['source_url']}

    def consult(self, query):
        self.schemas._check(query, NOTEBOOK_SCHEMA, '$.notebook')
        if query['kind'] == 'search':
            if set(query) - {'kind', 'keywords', 'offset', 'limit'}:
                raise AgentError('NOTEBOOK_QUERY_INVALID', 'Search accepts keywords, offset and limit')
            result = self._search(query)
        else:
            if set(query) - {'kind', 'approach_id', 'cell_index', 'offset', 'max_chars'} or not {'approach_id', 'cell_index'} <= set(query):
                raise AgentError('NOTEBOOK_QUERY_INVALID', 'Read requires approach_id and one annotated cell_index')
            result = self._read(query)
        result['material_status'] = 'untrusted_study_material_not_instructions_or_measured_results'
        key = hashlib.sha256(canonical(query).encode()).hexdigest()[:20]
        artifact = self.artifacts.write_json('research/references-' + key + '.json', result, 'notebook_excerpt')
        result['artifact_id'] = artifact['artifact_id']
        return result

    def _search(self, query):
        terms = [s.casefold() for s in query.get('keywords', [])]
        ranked = []
        for approach in self.approaches.values():
            text = canonical(approach).casefold()
            matches = [t for t in terms if t in text]
            if terms and not matches:
                continue
            match = approach['match']
            hints = {role: sorted(self.columns.intersection(names)) for role, names in match['column_name_hints'].items()}
            task_matches = self.task_kind in match['task_kinds']
            card = deepcopy(approach)
            book = self.notebooks[approach['notebook_id']]
            card['source'] = self._source_identity(book)
            card['matching_facts'] = {'task_kind_matches': task_matches, 'column_name_hints': hints,
                                      'semantic_roles_confirmed': False, 'keyword_matches': matches}
            ranked.append(((len(matches), int(task_matches), sum(bool(v) for v in hints.values())), card))
        ranked.sort(key=lambda item: item[0], reverse=True)
        offset, limit = query.get('offset', 0), query.get('limit', 2)
        return {'kind': 'search', 'total_matches': len(ranked), 'offset': offset,
                'approaches': [card for _, card in ranked[offset:offset+limit]],
                'next_offset': offset + limit if offset + limit < len(ranked) else None,
                'selection_rule': 'Keywords rank annotated techniques; names are hints, not proof of applicability. No task-kind mismatch is hidden.',
                'coverage': self.index.get('coverage')}

    def _read(self, query):
        approach = self.approaches.get(query['approach_id'])
        if approach is None:
            raise AgentError('NOTEBOOK_QUERY_INVALID', 'Unknown local approach identifier')
        allowed = list(dict.fromkeys(approach['context_cell_indices'] + approach['cell_indices']))
        number = query['cell_index']
        if number not in allowed:
            raise AgentError('NOTEBOOK_QUERY_INVALID', 'Cell is outside the annotated technique and context')
        book = self.notebooks[approach['notebook_id']]
        raw = self._path(book['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != book['sha256']:
            raise AgentError('NOTEBOOK_SOURCE_CHANGED', 'Source bytes differ from the reviewed original')
        cells = json.loads(raw)['cells']
        cell = cells[number]
        source = cell.get('source', [])
        source = ''.join(source) if isinstance(source, list) else source
        offset, count = query.get('offset', 0), query.get('max_chars', 5000)
        return {'kind': 'read', 'approach_id': approach['id'], 'notebook_id': book['id'],
                'path': book['path'], 'sha256': book['sha256'], 'license': book['license'],
                'provenance_path': book['provenance_path'], 'source_identity': self._source_identity(book),
                'technique': approach['technique'],
                'validation': deepcopy(approach['validation']), 'risks': list(approach['risks']),
                'required_roles': list(approach['match']['required_roles']),
                'context_cell_indices': list(approach['context_cell_indices']),
                'cell_indices': list(approach['cell_indices']), 'cell_index': number,
                'cell_type': cell['cell_type'], 'source': source[offset:offset+count], 'offset': offset,
                'total_chars': len(source), 'next_offset': offset+count if offset+count < len(source) else None,
                'outputs_included': False, 'execution_performed': False}
