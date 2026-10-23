"""Bounded, read-only research queries. The service never exposes holdout labels."""
import hashlib
import json
import threading
import time
from copy import deepcopy
from contextlib import nullcontext

from agent.contracts import Schemas, canonical
from agent.errors import AgentError
from tools.data import read_table
from agent.predictions import targets
from agent.tool_arguments import normalize_tool_arguments


FILTERS_SCHEMA = {'type': 'object', 'properties': {
    'mode': {'enum': ['all', 'any']}, 'clauses': {'type': 'array', 'minItems': 1, 'maxItems': 8,
        'items': {'type': 'object', 'properties': {
            'column': {'type': 'string', 'minLength': 1},
            'op': {'enum': ['eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'in', 'is_missing', 'not_missing']},
            'value': {'anyOf': [{'type': 'string', 'maxLength': 200}, {'type': 'number'},
                {'type': 'boolean'}, {'type': 'null'}, {'type': 'array', 'maxItems': 32,
                    'items': {'anyOf': [{'type': 'string', 'maxLength': 200}, {'type': 'number'}, {'type': 'boolean'}]}}]},
            'other_column': {'type': 'string', 'minLength': 1}},
            'required': ['column', 'op'], 'additionalProperties': False}}},
    'required': ['mode', 'clauses'], 'additionalProperties': False}
DERIVED_SCHEMA = {'type': 'array', 'maxItems': 8, 'items': {'type': 'object', 'properties': {
    'op': {'enum': ['missing', 'log1p', 'signed_log1p', 'square', 'sum', 'difference', 'product', 'ratio', 'cross']},
    'name': {'type': 'string', 'minLength': 1, 'maxLength': 100},
    'columns': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {'type': 'string', 'minLength': 1}}},
    'required': ['op', 'name', 'columns'], 'additionalProperties': False}}


def filter_columns(condition=None, filters=None):
    names = [condition['column']] if condition else []
    for clause in (filters or {}).get('clauses', []):
        names.append(clause['column'])
        if clause.get('other_column'):
            names.append(clause['other_column'])
    return list(dict.fromkeys(names))


def apply_filters(frame, condition=None, filters=None):
    import pandas as pd
    mask = pd.Series(True, index=frame.index)
    if condition:
        values, value = frame[condition['column']], condition['value']
        mask &= values.isna() if value is None else values.eq(value)
    checks = []
    for clause in (filters or {}).get('clauses', []):
        values, op = frame[clause['column']], clause['op']
        other = frame[clause['other_column']] if clause.get('other_column') else clause.get('value')
        if op in {'is_missing', 'not_missing'}:
            check = values.isna() if op == 'is_missing' else values.notna()
        elif op == 'in':
            if not isinstance(other, list):
                raise AgentError('INVALID_ANALYSIS', 'in requires a literal value list')
            check = values.isin(other)
        else:
            if 'value' not in clause and 'other_column' not in clause:
                raise AgentError('INVALID_ANALYSIS', 'Comparison requires value or other_column')
            try:
                check = getattr(values, {'eq': 'eq', 'ne': 'ne', 'lt': 'lt', 'lte': 'le', 'gt': 'gt', 'gte': 'ge'}[op])(other)
            except (TypeError, ValueError) as error:
                raise AgentError('INVALID_ANALYSIS', 'Incompatible filter scales/types: '+str(error)) from error
            check &= values.notna()
            if isinstance(other, pd.Series):
                check &= other.notna()
        checks.append(check.fillna(False))
    if checks:
        combined = checks[0]
        for check in checks[1:]:
            combined = combined & check if filters['mode'] == 'all' else combined | check
        mask &= combined
    return mask


ANALYSIS_SCHEMA = {
    "type": "object", "properties": {
        "kind": {"enum": ["summary", "association", "group", "compare", "dictionary", "report", "schema", "errors"]},
        "derived": DERIVED_SCHEMA,
        "filters": FILTERS_SCHEMA,
        "experiment_id": {"type": "string", "minLength": 1},
        "report_id": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "columns": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 12,
                    "description": "Existing non-target input columns, including identifiers for diagnosis. [] chooses bounded model-feature defaults, or an empty schema/report request."},
        "group_by": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "minLength": 1}}, {"type": "null"}],
                     "description": "For kind=group, a selected input feature from columns. Otherwise null."},
        "include_target": {"type": "boolean", "description": "Add fit-only target summaries separately from columns. False for compare/dictionary/report/schema."},
        "limit_bytes": {"type": "integer", "minimum": 256, "maximum": 6000},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 12},
        "filter": {"anyOf": [{"type": "object", "properties": {
            "column": {"type": "string", "minLength": 1},
            "value": {"anyOf": [{"type": "string", "maxLength": 200}, {"type": "number"},
                                  {"type": "boolean"}, {"type": "null"}]}},
            "required": ["column", "value"], "additionalProperties": False}, {"type": "null"}]}},
    "required": ["kind", "columns", "group_by", "include_target"], "additionalProperties": False}


def compact_profile(profile, max_columns=128):
    """Protect schema/missingness together instead of dropping whole useful fields."""
    if 'core_profile' in profile:
        return deepcopy(profile['core_profile'])
    names = profile.get("train_columns", [])
    columns = [[name, profile.get("train_dtypes", {}).get(name, "unknown"),
                profile.get("train_missing_fraction", {}).get(name),
                profile.get("test_missing_fraction", {}).get(name)] for name in names[:max_columns]]
    return {"train_rows": profile.get("train_rows"), "test_rows": profile.get("test_rows"),
            "column_fields": ["name", "dtype", "train_missing", "test_missing"], "columns": columns,
            "omitted_columns": max(0, len(names)-len(columns)),
            "details": "Adaptive mode exposes analyze_data for fit-only distributions, associations and groups."}


def summarize(frame):
    import numpy as np
    import pandas as pd
    records = []
    for name in frame:
        series = frame[name]
        item = {"column": name, "dtype": str(series.dtype), "missing_fraction": float(series.isna().mean()) if len(series) else None,
                "unique": int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            item["nonfinite"] = int((series.notna() & ~np.isfinite(series)).sum())
            item["quantiles"] = {str(q): float(v) for q, v in finite.quantile([0, .01, .5, .99, 1]).items()} if len(finite) else {}
            item["zeros"] = int((finite == 0).sum())
            item["mean"] = float(finite.mean()) if len(finite) else None
            item["std"] = float(finite.std(ddof=0)) if len(finite) else None
        else:
            item["top_values"] = [[str(k)[:100], int(v)] for k, v in series.value_counts().head(6).items()]
        records.append(item)
    return records


class AnalysisService:
    def __init__(self, spec, validation, artifacts, budget, log, schemas=None):
        self.spec, self.artifacts, self.budget, self.log = spec, artifacts, budget, log
        self.schemas = schemas or Schemas()
        self.validation_id = validation["validation_id"]
        # Keep only headers and the trusted fit partition. Each query reads only
        # its columns, discards holdout rows, filters, and then samples.
        self.fit_indices = set(artifacts.read_json(validation["train_rows_artifact_id"]))
        self.fit_rows = len(self.fit_indices)
        self.train_path = artifacts.resolve(spec['train_artifact_id'])
        self.test_path = artifacts.resolve(spec['test_artifact_id'])
        self.train_columns = list(read_table(self.train_path, nrows=0).columns)
        self.test_columns = list(read_table(self.test_path, nrows=0).columns)
        from agent.data_transport import transport
        tables = transport(artifacts)
        self.input_tables = {'fit': tables.existing(spec['train_artifact_id'], spec),
                             'test': tables.existing(spec['test_artifact_id'])}
        self.records, self.cache, self.lock = [], {}, threading.RLock()

    def recent(self, visible_orders=None):
        with self.lock:
            return deepcopy([r for r in self.records if visible_orders is None or r.get('work_order_id') in visible_orders][-6:])

    def _check_time(self, started):
        if not self.budget.snapshot()['search_allowed']:
            raise AgentError('SEARCH_CLOSED', 'Analysis cannot consume the final publication reserve')
        if self.budget.phase_remaining() < 1:
            raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Analysis reached the current phase deadline')
        if time.monotonic() - started >= 120:
            raise AgentError('ANALYSIS_TIMEOUT', 'Analysis exceeded its 120 second read allowance')

    def _sample(self, role, columns, condition, started, derived=None, filters=None):
        """Bound memory to 20k selected rows plus one 4k-row CSV chunk.

        Count the complete matching population. Uniform deterministic sampling
        happens after the filter, so a rare group is not lost in a global sample.
        The reader/check is cooperative: a single CSV chunk read is not preempted.
        """
        import numpy as np
        import pandas as pd
        derived = derived or []
        generated = {step['name'] for step in derived}
        needed = columns + filter_columns(condition, filters) + [c for step in derived for c in step['columns']]
        selected = list(dict.fromkeys(c for c in needed if c not in generated))
        path = self.train_path if role == 'fit' else self.test_path
        rng = np.random.default_rng(42)
        samples, priorities = [], np.empty(0)
        offset = population = total_rows = 0
        dtype = ({self.spec['target_column']: str} if role == 'fit' and self.spec.get('task_kind') not in {None, 'regression'} and self.spec['target_column'] in selected else None)
        from agent.data_transport import cached_chunks
        reference = self.input_tables[role]
        source = (cached_chunks(reference, selected) if reference else
                  read_table(path, usecols=selected, chunksize=4096, dtype=dtype))
        with source as reader:
            for chunk in reader:
                self._check_time(started)
                total_rows += len(chunk)
                if role == 'fit':
                    chunk = chunk.loc[[i in self.fit_indices for i in range(offset, offset+len(chunk))]]
                offset = total_rows
                if derived:
                    from agent.feature_engineering import FeatureProgram
                    for step in derived:
                        chunk[step['name']] = FeatureProgram._apply(chunk, step, {})
                chunk = chunk.loc[apply_filters(chunk, condition, filters)]
                population += len(chunk)
                if not len(chunk):
                    continue
                samples.append(chunk[columns])
                priorities = np.concatenate((priorities, rng.random(len(chunk))))
                combined = pd.concat(samples, ignore_index=True)
                if len(combined) > 20000:
                    keep = np.argpartition(priorities, 19999)[:20000]
                    combined = combined.iloc[keep].reset_index(drop=True)
                    priorities = priorities[keep]
                if int(combined.memory_usage(deep=True).sum()) > 64*1024*1024:
                    raise AgentError('ANALYSIS_MEMORY_LIMIT', 'Bounded sample exceeds 64 MiB; select fewer/narrower columns')
                samples = [combined]
        self._check_time(started)
        data = samples[0] if samples else pd.DataFrame(columns=columns)
        return data, population, total_rows

    def feature_frame(self):
        """Dispatcher seed input, never targets or excluded identifiers; no SQL."""
        import pandas as pd
        columns = [c for c in self.train_columns if c not in {*targets(self.spec), *self.spec['excluded_features']}]
        if not columns:
            return pd.DataFrame()
        with self.lock, self.budget.analysis_slot() if hasattr(self.budget, 'analysis_slot') else nullcontext():
            data, _, _ = self._sample('fit', columns, None, time.monotonic())
        return data

    def analyze(self, query, context=None):
        # Serialize bounded analyses and cache writes; no SQL connection crosses
        # into these research threads. Return copies so peers cannot edit evidence.
        with self.lock:
            self.query_context = context
            with self.budget.analysis_slot() if hasattr(self.budget, 'analysis_slot') else nullcontext():
                try:
                    return deepcopy(self._analyze(query))
                except (ValueError, TypeError, KeyError) as error:
                    raise AgentError('ANALYSIS_COMPUTATION_FAILED', str(error)[:600]) from error

    def _analyze(self, query):
        import numpy as np
        import pandas as pd
        started = time.monotonic()
        self._check_time(started)
        # Reuse the contract validator for direct as well as model callers.
        try:
            query = normalize_tool_arguments('analyze_data', query)
            self.schemas.validate(query, 'AnalysisQuery')
            query_body = canonical(query)
        except AgentError as exc:
            raise AgentError('INVALID_ANALYSIS', 'Invalid bounded analysis query: '+str(exc)) from exc
        query = deepcopy(query)
        if len(query_body.encode()) > 8192:
            raise AgentError('INVALID_ANALYSIS', 'Query exceeds 8 KiB')
        if query['kind'] not in {'schema', 'report'} and ('offset' in query or 'limit' in query or 'limit_bytes' in query):
            raise AgentError('INVALID_ANALYSIS', 'offset is for schema/report; limit is schema page count and limit_bytes is report page bytes')
        condition, filters, derived = query.get('filter'), query.get('filters'), query.get('derived', [])
        from agent.feature_schema import validate_program
        try:
            if derived:
                validate_program(derived, self.train_columns, targets(self.spec))
        except ValueError as error:
            raise AgentError('INVALID_ANALYSIS', str(error)) from error
        available = set(self.train_columns) | {step['name'] for step in derived}
        if any(c not in available or c in targets(self.spec) for c in filter_columns(condition, filters)):
            raise AgentError('INVALID_ANALYSIS', 'Filters need input/derived columns, never targets')
        if query['kind'] in {'report', 'schema', 'dictionary'} and (derived or filters):
            raise AgentError('INVALID_ANALYSIS', 'Metadata/report reads do not compute expressions or row filters')
        if query['kind'] != 'errors' and query.get('experiment_id'):
            raise AgentError('INVALID_ANALYSIS', 'experiment_id is only valid for kind=errors')
        if query['kind'] == 'report':
            if query['columns'] or query['include_target'] or query['group_by'] is not None or condition:
                raise AgentError('INVALID_ANALYSIS', 'Retrieve a saved report by report_id without new columns/target requests')
            with self.lock:
                known = next((r for r in self.cache.values() if r['artifact_id'] == query.get('report_id')), None)
            if known is None:
                raise AgentError('INVALID_ANALYSIS', 'report_id must refer to an analysis from this run and validation')
            self._check_report_visibility(known['artifact_id'])
            if 'offset' in query or 'limit_bytes' in query:
                full = self.artifacts.read_json(known['artifact_id'])
                raw = canonical(full).encode('utf-8')
                offset = query.get('offset', 0)
                if offset > len(raw) or (offset < len(raw) and raw[offset] & 0xC0 == 0x80):
                    raise AgentError('INVALID_ANALYSIS', 'Use next_offset from the report page')
                text = raw[offset:offset+query.get('limit_bytes', 4000)].decode('utf-8', errors='ignore')
                end = offset+len(text.encode('utf-8'))
                return {'artifact_id': known['artifact_id'], 'report': {'scope': full['scope'], 'text': text,
                    'offset': offset, 'total_bytes': len(raw), 'next_offset': end if end < len(raw) else None}}
            return deepcopy(known)
        if query.get('report_id') is not None:
            raise AgentError('INVALID_ANALYSIS', 'report_id is only valid for kind=report')
        owner = (getattr(self, 'query_context', None) or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')
        digest = hashlib.sha256(canonical([query, owner]).encode()).hexdigest()
        with self.lock:
            if digest in self.cache:
                return deepcopy(self.cache[digest])
        target = self.spec['target_column']
        target_columns = targets(self.spec)
        excluded = set(self.spec['excluded_features']) | set(target_columns)
        if query['kind'] == 'schema':
            if query['columns'] or query['include_target'] or query['group_by'] is not None or condition:
                raise AgentError('INVALID_ANALYSIS', 'Schema pages use no columns, filter or target values')
            names = list(dict.fromkeys(self.train_columns+self.test_columns))
            offset, limit = query.get('offset', 0), query.get('limit', 12)
            result = {'scope': 'schema_only', 'validation_id': self.validation_id, 'query': query,
                'total_columns': len(names), 'offset': offset,
                'next_offset': offset+limit if offset+limit < len(names) else None,
                'columns': [{'column': name, 'in_train': name in self.train_columns, 'in_test': name in self.test_columns,
                    'role': 'target' if name in target_columns else 'excluded' if name in excluded else 'feature'}
                    for name in names[offset:offset+limit]]}
            return self._save(digest, query, result, [])
        requested = list(dict.fromkeys(query['columns']))
        if not requested:
            requested = [c for c in self.train_columns if c not in excluded][:8]
        if any(c not in available or c in target_columns for c in requested):
            raise AgentError("INVALID_ANALYSIS", "columns must contain existing non-target input columns. Remove the target; use include_target=true for fit-only target summaries.",
                             details={'invalid_columns': [c for c in requested if c not in self.train_columns or c == target],
                                      'target_column': target})
        selected_targets = target_columns[:4] if query['include_target'] else []
        columns = requested + selected_targets
        if query['kind'] == 'errors':
            if query['group_by'] is not None:
                raise AgentError('INVALID_ANALYSIS', 'For compound error groups use a derived cross column; group_by is for fit group summaries')
            return self._errors(query, requested, started)
        # Deterministic, bounded computation; row-level labels never leave this service.
        if query['kind'] in {'compare', 'dictionary'} and query['include_target']:
            raise AgentError('INVALID_ANALYSIS', 'Feature comparisons and dictionary queries do not expose test targets')
        if query['kind'] != 'group' and query['group_by'] is not None:
            raise AgentError('INVALID_ANALYSIS', 'group_by is only valid for group queries')
        if query['kind'] == 'compare':
            missing = [c for c in requested + filter_columns(condition, filters) + [c for step in derived for c in step['columns']] if c not in self.test_columns and c not in {step['name'] for step in derived}]
            if missing:
                raise AgentError('INVALID_ANALYSIS', 'Selected features are absent from the unlabelled test table',
                                 details={'missing_test_columns': missing})
        if query['kind'] == 'dictionary' and condition:
            raise AgentError('INVALID_ANALYSIS', 'Dictionary metadata cannot be filtered by row values')
        data, population, _ = self._sample('fit', columns, condition, started, derived, filters) if query['kind'] != 'dictionary' else (pd.DataFrame(), 0, 0)
        result = {"scope": "fit_only", "validation_id": self.validation_id,
                  "fit_rows": self.fit_rows, "matched_fit_rows": population,
                  "analyzed_rows": len(data), "sampled": population > len(data), "query": query,
                  "sample_rule": "deterministic_uniform_priority_after_filter", "sample_seed": 42,
                  "partition_rows": self.fit_rows,
                  "omitted_target_columns": max(0, len(target_columns)-len(selected_targets)) if query["include_target"] else 0}
        # Excluding an ID from model inputs is not a ban on diagnosing its groups.
        # Analysis access never changes what the worker may train on.
        diagnostic = requested + filter_columns(condition, filters)
        result['diagnostic_only_columns'] = sorted(set(diagnostic).intersection(self.spec['excluded_features']))
        if query['kind'] == 'dictionary':
            result.update(scope='dictionary_only', dictionary=[])
            if self.spec.get('dictionary_artifact_id'):
                with read_table(self.artifacts.resolve(self.spec['dictionary_artifact_id']), chunksize=4096) as reader:
                    for chunk in reader:
                        self._check_time(started)
                        if 'feature' in chunk:
                            chunk = chunk[chunk['feature'].isin(requested)]
                        result['dictionary'].extend(json.loads(chunk.head(12-len(result['dictionary'])).to_json(orient='records', force_ascii=False)))
                        if len(result['dictionary']) >= 12:
                            break
        elif query['kind'] == 'compare':
            test, test_population, test_rows = self._sample('test', requested, condition, started, derived, filters)
            result.update(scope='fit_and_unlabelled_test_features', fit=summarize(data),
                          test=summarize(test), analyzed_test_rows=len(test), test_rows=test_rows,
                          matched_test_rows=test_population, test_sampled=len(test) < test_population)
            coverage = []
            for name in requested:
                if not pd.api.types.is_numeric_dtype(data[name]):
                    known = set(data[name].dropna().astype(str))
                    unseen = test[name].notna() & ~test[name].astype(str).isin(known)
                    coverage.append({'column': name, 'unseen_test_rows_in_sample': int(unseen.sum()),
                        'test_sample_rows': len(test), 'reference': 'bounded_fit_sample',
                        'exact_population_claim': False})
            result['category_coverage'] = coverage
        elif query['kind'] == 'summary':
            result['columns'] = summarize(data)
        elif query['kind'] == 'association':
            numeric = data.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan)
            corr = numeric.corr(method='spearman').replace([np.inf, -np.inf], np.nan)
            result['spearman'] = json.loads(corr.to_json())
            if query['include_target'] and not pd.api.types.is_numeric_dtype(data[target]):
                result['target_counts'] = summarize(data[[c for c in selected_targets if not pd.api.types.is_numeric_dtype(data[c])]])
            result['note'] = 'Association is descriptive, not causal. Non-numeric columns are omitted from correlation.'
        else:
            group = query['group_by']
            group_names = group if isinstance(group, list) else [group]
            if any(c not in requested for c in group_names):
                raise AgentError("INVALID_ANALYSIS", "group_by must be one of the selected input columns")
            values = (data[group_names[0]] if len(group_names) == 1 else
                      data[group_names].apply(lambda row: canonical([None if pd.isna(v) else str(v) for v in row]), axis=1))
            if pd.api.types.is_numeric_dtype(values) and values.nunique() > 12:
                values = pd.qcut(values.replace([np.inf, -np.inf], np.nan), 5, duplicates='drop')
            values = values.astype(object).where(values.notna(), '__MISSING__').astype(str)
            counts = values.value_counts().head(12)
            result['groups'] = []
            result['support_scope'] = 'analyzed_sample'
            result['omitted_groups'] = max(0, int(values.nunique())-len(counts))
            for label, count in counts.items():
                record = {'group': str(label)[:100], 'rows': int(count), 'support_fraction_in_sample': float(count/len(data))}
                if query['include_target']:
                    summaries = summarize(data.loc[values == label, selected_targets])
                    record['target'] = summaries[0] if len(summaries) == 1 else summaries
                result['groups'].append(record)
        self._check_time(started)
        result['duration_seconds'] = time.monotonic()-started
        return self._save(digest, query, result, requested)

    def _save(self, digest, query, result, requested):
        body = canonical(result)
        if len(body.encode()) > 16000:
            raise AgentError('ANALYSIS_TOO_LARGE', 'Select fewer columns for this query')
        # Keep the full report on disk. Tool replies have a separate API budget.
        reply = json.loads(body)
        omitted = 0
        while len(canonical(reply).encode()) > 6000:
            candidates = [(len(v), k) for k, v in reply.items() if isinstance(v, list) and k in {'columns', 'groups', 'dictionary', 'fit', 'test'} and v]
            if not candidates:
                break
            reply[max(candidates)[1]].pop()
            omitted += 1
        if omitted:
            reply['omitted_records'] = omitted
            reply['detail_hint'] = 'Full report is saved: kind=report, report_id, offset=0, limit_bytes=4000; follow next_offset.'
            if query['kind'] == 'schema':
                if not reply['columns']:
                    raise AgentError('ANALYSIS_TOO_LARGE', 'A schema column does not fit the tool reply')
                reply['next_offset'] = query.get('offset', 0)+len(reply['columns'])
        if len(canonical(reply).encode()) > 6000:
            raise AgentError('ANALYSIS_TOO_LARGE', 'Select fewer columns for a detailed tool reply')
        art = self.artifacts.write_json('analysis/'+digest+'.json', result, 'research_analysis')
        response = {'artifact_id': art['artifact_id'], 'report': reply}
        with self.lock:
            self.cache[digest] = response
            self.records.append({'artifact_id': art['artifact_id'], 'query': deepcopy(query), 'scope': result['scope'],
                'work_order_id': (getattr(self, 'query_context', None) or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')})
        self.log.event('research.analysis_completed', artifact_id=art['artifact_id'], kind=query['kind'],
                       columns=requested, rows=result.get('analyzed_rows', 0), scope=result['scope'])
        return deepcopy(response)

    def _check_report_visibility(self, artifact_id):
        context = getattr(self, 'query_context', None)
        if not context or not context['metadata'].get('graph_binding'):
            return
        binding = context['metadata']['graph_binding']
        record = next((r for r in self.records if r['artifact_id'] == artifact_id), {})
        scope = context['metadata'].get('detail_scope', {})
        if record.get('work_order_id') == binding['work_order_id']:
            return
        ref = context['metadata'].get('detail_artifact_id')
        visible = self.artifacts.read_json(ref)['sections']['analyses'] if ref else []
        if not any(r['artifact_id'] == artifact_id for r in visible):
            raise AgentError('MEMORY_SCOPE_DENIED', 'Analysis report is outside the frozen fork/work-order scope')

    def _errors(self, query, requested, started):
        context = getattr(self, 'query_context', None)
        if not context or not getattr(self, 'error_provider', None) or not query.get('experiment_id'):
            raise AgentError('INVALID_ANALYSIS', 'Errors require an authorized context and a measured experiment_id')
        metadata = context['metadata']
        ref = metadata.get('detail_artifact_id')
        document = self.artifacts.read_json(ref) if ref else {}
        frozen = document.get('sections', {})
        # A pending trial ID in an old series is NOT permission to inspect its
        # later measured result across an ancestor fork cutoff.
        visible = [eid for eid in frozen.get('experiments', [])
            if frozen.get('experiment/'+eid, {}).get('state') == 'evaluated'
            and 'diagnostics/'+eid in document.get('documents', {})]
        user = json.loads(context['messages'][1]['content'])

        visible += list(metadata.get('delivered_experiment_ids', []))
        if query['experiment_id'] not in visible:
            raise AgentError('MEMORY_SCOPE_DENIED', 'Error diagnostics require a visible measured experiment')
        if query['include_target']:
            raise AgentError('INVALID_ANALYSIS', 'Error diagnostics return aggregates, never target summaries or labels')
        result = self.error_provider(query, lambda: self._check_time(started))
        result['duration_seconds'] = time.monotonic()-started
        self._check_time(started)
        digest = hashlib.sha256(canonical([query, metadata.get('graph_binding')]).encode()).hexdigest()
        return self._save(digest, query, result, requested)
