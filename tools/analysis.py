"""Bounded, read-only research queries. The service never exposes holdout labels."""
import hashlib
import json
import threading
import time
from copy import deepcopy

from agent.contracts import Schemas, canonical
from agent.errors import AgentError
from tools.data import read_table


ANALYSIS_SCHEMA = {
    "type": "object", "properties": {
        "kind": {"enum": ["summary", "association", "group", "compare", "dictionary", "report", "schema"]},
        "report_id": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "columns": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 12},
        "group_by": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "include_target": {"type": "boolean"},
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
        self.records, self.cache, self.lock = [], {}, threading.RLock()

    def recent(self):
        with self.lock:
            return deepcopy(self.records[-6:])

    def _check_time(self, started):
        if not self.budget.snapshot()['search_allowed']:
            raise AgentError('SEARCH_CLOSED', 'Analysis cannot consume the final publication reserve')
        if self.budget.phase_remaining() < 1:
            raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Analysis reached the current phase deadline')
        if time.monotonic() - started >= 120:
            raise AgentError('ANALYSIS_TIMEOUT', 'Analysis exceeded its 120 second read allowance')

    def _sample(self, role, columns, condition, started):
        """Bound memory to 20k selected rows plus one 4k-row CSV chunk.

        Count the complete matching population. Uniform deterministic sampling
        happens after the filter, so a rare group is not lost in a global sample.
        The reader/check is cooperative: a single CSV chunk read is not preempted.
        """
        import numpy as np
        import pandas as pd
        selected = list(dict.fromkeys(columns + ([condition['column']] if condition else [])))
        path = self.train_path if role == 'fit' else self.test_path
        rng = np.random.default_rng(42)
        samples, priorities = [], np.empty(0)
        offset = population = total_rows = 0
        with read_table(path, usecols=selected, chunksize=4096) as reader:
            for chunk in reader:
                self._check_time(started)
                total_rows += len(chunk)
                if role == 'fit':
                    chunk = chunk.loc[[i in self.fit_indices for i in range(offset, offset+len(chunk))]]
                offset = total_rows
                if condition:
                    values, value = chunk[condition['column']], condition['value']
                    chunk = chunk.loc[values.isna() if value is None else values.eq(value)]
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
                samples = [combined]
        self._check_time(started)
        data = samples[0] if samples else pd.DataFrame(columns=columns)
        return data, population, total_rows

    def analyze(self, query):
        # Serialize bounded analyses and cache writes; no SQL connection crosses
        # into these research threads. Return copies so peers cannot edit evidence.
        with self.lock:
            return deepcopy(self._analyze(query))

    def _analyze(self, query):
        import numpy as np
        import pandas as pd
        started = time.monotonic()
        self._check_time(started)
        # Reuse the contract validator for direct as well as model callers.
        try:
            self.schemas.validate(query, 'AnalysisQuery')
            query_body = canonical(query)
        except AgentError as exc:
            raise AgentError('INVALID_ANALYSIS', 'Invalid bounded analysis query: '+str(exc)) from exc
        query = deepcopy(query)
        if len(query_body.encode()) > 8192:
            raise AgentError('INVALID_ANALYSIS', 'Query exceeds 8 KiB')
        if query['kind'] != 'schema' and ('offset' in query or 'limit' in query):
            raise AgentError('INVALID_ANALYSIS', 'offset and limit are only valid for schema')
        condition = query.get('filter')
        if condition is not None:
            if (condition['column'] not in self.train_columns
                    or condition['column'] in set(self.spec['excluded_features']) | {self.spec['target_column']}):
                raise AgentError('INVALID_ANALYSIS', 'Filter an input column, never the target')
        if query['kind'] == 'report':
            if query['columns'] or query['include_target'] or query['group_by'] is not None or condition:
                raise AgentError('INVALID_ANALYSIS', 'Retrieve a saved report by report_id without new columns/target requests')
            with self.lock:
                known = next((r for r in self.cache.values() if r['artifact_id'] == query.get('report_id')), None)
            if known is None:
                raise AgentError('INVALID_ANALYSIS', 'report_id must refer to an analysis from this run and validation')
            return deepcopy(known)
        if query.get('report_id') is not None:
            raise AgentError('INVALID_ANALYSIS', 'report_id is only valid for kind=report')
        digest = hashlib.sha256(canonical(query).encode()).hexdigest()
        with self.lock:
            if digest in self.cache:
                return self.cache[digest]
        target = self.spec['target_column']
        excluded = set(self.spec['excluded_features']) | {target}
        if query['kind'] == 'schema':
            if query['columns'] or query['include_target'] or query['group_by'] is not None or condition:
                raise AgentError('INVALID_ANALYSIS', 'Schema pages use no columns, filter or target values')
            names = list(dict.fromkeys(self.train_columns+self.test_columns))
            offset, limit = query.get('offset', 0), query.get('limit', 12)
            result = {'scope': 'schema_only', 'validation_id': self.validation_id, 'query': query,
                'total_columns': len(names), 'offset': offset,
                'next_offset': offset+limit if offset+limit < len(names) else None,
                'columns': [{'column': name, 'in_train': name in self.train_columns, 'in_test': name in self.test_columns,
                    'role': 'target' if name == target else 'excluded' if name in excluded else 'feature'}
                    for name in names[offset:offset+limit]]}
            return self._save(digest, query, result, [])
        requested = list(dict.fromkeys(query['columns']))
        if not requested:
            requested = [c for c in self.train_columns if c not in excluded][:8]
        if any(c not in self.train_columns or c in excluded for c in requested):
            raise AgentError("INVALID_ANALYSIS", "Select existing input features; request target through include_target")
        columns = requested + ([target] if query['include_target'] else [])
        # Deterministic, bounded computation; row-level labels never leave this service.
        if query['kind'] in {'compare', 'dictionary'} and query['include_target']:
            raise AgentError('INVALID_ANALYSIS', 'Feature comparisons and dictionary queries do not expose test targets')
        if query['kind'] != 'group' and query['group_by'] is not None:
            raise AgentError('INVALID_ANALYSIS', 'group_by is only valid for group queries')
        if query['kind'] == 'compare' and any(c not in self.test_columns for c in requested + ([condition['column']] if condition else [])):
            raise AgentError('INVALID_ANALYSIS', 'Comparison columns must also exist in test')
        if query['kind'] == 'dictionary' and condition:
            raise AgentError('INVALID_ANALYSIS', 'Dictionary metadata cannot be filtered by row values')
        data, population, _ = self._sample('fit', columns, condition, started) if query['kind'] != 'dictionary' else (pd.DataFrame(), 0, 0)
        result = {"scope": "fit_only", "validation_id": self.validation_id,
                  "fit_rows": self.fit_rows, "matched_fit_rows": population,
                  "analyzed_rows": len(data), "sampled": population > len(data), "query": query}
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
            test, test_population, test_rows = self._sample('test', requested, condition, started)
            result.update(scope='fit_and_unlabelled_test_features', fit=summarize(data),
                          test=summarize(test), analyzed_test_rows=len(test), test_rows=test_rows,
                          matched_test_rows=test_population, test_sampled=len(test) < test_population)
        elif query['kind'] == 'summary':
            result['columns'] = summarize(data)
        elif query['kind'] == 'association':
            numeric = data.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan)
            corr = numeric.corr(method='spearman').replace([np.inf, -np.inf], np.nan)
            result['spearman'] = json.loads(corr.to_json())
            if query['include_target'] and not pd.api.types.is_numeric_dtype(data[target]):
                result['target_counts'] = summarize(data[[target]])
            result['note'] = 'Association is descriptive, not causal. Non-numeric columns are omitted from correlation.'
        else:
            group = query['group_by']
            if group not in requested:
                raise AgentError("INVALID_ANALYSIS", "group_by must be one of the selected input columns")
            values = data[group]
            if pd.api.types.is_numeric_dtype(values) and values.nunique() > 12:
                values = pd.qcut(values.replace([np.inf, -np.inf], np.nan), 5, duplicates='drop').astype(object)
            values = values.fillna('__MISSING__').astype(str)
            counts = values.value_counts().head(12)
            result['groups'] = []
            for label, count in counts.items():
                record = {'group': str(label)[:100], 'rows': int(count)}
                if query['include_target']:
                    record['target'] = summarize(data.loc[values == label, [target]])[0]
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
            reply['detail_hint'] = 'Full report is saved; select fewer columns for a detailed tool reply.'
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
            self.records.append({'artifact_id': art['artifact_id'], 'query': query, 'scope': result['scope']})
        self.log.event('research.analysis_completed', artifact_id=art['artifact_id'], kind=query['kind'],
                       columns=requested, rows=result.get('analyzed_rows', 0), scope=result['scope'])
        return response
