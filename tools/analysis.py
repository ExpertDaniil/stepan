"""Bounded, read-only research queries. The service never receives holdout labels."""
import hashlib
import json
import threading
from copy import deepcopy

from agent.contracts import canonical
from agent.errors import AgentError
from tools.data import read_table
from agent.predictions import read_training, targets


ANALYSIS_SCHEMA = {
    "type": "object", "properties": {
        "kind": {"enum": ["summary", "association", "group", "compare", "dictionary", "report"]},
        "report_id": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "columns": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 12,
                    "description": "Input features only; never include the target. [] chooses bounded defaults."},
        "group_by": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                     "description": "For kind=group, a selected input feature from columns. Otherwise null."},
        "include_target": {"type": "boolean", "description": "Add fit-only target summaries separately from columns. False for compare/dictionary/report."}},
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
        item = {"column": name, "dtype": str(series.dtype), "missing_fraction": float(series.isna().mean()),
                "unique": int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            item["nonfinite"] = int((series.notna() & ~np.isfinite(series)).sum())
            item["quantiles"] = {str(q): float(v) for q, v in finite.quantile([0, .01, .5, .99, 1]).items()} if len(finite) else {}
            item["zeros"] = int((finite == 0).sum())
        else:
            item["top_values"] = [[str(k)[:100], int(v)] for k, v in series.value_counts().head(6).items()]
        records.append(item)
    return records


class AnalysisService:
    def __init__(self, spec, validation, artifacts, budget, log):
        self.spec, self.artifacts, self.budget, self.log = spec, artifacts, budget, log
        self.validation_id = validation["validation_id"]
        # Trusted loader discards validation rows before exposing a frame to a query.
        indices = artifacts.read_json(validation["train_rows_artifact_id"])
        self.fit = read_training(artifacts.resolve(spec["train_artifact_id"]), spec).iloc[indices].copy()
        self.fit_rows = len(self.fit)
        if len(self.fit) > 20000:
            self.fit = self.fit.sample(20000, random_state=42).copy()
        self.test = read_table(artifacts.resolve(spec['test_artifact_id'])).drop(columns=targets(spec), errors='ignore')
        self.test_rows = len(self.test)
        if len(self.test) > 20000:
            self.test = self.test.sample(20000, random_state=42).copy()
        self.dictionary = (read_table(artifacts.resolve(spec['dictionary_artifact_id']))
                           if spec.get('dictionary_artifact_id') else None)
        self.records, self.cache, self.lock = [], {}, threading.RLock()

    def recent(self):
        with self.lock:
            return deepcopy(self.records[-6:])

    def analyze(self, query):
        import numpy as np
        import pandas as pd
        if not self.budget.snapshot()["search_allowed"]:
            raise AgentError("SEARCH_CLOSED", "Analysis cannot consume the final publication reserve")
        if self.budget.phase_remaining() < 1:
            raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Analysis reached the current phase deadline')
        # Independently validate the public tool, including direct callers.
        if (not set(ANALYSIS_SCHEMA['required']) <= set(query) or set(query)-set(ANALYSIS_SCHEMA['properties'])
                or query['kind'] not in ANALYSIS_SCHEMA['properties']['kind']['enum']
                or not isinstance(query['columns'], list) or len(query['columns']) > 12
                or any(not isinstance(c, str) for c in query['columns'])
                or not (query['group_by'] is None or isinstance(query['group_by'], str))
                or not isinstance(query['include_target'], bool)):
            raise AgentError("INVALID_ANALYSIS", "Invalid bounded analysis query")
        if query['kind'] == 'report':
            if query['columns'] or query['include_target'] or query['group_by'] is not None:
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
                return deepcopy(self.cache[digest])
        target = self.spec['target_column']
        target_columns = targets(self.spec)
        excluded = set(self.spec['excluded_features']) | set(target_columns)
        requested = list(dict.fromkeys(query['columns']))
        if not requested:
            requested = [c for c in self.fit if c not in excluded][:8]
        if any(c not in self.fit or c in excluded for c in requested):
            raise AgentError("INVALID_ANALYSIS", "columns must contain existing input features only. Remove the target/excluded columns; use include_target=true for fit-only target summaries.",
                             details={'invalid_columns': [c for c in requested if c not in self.fit or c in excluded],
                                      'target_column': target})
        columns = requested + (target_columns if query['include_target'] else [])
        # Deterministic, bounded computation; row-level labels never leave this service.
        data = self.fit[columns]
        sampled = self.fit_rows > len(data)
        if sampled:
            data = data.sample(20000, random_state=42)
        result = {"scope": "fit_only", "validation_id": self.validation_id,
                  "fit_rows": self.fit_rows, "analyzed_rows": len(data), "sampled": sampled, "query": query}
        if query['kind'] in {'compare', 'dictionary'} and query['include_target']:
            raise AgentError('INVALID_ANALYSIS', 'Feature comparisons and dictionary queries do not expose test targets')
        if query['kind'] == 'dictionary':
            dictionary = self.dictionary
            if dictionary is not None and 'feature' in dictionary:
                dictionary = dictionary[dictionary['feature'].isin(requested)]
            result['dictionary'] = (json.loads(dictionary.head(12).to_json(orient='records', force_ascii=False))
                                    if dictionary is not None else [])
        elif query['kind'] == 'compare':
            missing = [c for c in requested if c not in self.test]
            if missing:
                raise AgentError('INVALID_ANALYSIS', 'Selected features are absent from the unlabelled test table',
                                 details={'missing_test_columns': missing})
            test = self.test[requested]
            if len(test) > 20000:
                test = test.sample(20000, random_state=42)
            result.update(scope='fit_and_unlabelled_test_features', fit=summarize(data),
                          test=summarize(test), analyzed_test_rows=len(test), test_rows=self.test_rows,
                          test_sampled=len(test) < self.test_rows)
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
                values = pd.qcut(values.replace([np.inf, -np.inf], np.nan), 5, duplicates='drop')
            values = values.astype(object).where(values.notna(), '__MISSING__').astype(str)
            counts = values.value_counts().head(12)
            result['groups'] = []
            for label, count in counts.items():
                record = {'group': str(label)[:100], 'rows': int(count)}
                if query['include_target']:
                    summary = summarize(data.loc[values == label, target_columns])
                    record['target'] = summary[0] if len(summary) == 1 else summary
                result['groups'].append(record)
        body = canonical(result)
        if len(body.encode()) > 16000:
            raise AgentError('ANALYSIS_TOO_LARGE', 'Select fewer columns for this query')
        art = self.artifacts.write_json('analysis/'+digest+'.json', result, 'research_analysis')
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
        response = {'artifact_id': art['artifact_id'], 'report': reply}
        with self.lock:
            self.cache[digest] = response
            self.records.append({'artifact_id': art['artifact_id'], 'query': deepcopy(query), 'scope': result['scope']})
        self.log.event('research.analysis_completed', artifact_id=art['artifact_id'], kind=query['kind'],
                       columns=requested, rows=len(data), scope=result['scope'])
        return deepcopy(response)
