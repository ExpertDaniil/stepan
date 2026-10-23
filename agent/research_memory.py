"""Read-only projection of immutable decisions and dispatcher-owned measurements.

No LLM verdict closes an idea. `open` means unresolved, not necessarily worth funding.
Graph callers supply visible work orders/experiments, including ancestor event cuts.
"""
from copy import deepcopy
import hashlib

from agent.contracts import canonical


def idea_id(question, mechanism):
    key = [' '.join(value.lower().split()) for value in (question, mechanism)]
    return 'research-idea-'+hashlib.sha256(canonical(key).encode()).hexdigest()[:24]


def build_research_memory(store, artifacts, experiment_ids, *, work_order_ids=None, branch_cuts=None):
    """Must be called by the dispatcher; model threads never access the store."""
    configurations, observations_by_id = {}, {}
    for ident in dict.fromkeys(experiment_ids):
        exp = store.get(ident)
        plan = exp['plan']
        core = store.fingerprint_core(plan, artifacts.meta(plan['code_artifact_id'])['sha256'])
        ev = store.experiment_evaluation(ident)
        trusted = exp['state'] == 'evaluated' and ev and ev['valid'] and not ev['violations']
        observation = {'experiment_id': ident, 'state': exp['state'],
                       'primary_metric': deepcopy(ev['primary_metric']) if trusted else None,
                       'diagnostics_artifact_id': ev['diagnostics_artifact_id'] if trusted else None}
        configurations.setdefault(canonical(core), []).append(observation)
        observations_by_id[ident] = observation
    with artifacts.lock:
        records = [ident for ident, entry in artifacts.entries.items()
                   if entry['meta']['kind'] == 'research_decision']
        series_artifacts = [ident for ident, entry in artifacts.entries.items()
                            if entry['meta']['kind'] == 'optuna_series']
    series_latest = {}
    for ident in series_artifacts:
        series = artifacts.read_json(ident)
        if branch_cuts is not None and (series['branch_id'] not in branch_cuts or series['graph_event_seq'] is None
                                       or series['graph_event_seq'] > branch_cuts[series['branch_id']]):
            continue
        old = series_latest.get(series['series_id'])
        if old is None or series['revision'] > old['revision']:
            series_latest[series['series_id']] = {**series, 'artifact_id': ident}
    visible_orders = set(work_order_ids) if work_order_ids is not None else None
    decisions, ideas = [], {}
    for ident in records:
        record = artifacts.read_json(ident)
        order = record.get('work_order_id')
        if (visible_orders is not None and order not in visible_orders) or (visible_orders is None and order is not None):
            continue
        notes = record.get('research_notes')
        if notes is None:
            notes = {'idea_id': None, 'question': record['hypothesis'], 'mechanism': record['expected_effect'],
                     'decision_reason': 'Not supplied by the model; legacy decision retained.',
                     'previous_result_interpretation': '', 'alternatives': []}
        selected = None
        # A stop is a budget/priority decision; it must not invent a new tested idea.
        candidates = ([notes] if record['action'] != 'stop' else [])+notes['alternatives']
        for candidate in candidates:
            key = candidate.get('idea_id') or idea_id(candidate['question'], candidate['mechanism'])
            if key not in ideas:
                ideas[key] = {'idea_id': key, 'question': candidate['question'], 'mechanism': candidate['mechanism'],
                              'status': 'open', 'decision_artifact_ids': [], 'tested_experiment_ids': [],
                              'evidence_ids': [], 'knowledge_ids': [], 'next_test': None, 'why_deferred': None}
            item = ideas[key]
            item['decision_artifact_ids'] = list(dict.fromkeys([*item['decision_artifact_ids'], ident]))
            for field in ('evidence_ids', 'knowledge_ids'):
                item[field] = list(dict.fromkeys([*item[field], *candidate.get(field, record.get(field, []))]))
            if candidate is notes:
                selected = key
                item['last_decision_role'] = 'selected'
            else:
                item['last_decision_role'] = 'deferred'
                item['next_test'], item['why_deferred'] = candidate['next_test'], candidate['why_deferred']
        configuration = record.get('configuration')
        observations = list(configurations.get(canonical(configuration['fingerprint_core']), [])) if configuration else []
        tuning_series = []
        for series in series_latest.values():
            if configuration is None or series['source_plan_id'] != configuration['plan_id']:
                continue
            summary = {key: deepcopy(series[key]) for key in ('series_id', 'study_id', 'artifact_id', 'status', 'stop_reason',
                'search_space', 'max_trials', 'budget_seconds', 'elapsed_seconds', 'proposed_trials', 'completed_trials',
                'failed_trials', 'cached_trials', 'warm_start_trials', 'best_before', 'best_experiment_id', 'best_metric')}
            summary['trials'] = []
            for trial in series['trials']:
                # A pre-fork snapshot cannot expose a later result through Store.
                observation = observations_by_id.get(trial['experiment_id'])
                if trial['work_order_id'] is not None and work_order_ids is not None and trial['work_order_id'] not in work_order_ids:
                    continue
                item = deepcopy(trial)
                if trial['primary_metric'] is not None and observation and observation['primary_metric'] is not None:
                    item['primary_metric'] = deepcopy(observation['primary_metric'])
                    observations.append(observation)
                elif trial['primary_metric'] is not None:
                    # Do not promote an artifact score without a visible trusted evaluation.
                    item['primary_metric'] = None
                summary['trials'].append(item)
            for field, eid_field in (('best_metric', 'best_experiment_id'),):
                eid = summary[eid_field]
                if eid and (eid not in observations_by_id or observations_by_id[eid]['primary_metric'] is None):
                    summary[field], summary[eid_field] = None, None
            before = summary['best_before']
            if before and (before['experiment_id'] not in observations_by_id or
                           observations_by_id[before['experiment_id']]['primary_metric'] is None):
                summary['best_before'] = None
            tuning_series.append(summary)
        observations = list({o['experiment_id']: o for o in observations}.values())
        if selected:
            item = ideas[selected]
            item['tested_experiment_ids'] = list(dict.fromkeys([*item['tested_experiment_ids'],
                *(o['experiment_id'] for o in observations if o['primary_metric'] is not None)]))
            item['status'] = ('measured' if item['tested_experiment_ids'] else
                              'running' if any(o['state'] in {'queued', 'running', 'trained'} for o in observations) or
                              any(s['status'] == 'running' for s in tuning_series) else
                              'failed' if observations and all(o['state'] in {'failed', 'invalid', 'timed_out', 'cancelled'} for o in observations) else 'proposed')
        for candidate in notes['alternatives']:
            key = candidate.get('idea_id') or idea_id(candidate['question'], candidate['mechanism'])
            if not ideas[key]['tested_experiment_ids'] and ideas[key].get('status') not in {'running', 'measured'}:
                ideas[key]['status'] = 'deferred'
        decisions.append({'decision_artifact_id': ident, 'action': record['action'], 'idea_id': selected,
            'question': notes['question'], 'decision_reason': notes['decision_reason'],
            'model_interpretation': notes['previous_result_interpretation'],
            'model_interpretations_verified': False, 'model_claims': record.get('claims', []),
            'notes_supplied': record.get('research_notes') is not None,
            'evidence_ids': record['evidence_ids'], 'configuration': deepcopy(configuration),
            'requested_tuning': record.get('tuning'), 'tuning_series': tuning_series,
            'observations': deepcopy(observations),
            'conclusion_scope': ('Only the listed complete configurations were measured; individual component effects remain unidentified.'
                if any(o['primary_metric'] is not None for o in observations) else
                'No verified measurement is attached to this decision; do not invent an outcome.')})
    return {'open_ideas': list(ideas.values()), 'decisions': decisions,
        'instruction': 'Ideas persist across decisions. Open means unresolved, not a requirement to test every idea. '
            'Observations come from the controller; interpretations and priority reasons come from the model. '
            'A failed combined pipeline does not refute each component or exhaust a model family. '
            'Tuning_series links the actual Optuna candidates and outcomes to the question. Reaching a trial/time budget does not prove a ceiling. '
            'Reuse idea_id when continuing an existing question. No semantic deduplication or causal attribution is claimed.'}
