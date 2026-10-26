"""Dispatcher-owned error questions and paired SELECTION comparisons.

The model receives advice, not forms. Internal assignments freeze the proposal's
base and group before evaluation. Group discoveries are never additive claims.
"""
from copy import deepcopy
import hashlib
import time

from agent.contracts import canonical
from agent.errors import AgentError


def group_mask(data, truth, definition):
    import numpy as np
    import pandas as pd
    kind = definition['kind']
    if kind == 'target_class':return np.asarray(truth).reshape(-1).astype(str) == definition['label']
    series = data[definition['column']]
    if kind == 'missing':return series.isna().to_numpy()
    if kind == 'equals':return (series.notna() & series.astype(str).eq(definition['value'])).to_numpy()
    if kind == 'range':
        number = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
        mask = number.notna()
        if definition['lower'] is not None:mask &= number.ge(definition['lower'])
        if definition['upper'] is not None:mask &= number.lt(definition['upper'])
        return mask.to_numpy()
    raise ValueError('Unknown frozen group definition')


def paired_result(task, truth, base, candidate, mask):
    import numpy as np
    from agent.evaluation import score_predictions
    y, a, b = np.asarray(truth), np.asarray(base), np.asarray(candidate)
    mask = np.asarray(mask, dtype=bool)
    direction = 1 if task['metric']['direction'] == 'minimize' else -1
    def part(selected):
        count = int(selected.sum())
        if not count:return {'rows': 0, 'base': None, 'candidate': None, 'improvement': None}
        try:
            left = float(score_predictions(task, y[selected], a[selected]))
            right = float(score_predictions(task, y[selected], b[selected]))
            result = {'rows': count, 'base': left, 'candidate': right, 'improvement': (left-right)*direction}
        except ValueError:
            result = {'rows': count, 'base': None, 'candidate': None, 'improvement': None,
                      'limitation': 'Metric undefined on this subset'}
        if task['task_kind'] == 'regression':
            ya = y[selected];pa = a[selected].reshape(ya.shape);pb = b[selected].reshape(ya.shape)
            result['sse_reduction'] = float(((ya-pa)**2).sum()-((ya-pb)**2).sum())
        return result
    return {'scope': 'paired_selection_rows', 'overall': part(np.ones(len(y), dtype=bool)),
            'inside_group': part(mask), 'outside_group': part(~mask),
            'group_scores_additive': False,
            'limitations': 'Complete package on the unchanged selection rows. F1/subset scores are not additive; '
                'no component causality, attainable gain or independent confirmation is inferred.'}


VERDICT_PRIORITY = {'fixable': 3.0, 'explainable': 2.0, 'noise': 0.1}


def slice_problem(item, source_artifact_id):
    """A mined slice as an error-research problem; its card is what researchers read."""
    from agent.error_slices import compact_card
    robust = [r for r in item.get('remedies', []) if r['robust']]
    gain = robust[0]['relative_gain'] if robust else 0.0
    definition = {'kind': 'rule', 'slice_id': item['slice_id'], 'conditions': deepcopy(item['conditions']),
                  'entity': deepcopy(item.get('entity'))}
    return {'definition': definition, 'priority_cost': VERDICT_PRIORITY[item['verdict']]+100*gain+item['excess'],
            'row_membership_sha256': hashlib.sha256(canonical(definition).encode()).hexdigest(),
            'card': compact_card(item, source_artifact_id), 'fixability': item['verdict']}


def slice_question(card):
    robust = [r for r in card.get('remedies', []) if r['robust']]
    head = f"Slice {card['slice_id']} [{card['rule']}]: {100*card['row_share']:.1f}% rows carry {100*card['loss_share']:.1f}% of the loss."
    if card['verdict'] == 'fixable' and robust:
        best = robust[0]
        return (head+f" Verdict fixable: remedy {best['remedy_ref']} ({best['action']}) improved held-out OOF folds "
                f"{best['folds_improved']}/{best['folds']}. Execute it with a features action (slice_remedies), alone or with features, "
                "or explain from evidence why not.")
    if card['verdict'] == 'explainable':
        features = ', '.join(t['feature'] for t in (card.get('explainability') or {}).get('top_features', [])[:4])
        return head+f" Verdict explainable: the residual is predictable (via {features}). Design a feature_program that gives the model this signal."
    return head+' Verdict noise: no tested correction or input explains it; prefer another slice or direction.'


def rule_membership(definition, data, base_prediction, spec, history_source=None):
    """Rows of ``data`` inside a rule slice, using the research base's own predictions as meta."""
    import numpy as np
    from agent import slice_actions as actions
    regression = spec['task_kind'] == 'regression'
    kind = spec['metric']['prediction_kind']
    labels = list(spec.get('allowed_labels') or [])
    values = np.asarray(base_prediction)
    if regression:
        base, probabilities = values.astype(float).reshape(len(values), -1)[:, 0], None
    elif kind == 'probabilities':
        probabilities = values.astype(float)
        base = np.asarray(labels)[probabilities.argmax(axis=1)]
    else:
        base, probabilities = values.reshape(-1).astype(str), None
    history = None
    if definition.get('entity') and any(c['column'].startswith(actions.HISTORY_PREFIX) for c in definition['conditions']):
        if history_source is None:
            raise ValueError('History rule needs fit-partition rows')
        from agent.strong import EntityHistory
        entity = definition['entity']
        fit_x, fit_y = history_source
        history = EntityHistory(entity['columns'], entity['period_column'], prefix=actions.HISTORY_PREFIX).fit(fit_x, fit_y)
        history = history.transform(data[fit_x.columns]).reset_index(drop=True)
    meta = actions.meta_frame(base, regression=regression, probabilities=probabilities, history=history)
    return actions.rule_mask(actions.rule_frame(data, meta), definition['conditions'])


class ErrorResearch:
    def __init__(self, controller):
        self.c = controller
        self.problems, self.assignments, self.bindings = {}, {}, {}
        self.results, self.revision = [], 0

    def save(self):
        self.revision += 1
        art = self.c.artifacts.write_json(f'research/error-queue-{self.revision:05d}.json',
            {'version': 'error-research-v1', 'problems': list(self.problems.values()),
             'assignments': list(self.assignments.values()), 'results': self.results}, 'error_research_queue')
        self.artifact_id = art['artifact_id']

    def ingest(self, record):
        from agent.artifacts import file_sha256
        identity_id = record['identity_artifact_id']
        for aid in (identity_id, record['artifact_id']):
            if file_sha256(self.c.artifacts.resolve(aid)) != self.c.artifacts.meta(aid)['sha256']:
                raise AgentError('MEMORY_SOURCE_CHANGED', 'Diagnostic source changed')
        identity = self.c.artifacts.read_json(identity_id)
        if (identity.get('experiment_id') != record['experiment_id']
                or any(identity.get(k) != v for k, v in self.identity().items())):
            raise AgentError('MEMORY_SCOPE_DENIED', 'Diagnostic identity differs from the current task/protocol')
        if self.c.artifacts.read_json(record['artifact_id']) != record['report']:
            raise AgentError('MEMORY_SOURCE_CHANGED', 'Diagnostic report differs from its immutable artifact')
        scope = self.publication(record['experiment_id'])
        slices = (record['report'].get('error_slices') or {}).get('slices') or []
        groups = [slice_problem(item, record['artifact_id']) for item in slices] or record['report'].get('hard_groups', [])
        for group in groups:
            key = hashlib.sha256(canonical([identity, record['artifact_id'], group['definition']]).encode()).hexdigest()
            if key in self.problems:continue
            self.problems[key] = {'key': key, 'observation': deepcopy(group.get('card', group)), 'source_artifact_id': record['artifact_id'],
                'identity_artifact_id': identity_id, 'scope': deepcopy(scope),
                'base_experiment_id': record['experiment_id'], 'definition': deepcopy(group['definition']),
                'status': 'open', 'attempts': [], 'completed_monotonic': record['completed_monotonic'],
                'priority': group['priority_cost'],
                'focus_key': hashlib.sha256(canonical([self.identity(),
                    group.get('row_membership_sha256', group['definition'])]).encode()).hexdigest(),
                'fixability': group.get('fixability', 'unknown; high error is measured cost, not evidence of recoverable signal'),
                'sampled': record['report']['sampled'], 'support_scope': 'diagnostic_sample_only'}
        self.save()

    def identity(self):
        c = self.c
        return {k: c.spec[k] for k in ('dataset_id', 'task_spec_id')} | {
            'validation_id': c.validation['validation_id'], 'protocol_sha256': c.validation['protocol_sha256']}

    def publication(self, eid, branch=None):
        """Dispatcher-only provenance; ancestor observations obey the fork cut."""
        graph = getattr(self.c, 'graph', None)
        if graph and branch is None:
            row = self.c.store.db.execute('SELECT w.branch_id FROM g_bindings b JOIN g_work_orders w '
                'USING(run_id,work_order_id) WHERE b.run_id=? AND b.experiment_id=? LIMIT 1',
                (self.c.store.run_id, eid)).fetchone()
            branch = row[0] if row else None
        return {**self.identity(), 'branch_id': branch or 'legacy',
                'published_event_seq': graph.cut() if graph else None}

    def allowed(self, scope, branch):
        if not scope or any(scope.get(k) != v for k, v in self.identity().items()):return False
        graph = getattr(self.c, 'graph', None)
        if not graph or branch == 'legacy':return True
        cuts = dict(graph.lineage_cuts(branch))
        origin, published = scope.get('branch_id'), scope.get('published_event_seq')
        return origin in cuts and isinstance(published, int) and published <= cuts[origin]

    def visible_source(self, eid, visible):
        if eid not in visible:return False
        c = self.c
        exp = c.store.get(eid)
        plan = exp['plan']
        ev = c.store.experiment_evaluation(eid)
        return (exp['state'] == 'evaluated'
            and all(plan.get(k) == v for k, v in self.identity().items() if k != 'protocol_sha256')
            and ev and ev['valid'] and not ev['violations']
            and ev['primary_metric']['validation_id'] == c.validation['validation_id']
            and ev['primary_metric']['dataset_id'] == c.spec['dataset_id'])

    def advice(self, ids, peer, branch='legacy'):
        c = self.c
        visible = set(ids)
        measured = [(c.store.experiment_evaluation(eid)['primary_metric'], eid) for eid in ids
                    if self.visible_source(eid, visible)]
        if not measured:return {'instruction': 'Establish a learned reference; useful analysis and executable changes need no research forms.'}
        metric, base = max(measured, key=lambda p: p[0]['value']*(1 if p[0]['direction'] == 'maximize' else -1))
        visible = {eid for _, eid in measured}
        assignment_key = branch+'/'+peer
        previous = self.assignments.get(assignment_key)
        eligible = [p for p in self.problems.values() if p['base_experiment_id'] in visible and self.allowed(p['scope'], branch)]
        latest = {}
        for p in eligible:
            eid = p['base_experiment_id']
            if eid not in latest or p['completed_monotonic'] > latest[eid]['completed_monotonic']:latest[eid] = p
        candidates = [p for p in eligible if p['source_artifact_id'] == latest[p['base_experiment_id']]['source_artifact_id']]
        candidates.sort(key=lambda p: (p['base_experiment_id'] == base, p['priority']), reverse=True)
        previous_problem = self.problems.get(previous.get('problem_key')) if previous else None
        # A dialogue may start before its first OOF report exists. Refresh an
        # empty assignment once evidence arrives; dispatched tickets stay frozen.
        if (previous and previous.get('leader_experiment_id') == base and previous.get('status') == 'active'
                and self.allowed(previous.get('scope'), branch) and previous['base_experiment_id'] in visible
                and (previous_problem in eligible if previous_problem else not candidates)):
            assigned = previous
        else:
            occupied = {a.get('focus_key') for name, a in self.assignments.items() if name != assignment_key
                and a.get('branch_id') == branch and a.get('status') == 'active'
                and a['base_experiment_id'] in visible and self.allowed(a.get('scope'), branch)}
            problem = next((p for p in candidates if p['focus_key'] not in occupied), candidates[0] if candidates else None)
            shared = bool(problem and problem['focus_key'] in occupied)
            assigned = {'researcher_id': peer, 'branch_id': branch,
                'base_experiment_id': problem['base_experiment_id'] if problem else base,
                'leader_experiment_id': base, 'problem_key': problem['key'] if problem else None,
                'scope': deepcopy(problem['scope']) if problem else self.publication(base, branch),
                'focus_key': problem['focus_key'] if problem else None, 'shared_focus': shared,
                'source_artifact_id': problem['source_artifact_id'] if problem else None,
                'observation': deepcopy(problem['observation']) if problem else None,
                'definition': deepcopy(problem['definition']) if problem else None,
                'status': 'active', 'question': (slice_question(problem['observation']) if problem and problem['definition'].get('kind') == 'rule' else
                    'Investigate '+canonical(problem['definition']) if problem else
                    ('Investigate representation and systematic bias.' if peer.endswith('1') else 'Investigate alternate error explanation or estimator structure.')),
                'next_step': 'Read the current solution and available fit OOF errors; analyze a useful question and submit executable changes.'}
            if shared:assigned['question'] += ' Same observed rows as the peer; consider an alternative explanation or remedy.'
            self.assignments[assignment_key] = assigned
            self.save()
        problem = self.problems.get(assigned.get('problem_key'))
        from agent.fit_diagnostics import compact_group
        return {'leader': {'experiment_id': base, 'metric': metric},
            'recommendation': {k: deepcopy(assigned[k]) for k in ('question', 'next_step', 'base_experiment_id')},
            'observation': (deepcopy(assigned['observation']) if (assigned.get('observation') or {}).get('slice_id')
                            else compact_group(assigned['observation']) if assigned.get('observation') else None),
            'source_experiment_id': problem['base_experiment_id'] if problem else None,
            'source_artifact_id': assigned['source_artifact_id'], 'shared_focus': assigned.get('shared_focus', False),
            'detail_hint': 'read_memory section=error_observation for the frozen source report',
            'diagnostic_current_for_leader': bool(problem and problem['base_experiment_id'] == base),
            'relevant_attempts': deepcopy([a for a in problem.get('attempts', [])
                if a['candidate_experiment_id'] in visible and a['proposal_base_experiment_id'] in visible
                and self.allowed(a.get('scope'), branch)][-3:]) if problem else [],
            'peer_status': [{k: deepcopy(a[k]) for k in ('researcher_id', 'question', 'status')}
                for name, a in self.assignments.items() if name != assignment_key and a.get('branch_id') == branch
                and a['base_experiment_id'] in visible and self.allowed(a.get('scope'), branch)],
            'instruction': 'Recommendation only. Change direction or combine changes freely. No problem IDs, expected gain or provenance forms are required. '
                'Slices are ranked by loss share minus row share on fit-only OOF; a fixable slice lists remedy_ref values '
                'you may execute via a features action field slice_remedies (frozen, cross-fitted corrections). '
                'Use run_analysis(experiment_id="best") for aligned fit-only OOF data; old reports remain tied to their source.'}

    def bind(self, plan, ticket):
        peer = plan['researcher_id']
        assigned = deepcopy(self.assignments.get(ticket['work_order']['branch_id']+'/'+peer, {}))
        # Called after a dialogue; ticket carries its dispatch-time assignment.
        assigned = deepcopy(ticket.get('error_assignment', assigned))
        assigned['work_order_id'] = ticket['work_order']['work_order_id']
        assigned['plan_id'] = plan['plan_id']
        self.bindings[plan['plan_id']] = assigned
        return assigned

    def checkpoint(self, peer, question, error=None, branch='legacy'):
        assigned = self.assignments.get(branch+'/'+peer)
        if assigned:
            assigned['question'] = question[:1000] or assigned['question']
            assigned['next_step'] = ('Repair the saved error, reuse completed analyses and drafts, then submit the next executable version.'
                if error else 'Continue this question from saved analyses, transcripts and drafts; avoid rerunning completed checks.')
            assigned['last_error'] = str(error)[:600] if error else None
            self.save()

    def compare(self, eid, binding, journal=True):
        """Paired selection comparison; ``journal`` adds the readable research line (research candidates only)."""
        c = self.c
        base = binding.get('base_experiment_id')
        if not base or base == eid:return
        try:
            from agent.evidence import compare
            contrast = compare(c.store, c.artifacts, base, eid)
            if not contrast['comparable']:return
            import numpy as np
            import pandas as pd
            from agent.artifacts import file_sha256
            from agent.data_transport import transport
            from agent.predictions import targets, validation_columns, checked_matrix
            protocol = c.evaluator.protocols[c.validation['validation_id']]
            rows = protocol.get('selection_indices', protocol['valid_indices'])
            columns = validation_columns(c.spec);kind = c.spec['metric']['prediction_kind']
            leader = c.store.best()
            # Effect is measured against the research base; utility against the
            # current leader, when that is a different measured solution.
            versus_leader = leader not in {None, base, eid} and compare(c.store, c.artifacts, leader, eid)['comparable']
            matrices, sources = [], []
            for ident in (base, eid, *([leader] if versus_leader else [])):
                source = c.evaluator.diagnostic_sources.get(ident)
                if not source:raise ValueError('No trusted saved predictions for '+ident)
                path = c.artifacts.resolve(source['predictions_artifact_id'])
                if file_sha256(path) != source['predictions_sha256']:raise ValueError('Saved prediction hash changed')
                frame = pd.read_csv(path, dtype={col:str for col in columns} if kind == 'labels' else None, keep_default_na=False)
                if list(frame.columns) != ['__row_index', *columns] or frame['__row_index'].tolist() != protocol['valid_indices']:
                    raise ValueError('Prediction row identity differs')
                values = frame.set_index('__row_index').loc[rows, columns].to_numpy()
                matrices.append(checked_matrix(values, len(rows), len(columns), kind, labels=c.spec['allowed_labels']))
                sources.append(source['predictions_artifact_id'])
            definition = binding.get('definition')
            names = targets(c.spec)
            if definition and definition.get('kind') == 'rule':
                entity = definition.get('entity') or {}
                wanted = [*names, *[x['column'] for x in definition['conditions'] if not x['column'].startswith('__')],
                          *entity.get('columns', []), *([entity['period_column']] if entity else [])]
            else:
                wanted = [*names, *([definition['column']] if definition and definition.get('column') else [])]
            data = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, columns=list(dict.fromkeys(wanted)), rows=rows)
            truth = data[names if len(names) > 1 else names[0]]
            if definition and definition.get('kind') == 'rule':
                source = None
                if definition.get('entity'):
                    entity = definition['entity']
                    columns = list(dict.fromkeys([*entity['columns'], entity['period_column']]))
                    fit = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, columns=[*columns, names[0]],
                                                      rows=protocol['train_indices'])
                    source = (fit[columns], fit[names[0]].to_numpy(dtype=float))
                mask = rule_membership(definition, data, matrices[0], c.spec, source)
            else:
                mask = group_mask(data, truth, definition) if definition else np.zeros(len(rows), dtype=bool)
            result = paired_result(c.spec, truth, matrices[0], matrices[1], mask)
            result['versus_current_leader'] = (
                {'experiment_id': leader, **paired_result(c.spec, truth, matrices[2], matrices[1], mask)['overall']}
                if versus_leader else {'experiment_id': leader, 'same_as': 'candidate' if leader == eid else 'research_base'})
            result.update(candidate_experiment_id=eid, proposal_base_experiment_id=base,
                current_leader_experiment_id=leader, definition=definition,
                group_fixed_before_evaluation=True, prediction_artifact_ids=sources,
                work_order_id=binding.get('work_order_id'), source_artifact_id=binding.get('source_artifact_id'))
            result['status'] = ('overall_improved' if result['overall']['improvement'] > 0 else
                'partial_group_improvement' if (result['inside_group']['improvement'] or 0) > 0 else 'no_gain')
            art = c.artifacts.write_json('research/paired-'+eid+'.json', result, 'paired_research_result')
            summary = {'artifact_id': art['artifact_id'], **{k:deepcopy(result[k]) for k in (
                'candidate_experiment_id','proposal_base_experiment_id','status','overall','inside_group','outside_group',
                'versus_current_leader')},
                'scope': self.publication(eid, binding.get('branch_id')),
                'source_artifact_id': binding.get('source_artifact_id')}
            self.results.append(summary)
            key = binding.get('problem_key')
            if key in self.problems:
                self.problems[key]['attempts'].append(summary)
                self.problems[key]['priority'] /= 1.25
                self.problems[key]['status'] = result['status']
            peer = binding.get('researcher_id')
            assignment_key = binding.get('branch_id', 'legacy')+'/'+str(peer)
            if (assignment_key in self.assignments
                    and self.assignments[assignment_key].get('problem_key') == binding.get('problem_key')):
                self.assignments[assignment_key]['status'] = 'evaluated'
            self.save()
            c.log.event('research.paired_comparison', artifact_id=art['artifact_id'], experiment_id=eid,
                        base_experiment_id=base, status=result['status'], leader_experiment_id=leader,
                        improvement_vs_leader=result['versus_current_leader'].get('improvement'))
            if journal:
                c.log.event('research.journal', kind='candidate', experiment_id=eid, researcher_id=peer,
                            slice_id=(definition or {}).get('slice_id'), status=result['status'],
                            line=candidate_line(c, eid, peer, definition, result))
        except (AgentError, ValueError, OSError, KeyError) as error:
            c.log.event('research.paired_unavailable', experiment_id=eid, base_experiment_id=base, reason=str(error)[:500])


def candidate_line(c, eid, peer, definition, result):
    """One human-readable journal line: what was tried, on which slice, with what paired effect."""
    plan = c.store.get(eid)['plan']
    params = plan['model']['hyperparameters']
    changes = []
    if params.get('slice_corrections'):
        changes.append('remedies '+', '.join(f"{x['slice_id']}:{x['action']}" for x in params['slice_corrections']))
    if params.get('feature_program'):
        changes.append('features '+', '.join(step['name'] for step in params['feature_program'][:4])
                       +(f" (+{len(params['feature_program'])-4})" if len(params['feature_program']) > 4 else ''))
    if not changes:
        changes.append(plan['recipe_id'])
    metric = c.spec['metric']['metric_id']
    def pair(part):
        if part.get('base') is None or part.get('candidate') is None:
            return 'n/a'
        return f"{part['base']:.6g}->{part['candidate']:.6g}"
    where = ''
    if definition and definition.get('kind') == 'rule':
        from agent.slice_actions import describe
        where = f" on {definition.get('slice_id')} [{describe(definition['conditions'])}]"
    elif definition:
        where = ' on '+canonical(definition)[:120]
    leader = result.get('versus_current_leader', {})
    versus = (f"; vs leader {leader.get('improvement'):+.6g}" if isinstance(leader.get('improvement'), (int, float)) else '')
    return (f"{peer}: {'; '.join(changes)}{where}: slice {metric} {pair(result['inside_group'])}, "
            f"overall {pair(result['overall'])}{versus} => {result['status']}")

