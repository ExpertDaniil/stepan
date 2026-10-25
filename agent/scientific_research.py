"""Evidence-grounded research episodes, bounded diagnostics and admission review.

All store access is dispatcher-owned. Model prose is an explicit proposal, never
an authoritative fact or permission to inspect withheld labels.
"""
from copy import deepcopy


def episode_allowance(config):
    return min(1800, max(15, config.time_budget_seconds*config.research_episode_budget_fraction))

def compact_protocol(protocol):
    """Shrink displayed hints; full records remain in the scoped detail snapshot."""
    result = deepcopy(protocol)
    for index, record in enumerate(result.get('fit_diagnostics', [])):
        report = record.get('report', {})
        for field in ('hard_groups', 'error_associations'):
            values = report.get(field, [])
            report['omitted_'+field] = report.get('omitted_'+field, 0)+max(0, len(values)-4)
            report[field] = values[:4]
        record['detail_hint'] = 'read_memory section=fit_diagnostics/'+str(index)
    episode = result.get('episode')
    if episode and episode.get('last_test_plan'):
        for key, value in episode['last_test_plan'].items():
            if isinstance(value, str) and len(value) > 240:
                episode['last_test_plan'][key] = value[:240]+' [excerpt; read scientific_protocol for full record]'
                episode['last_test_plan_is_excerpt'] = True
    return result


def synthetic_notes(action, user):
    """Explicit mock/test fixture only; never applied to a live model response."""
    projection = user.get('profile_summary', user)
    episode = projection.get('scientific_protocol', {}).get('episode')
    best = projection.get('research_landscape', {}).get('strongest_measured_learned')
    evidence = list(action.get('evidence_ids', []))
    if best:
        evidence = list(dict.fromkeys([*evidence, best['experiment_id']]))
    return {'idea_id': episode['idea_id'] if episode else None,
        'question': episode['question'] if episode else 'Does the synthetic candidate improve the measured reference?',
        'mechanism': action['hypothesis'][:1000], 'decision_reason': 'Execute a bounded synthetic comparison of complete configurations.',
        'previous_result_interpretation': 'Only controller measurements establish a score; no component mechanism is proved.',
        'alternatives': [{'idea_id': None, 'question': 'Could the existing representation already capture the same signal?',
            'mechanism': 'A flexible baseline may already encode the proposed input relationship.',
            'next_test': 'Compare this complete candidate with the recorded baseline on the unchanged split.',
            'why_deferred': 'First execute the proposed controlled synthetic comparison.',
            'evidence_ids': evidence, 'knowledge_ids': []}],
        'test_plan': {'observation': 'The supplied synthetic schema and measured reference are available; improvement remains untested.',
            'prediction': 'The complete candidate improves the recorded reference on the fixed selection protocol.',
            'falsifier': 'The complete candidate matches or worsens the reference score under the same protocol.',
            'cheap_check': 'Validate source, feature column names and the declared prediction interface before fitting.',
            'counterargument': 'The existing representation may already capture the signal and the new inputs may add noise.',
            'evidence_ids': evidence, 'continuation': 'stop' if action['action'] == 'stop' else 'followup' if episode else 'new',
            'switch_reason': '', 'alternatives_unavailable_reason': ''}}


def episode_context(memory, researcher_id, jobs, allowance):
    decisions = [d for d in memory.get('decisions', []) if d.get('researcher_id') == researcher_id and d.get('idea_id')]
    if not decisions:
        return None
    latest = decisions[-1]
    linked = [d for d in decisions if d['idea_id'] == latest['idea_id']]
    observations = {o['experiment_id']: o for d in linked for o in d['observations']}
    started = min((d['created_monotonic'] for d in linked if d.get('created_monotonic') is not None), default=0)
    spent = sum(j.event['duration_seconds'] for j in jobs if j.reported and j.phase == 'train'
                and not getattr(j, 'diagnostic', False) and j.experiment_id in observations and j.event
                and getattr(j, 'started', 0) >= started)
    last = linked[-1]
    return {'idea_id': latest['idea_id'], 'question': latest['question'],
            'measured_experiment_ids': [eid for eid, o in observations.items() if o['primary_metric'] is not None],
            'failed_experiment_ids': [eid for eid, o in observations.items() if o['state'] in {'failed', 'invalid', 'timed_out', 'cancelled'}],
            'decision_count': len(linked), 'charged_training_seconds': round(spent, 3),
            'allowance_seconds': allowance, 'remaining_seconds': round(max(0, allowance-spent), 3),
            'last_test_plan': deepcopy(last.get('test_plan')),
            'evidence_ids': list(dict.fromkeys(e for d in linked for e in d['evidence_ids'])),
            'knowledge_ids': list(dict.fromkeys(e for d in linked for e in d.get('knowledge_ids', []))),
            'instruction': 'Continue this question when useful, or explicitly explain a switch/reconsideration. '
                'A measured joint configuration does not establish or refute each component. No minimum trial count.'}


class ScientificResearch:
    def __init__(self, controller):
        self.c = controller
        self.funding = {}
        from agent.fit_diagnostics import FitDiagnostics
        self.diagnostics = FitDiagnostics(controller)

    def context(self, visible_ids, memory, researcher_id):
        c = self.c
        previous_reports = len(self.diagnostics.records)
        self.diagnostics.ensure(visible_ids)
        if len(self.diagnostics.records) > previous_reports:
            c.publish_research_updates()
        allowance = episode_allowance(c.config)
        episode = episode_context(memory, researcher_id, c.runner.jobs.values(), allowance)
        reports = self.diagnostics.visible(visible_ids)
        extended = bool(episode and (len(episode['measured_experiment_ids']) >= 2 or episode['failed_experiment_ids']))
        return {'version': ('scientific-v1' if c.config.research_protocol == 'legacy' else 'research-v2'), 'episode': episode, 'fit_diagnostics': reports,
                'decision_record_required': c.config.research_protocol == 'legacy',
                'deliberation': {'extended_requested': extended,
                    'reason': 'Multiple related measurements or an execution failure warrant reassessing explanations.' if extended else 'Use the normal dialogue budget for a well-specified next test.',
                    'provider_thinking_application': 'unverified'},
                'instruction': ('Supply research_notes.test_plan. ' if c.config.research_protocol == 'legacy' else 'Analyze useful questions and submit executable changes; explanatory forms are optional. ')+
                    'Cite completed evidence. Diagnostics discover candidate patterns, not proven mechanisms. '
                    'Do not optimize private reasoning length or invent measured outcomes.'}

    def review(self, plan, intent, branch):
        c = self.c
        self.funding.pop(plan['plan_id'], None)
        notes = intent.get('research_notes') or {}
        test = notes.get('test_plan')
        if not test and intent.get('investigation'):
            inv = intent['investigation']
            from agent.research_memory import idea_id
            episode = intent.get('scientific_protocol', {}).get('episode')
            same_question = episode and episode['idea_id'] == idea_id(inv['question'], inv['mechanism'])
            test = {'evidence_ids': intent.get('evidence_ids', []), 'continuation': 'followup' if same_question else 'new',
                    'counterargument': inv['alternative_explanation'], 'prediction': inv['expected_observation'],
                    'falsifier': inv['decision_rule'], 'cheap_check': inv['question']}

        # Legacy saved actions remain executable; new contexts enforce the record
        # in Gateway and Researcher before producing a plan.
        if not test:
            return None
        protocol = intent.get('scientific_protocol', {})
        episode = protocol.get('episode')
        if episode:
            from agent.research_memory import build_research_memory
            memory = (c.graph.research_memory(branch, []) if c.graph else
                      build_research_memory(
                          c.store, c.artifacts, [r[0] for r in c.store.db.execute(
                              'SELECT experiment_id FROM experiments WHERE run_id=?', (c.store.run_id,))]))
            memory['decisions'] = [d for d in memory['decisions']
                                   if (d.get('configuration') or {}).get('plan_id') != plan['plan_id']]
            latest = episode_context(memory, plan['researcher_id'], c.runner.jobs.values(), episode['allowance_seconds'])
            if latest and latest['idea_id'] == episode['idea_id']:
                episode = latest
        estimate = c.budget.duration.estimate(plan)
        expensive = estimate >= 30 or (intent.get('tuning') or {}).get('budget_seconds', 0) >= 60
        all_evidence = set(intent.get('evidence_ids', [])) | set(test['evidence_ids'])
        reference = intent.get('strongest_measured_reference')
        reference_ids = {reference} if reference else set()
        reference_ids.update(r['artifact_id'] for r in protocol.get('fit_diagnostics', []) if r['experiment_id'] == reference)
        # A newer, better result delivered during the dialogue is also a valid
        # comparator; never force a stale context's older reference.
        if reference and reference.startswith('exp-') and hasattr(c, 'store'):
            anchor = c.store.experiment_evaluation(reference)
            if anchor:
                old = anchor['primary_metric']
                for ident in all_evidence:
                    if not ident.startswith('exp-'):
                        continue
                    result = c.store.experiment_evaluation(ident)
                    if result and result['valid'] and not result['violations']:
                        new = result['primary_metric']
                        if (all(new[k] == old[k] for k in ('dataset_id', 'validation_id', 'fidelity', 'metric_id', 'direction'))
                                and (new['value'] <= old['value'] if old['direction'] == 'minimize' else new['value'] >= old['value'])):
                            reference_ids.add(ident)
        reason = None
        if expensive and reference_ids and not all_evidence & reference_ids:
            reason = 'An expensive action needs a comparison with the strongest visible measured reference or its fit-diagnostic report.'
        if episode and test['continuation'] == 'followup' and episode['remaining_seconds'] < estimate:
            reason = 'The current question has consumed its training allowance. State an explicit reconsideration using new evidence or switch to a cheaper unresolved question.'
        if episode and test['continuation'] == 'reconsider' and episode['remaining_seconds'] < estimate:
            if not all_evidence-set(episode.get('evidence_ids', [])):
                reason = 'Renewing a spent episode needs newly observed evidence, not only a renamed question.'
        allowance = episode_allowance(c.config) if hasattr(c, 'config') else 600
        funded = episode['remaining_seconds'] if episode and test['continuation'] == 'followup' else allowance
        if not reason and funded < estimate:
            reason = 'The predicted fit exceeds this question\'s funding. Choose a cheaper discriminating test or reconsider the question with new evidence.'
        if not reason:
            self.funding[plan['plan_id']] = max(1, int(funded))
            plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], self.funding[plan['plan_id']])
        report = {'plan_id': plan['plan_id'], 'branch_id': branch, 'expensive': expensive,
                  'funded_training_seconds': self.funding.get(plan['plan_id']),
                  'estimated_training_seconds': estimate, 'counterargument': test['counterargument'],
                  'prediction': test['prediction'], 'falsifier': test['falsifier'],
                  'cheap_check': test['cheap_check'], 'evidence_ids': sorted(all_evidence),
                  'admission': 'deferred' if reason else 'admitted', 'reason': reason,
                  'model_prose_verified': False, 'extra_llm_calls': 0}
        art = c.artifacts.write_json('research/review-'+plan['plan_id']+'.json', report, 'research_preflight')
        c.log.event('research.preflight_reviewed', artifact_id=art['artifact_id'], plan_id=plan['plan_id'],
                    expensive=expensive, admission=report['admission'], reason=reason)
        if reason:
            c.record_research_feedback(plan['researcher_id'], branch, 'research_design_deferred', reason,
                                       code='RESEARCH_DESIGN_REVIEW')
        return reason
