"""One-way transition from branch exploration to refinement of the measured best."""
import json
import math

from agent.errors import AgentError
from agent.store import comparable


class RefinementPolicy:
    def refinement(self):
        row = self.db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name='refinement'", (self.run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def refinement_enabled(self):
        return self.policy['scheduling'].get('refinement_enabled', self.config.graph_refinement_enabled)

    def phase_limits(self):
        fraction = self.policy['scheduling'].get('search_fraction', self.config.graph_search_fraction)
        def kept(total):
            return max(0, total-math.ceil(total*(1-fraction)-1e-9))
        return {'search_fraction': fraction,
                'call_limit': min(self.config.llm_max_calls, max(2, kept(self.config.llm_max_calls))),
                'output_limit': min(self.config.llm_max_total_output_tokens, max(2*self.config.llm_bootstrap_output_tokens, kept(self.config.llm_max_total_output_tokens)))}

    def apply_phase_caps(self):
        if self.refinement_enabled() and not self.refinement():
            limits = self.phase_limits()
            # Fixed wall boundary: a changing estimate of final refit must not
            # move the 85/15 transition back and forth during research.
            deadline = self.budget.started + (self.config.time_budget_seconds-self.budget.output_reserve())*limits['search_fraction']
            self.budget.set_phase_caps(calls=limits['call_limit'], output_tokens=limits['output_limit'], deadline=deadline)
        else:
            self.budget.set_phase_caps()

    def refinement_due(self, pending=0):
        if not self.refinement_enabled() or self.refinement():
            return None
        limits = self.phase_limits()
        left = (self.config.time_budget_seconds-self.budget.output_reserve())*limits['search_fraction']-self.elapsed()/1000
        if left <= 1:
            return 'search_time_share_reached'
        if not pending:
            heads = [h for row in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND state='active'", (self.run_id,))
                     for h in self.heads(row[0]) if h['experiment_id']]
            costs = [self.budget.duration.estimate(self.store.get(h['experiment_id'])['plan']) for h in heads]
            train = sum(costs)/self.policy['max_train_workers'] if costs else 1
            publication = max((self.budget.candidate_reserve(self.store.get(h['experiment_id'])['plan']) for h in heads),
                              default=self.budget.reserve_seconds())
            api = self.budget.snapshot()['llm_calls_remaining'] > 0 and self.budget.token_snapshot()['output_available'] >= self.config.llm_max_output_tokens
            reason = self.budget.reasoning_estimate() if api else 0
            if left <= train+reason+publication+1:
                return 'next_research_batch_would_cross_phase_boundary'
        return None

    def best_origin(self, experiment_id):
        ev = self.store.experiment_evaluation(experiment_id)
        expected = {'dataset_id': self.spec['dataset_id'], 'validation_id': self.validation['validation_id'],
                    'metric_id': self.spec['metric']['metric_id'], 'direction': self.spec['metric']['direction'], 'fidelity': 'full'}
        if (not ev or not ev['valid'] or ev['violations'] or self.store.get(experiment_id)['state'] != 'evaluated'
                or not comparable(ev['primary_metric'], expected)):
            raise AgentError('INVALID_REFINEMENT_SOURCE', 'Refinement requires a trusted comparable full evaluation')
        head = {'experiment_id': experiment_id, 'solution_id': self.experiment_solution(experiment_id)}
        for row in self.db.execute('SELECT DISTINCT w.branch_id FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND b.experiment_id=? ORDER BY b.rowid', (self.run_id, experiment_id)):
            branch = self.branch(row[0])
            if self.head_satisfies(head, branch['constraints']):
                return branch
        raise AgentError('INVALID_REFINEMENT_SOURCE', 'No compatible recorded origin branch')

    def start_refinement(self, p, key):
        if not self.refinement_enabled():
            raise AgentError('REFINEMENT_DISABLED', 'Final refinement is disabled')
        if self.refinement():
            raise AgentError('REFINEMENT_ALREADY_STARTED', 'Final refinement is a one-way transition')
        if p['expected_graph_revision'] != self.revision():
            raise AgentError('GRAPH_REVISION_CONFLICT', 'Graph changed before refinement')
        if self.store.best() != p['source_experiment_id']:
            raise AgentError('BEST_CHANGED', 'Refinement must start from the current measured global best')
        if not self.budget.snapshot()['search_allowed']:
            raise AgentError('SEARCH_CLOSED', 'No refinement time remains after finalization reserve')
        if (self.runner.active() or self.budget.token_snapshot()['output_reserved'] or self.db.execute(
                "SELECT 1 FROM g_work_orders WHERE run_id=? AND state IN ('preparing','issued','submitted')", (self.run_id,)).fetchone()):
            raise AgentError('PEER_BUSY', 'Finish existing reasoning and training before final refinement')
        source = self.best_origin(p['source_experiment_id'])
        prior_heads = self.heads(source['branch_id'])
        solution_id = self.experiment_solution(p['source_experiment_id'])
        record = {'phase': 'refine', 'branch_id': source['branch_id'], 'source_experiment_id': p['source_experiment_id'],
                  'source_solution_id': solution_id, 'prior_heads': prior_heads, 'seed_heads': [],
                  'started_elapsed_ms': self.elapsed(), 'started_event_seq': 0, 'reason': p['reason'], 'phase_limits': self.phase_limits()}
        def mutate():
            if p['expected_graph_revision'] != self.revision() or self.store.best() != p['source_experiment_id']:
                raise AgentError('GRAPH_REVISION_CONFLICT', 'Refinement source changed')
            # Preserve the binary decision tree. Reactivate its measured origin
            # exclusively for tuning; G02 is prohibited in the final phase.
            for row in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND branch_id<>? AND state IN ('active','paused')", (self.run_id, source['branch_id'])).fetchall():
                self.db.execute("UPDATE g_branches SET state='archived',revision=revision+1 WHERE run_id=? AND branch_id=?", (self.run_id, row[0]))
                self.event('G15', 'branch', row[0], 'GBranch', self.branch(row[0]))
            self.db.execute("UPDATE g_branches SET state='active',revision=revision+1 WHERE run_id=? AND branch_id=?", (self.run_id, source['branch_id']))
            for old in prior_heads:
                self.db.execute('UPDATE g_heads SET solution_id=?,experiment_id=?,revision=revision+1 WHERE run_id=? AND branch_id=? AND researcher_id=?',
                                (solution_id, p['source_experiment_id'], self.run_id, source['branch_id'], old['researcher_id']))
            record['seed_heads'] = self.heads(source['branch_id'])
            record['started_event_seq'] = self.next_seq()
            self.event('G15', 'branch', source['branch_id'], 'GRefinement', record)
            self.set_metadata_tx('refinement', record)
            self.event('G15', 'branch', source['branch_id'], 'GBranch', self.branch(source['branch_id']))
            return {'refinement': record, 'branch': self.branch(source['branch_id']), 'graph_revision': 0}
        result = self.commit('G15', p, key, mutate)
        if getattr(self.controller, 'local_search', None):
            self.controller.local_search.series.clear()
        if hasattr(self.controller, 'no_fit_branches'):
            self.controller.no_fit_branches.clear()
        self.apply_phase_caps()
        self.log.event('search.phase_changed', phase='refine', branch_id=source['branch_id'], source_experiment_id=p['source_experiment_id'],
                       reason=p['reason'], elapsed_seconds=self.elapsed()/1000, remaining_seconds=self.budget.remaining(), phase_limits=record['phase_limits'])
        return result
