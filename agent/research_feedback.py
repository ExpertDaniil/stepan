"""Dispatcher-published feedback; API threads never read SQLite or expand lineage."""
from copy import deepcopy
import hashlib
import threading

from agent.contracts import canonical


class FeedbackMailbox:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.lock = threading.RLock()
        self.views = {}

    def publish(self, scope, value):
        digest = hashlib.sha256(canonical(value).encode()).hexdigest()
        with self.lock:
            old = self.views.get(scope)
            if old and old['revision'] == digest:
                return old
            art = self.artifacts.write_json('research/updates-'+digest+'.json', value, 'research_updates')
            view = {'revision': digest, 'artifact_id': art['artifact_id'], **deepcopy(value)}
            self.views[scope] = view
            return deepcopy(view)

    def latest(self, scope):
        with self.lock:
            return deepcopy(self.views.get(scope))


def compact_memory(memory):
    """Keep recent questions, measured outcomes and retrievable provenance explicit."""
    result = deepcopy(memory)
    decisions = result.get('decisions', [])
    result['omitted_decision_count'] = max(0, len(decisions)-6)
    result['omitted_decision_artifact_ids'] = [d['decision_artifact_id'] for d in decisions[:-6]][-24:]
    result['decisions'] = decisions[-6:]
    ideas = result.get('open_ideas', [])
    result['omitted_idea_count'] = max(0, len(ideas)-24)
    result['open_ideas'] = ideas[-24:]
    for decision in result['decisions']:
        configuration = decision.get('configuration')
        if configuration:
            # Exact configuration remains in the immutable decision artifact.
            core = configuration.pop('fingerprint_core', None)
            if core:
                configuration['fingerprint_sha256'] = hashlib.sha256(canonical(core).encode()).hexdigest()
                configuration['configuration_details'] = 'Read decision_artifact_id; effective base is supplied separately.'
        observations = decision.get('observations', [])
        decision['omitted_observation_count'] = max(0, len(observations)-4)
        decision['observations'] = observations[-4:]
        for series in decision.get('tuning_series', []):
            trials = series.get('trials', [])
            retained = [t for t in trials if t.get('experiment_id') == series.get('best_experiment_id')]
            retained += [t for t in trials[-4:] if t not in retained]
            series['trials'] = retained
            series['omitted_trial_count'] = len(trials)-len(retained)
    return result
