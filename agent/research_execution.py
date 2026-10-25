"""Bounded research workers and typed, renewable recovery of individual actions."""
from concurrent.futures import Future
from contextvars import copy_context
from copy import deepcopy
import hashlib
import threading
import time

from agent.contracts import canonical


class ResearchExecutor:
    """Only a bounded number of daemon workers; shutdown never joins a stuck API.

    Gateway cancellation fences late answers. Unlike ThreadPoolExecutor these
    threads are not registered with Python's unbounded interpreter-exit join.
    """
    def __init__(self, max_workers, thread_name_prefix='researcher'):
        self.slots = threading.BoundedSemaphore(max_workers)
        self.futures = set()
        self.closed = False
        self.prefix = thread_name_prefix
        self.lock = threading.Lock()

    def submit(self, function, *args):
        if self.closed:raise RuntimeError('Research executor closed')
        if not self.slots.acquire(blocking=False):raise RuntimeError('Research executor has no free slot')
        future = Future()
        with self.lock:self.futures.add(future)
        def run():
            try:
                if not future.set_running_or_notify_cancel():return
                try:
                    value, error = function(*args), None
                except BaseException as exc:
                    value, error = None, exc
                # Publish readiness only after the execution slot is reusable.
                self.slots.release()
                with self.lock:self.futures.discard(future)
                if error is not None:future.set_exception(error)
                else:future.set_result(value)
                return
            finally:
                if future.cancelled():
                    self.slots.release()
                    with self.lock:self.futures.discard(future)
        threading.Thread(target=copy_context().run, args=(run,), name=self.prefix, daemon=True).start()
        return future

    def shutdown(self, wait=False, cancel_futures=True):
        self.closed = True
        if cancel_futures:
            with self.lock:
                for future in self.futures:future.cancel()
        # No unbounded join, even if a caller accidentally requests wait=True.


class PeerRecovery:
    """An explicit stop awaits evidence; checkpoints/errors resume on a timer.

    Failed input signatures are quarantined independently of the peer. Changing
    code/input/error or receiving new data does not inherit another action's stop.
    """
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.peers, self.actions = {}, {}

    def ready(self, peer, generation):
        state = self.peers.get(peer)
        if not state:return True
        if state['generation'] != generation:return True
        return state['kind'] != 'explicit_stop' and self.clock() >= state['retry_at']

    def defer(self, peer, kind, generation, signature='', reason=''):
        previous = self.peers.get(peer, {})
        count = previous.get('repetitions', 0)+1 if previous.get('signature') == signature else 1
        delay = 0 if kind == 'continuation' else min(120, (2 if kind == 'checkpoint' else 5)*2**min(count-1, 5))
        state = {'researcher_id': peer, 'kind': kind, 'generation': generation, 'signature': signature,
                 'repetitions': count, 'retry_at': self.clock()+delay, 'reason': reason[:600]}
        self.peers[peer] = state
        return deepcopy(state)

    def failure(self, peer, error, generation, input_signature=''):
        code = getattr(error, 'code', type(error).__name__)
        kind = ('provider_pause' if code in {'LLM_UNAVAILABLE','LLM_RATE_LIMIT','LLM_COOLDOWN','LLM_BUDGET_EXHAUSTED',
                   'LLM_PROVIDER_CREDITS_EXHAUSTED'} else 'continuation' if code in {
                   'LLM_OPERATION_TIMEOUT','CONTEXT_TOO_LARGE','LLM_OUTPUT_TRUNCATED','SEARCH_PHASE_BUDGET_EXHAUSTED'} else 'repair')
        signature = hashlib.sha256(canonical([code, input_signature, str(error)[:600]]).encode()).hexdigest()
        return self.defer(peer, kind, generation, signature, str(error))

    def quarantine(self, source, inputs, error):
        signature = hashlib.sha256(canonical([source, inputs, error]).encode()).hexdigest()
        state = self.actions.setdefault(signature, {'count': 0, 'retry_at': 0})
        state['count'] += 1
        state['retry_at'] = self.clock()+min(300, 15*2**min(state['count']-1, 5))
        return signature, deepcopy(state)
