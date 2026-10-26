class AgentError(Exception):
    def __init__(self, code, message, *, retryable=False, details=None, retry_after_seconds=None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}
        self.retry_after_seconds = retry_after_seconds

    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable,
                "retry_after_seconds": self.retry_after_seconds, "details": self.details}


# Failures of the execution environment say nothing about a candidate's
# hypothesis; resource limits measure cost, not predictive usefulness.
ENVIRONMENT_FAILURE_CODES = frozenset({'GUARD_ENVIRONMENT_BLOCKED', 'RESOURCE_BUSY', 'PROCESS_START_FAILED', 'STALE_LEASE'})
RESOURCE_FAILURE_CODES = frozenset({'JOB_TIMEOUT', 'JOB_MEMORY_LIMIT', 'PREFLIGHT_TIMEOUT'})


def failure_scope(error):
    """Classify a stored error (code, dict or JSON text) without guessing from prose."""
    import json
    code = error
    if isinstance(error, str) and error.startswith('{'):
        try:
            code = json.loads(error)
        except ValueError:
            code = None
    elif isinstance(error, str) and not error.replace('_', '').isalpha():
        code = None  # Prose (for example a str() of an exception) carries no code.
    if isinstance(code, dict):
        code = code.get('code')
    if not isinstance(code, str):
        return 'unknown'
    return ('execution_environment' if code in ENVIRONMENT_FAILURE_CODES else
            'resource_limit' if code in RESOURCE_FAILURE_CODES else 'candidate')
