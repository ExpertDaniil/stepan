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
