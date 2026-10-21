from agent.errors import AgentError


def check(payload):
    constraints = payload["constraints"]
    required = set(constraints["required_capability_ids"])
    forbidden = set(constraints["forbidden_capability_ids"])
    actual = set(payload["effective_capability_ids"])
    if required & forbidden:
        raise AgentError("CONSTRAINT_CONFLICT", "A capability cannot be required and forbidden")
    if not required <= actual or actual & forbidden:
        raise AgentError("CAPABILITY_FORBIDDEN" if actual & forbidden else "REQUIRED_CAPABILITY_MISSING",
                         "Candidate violates branch restrictions",
                         details={"missing": sorted(required-actual), "forbidden": sorted(forbidden & actual)})
    return {"valid": True, "satisfied_required_capability_ids": sorted(required)}
