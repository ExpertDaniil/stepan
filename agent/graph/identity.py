import hashlib

from agent.contracts import canonical
from agent.errors import AgentError


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def solution(payload):
    sha = digest(payload["core"])
    return {"solution_id": "sol-" + sha, "solution_hash": sha, "canonicalization": "pyjson-v1"}


def component(value):
    result = dict(value)
    result.pop("component_id", None)
    return {**result, "component_id": "cmp-" + digest(result)}


def dependency_closure(components):
    by_id = {c["component_id"]: c for c in components}
    if len(by_id) != len(components):
        raise AgentError("MANIFEST_MISMATCH", "Duplicate component IDs")
    visiting, done, capabilities = set(), set(), set()
    def visit(ident):
        if ident in visiting:
            raise AgentError("COMPONENT_DEPENDENCY_CYCLE", "Component dependency cycle")
        if ident in done:
            return
        if ident not in by_id:
            raise AgentError("COMPONENT_DEPENDENCY_MISSING", ident)
        visiting.add(ident)
        for dep in by_id[ident]["dependency_component_ids"]:
            visit(dep)
        visiting.remove(ident)
        done.add(ident)
        capabilities.add(by_id[ident]["capability_id"])
    for ident in by_id:
        visit(ident)
    return sorted(capabilities), sorted(done)
