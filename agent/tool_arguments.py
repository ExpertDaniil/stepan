"""Canonical tool operations, shared by transcript cache and service execution."""
from copy import deepcopy


def normalize_tool_arguments(name, args):
    if not isinstance(args, dict):
        return args
    value = deepcopy(args)
    if name == 'search_knowledge':
        ids = value.get('ids', [])
        limit = value.get('max_results', 3)
        known = {'query', 'ids', 'tags', 'offset', 'max_results'}
        if (not set(value) <= known or not isinstance(ids, list) or len(ids) > 3 or any(not isinstance(i, str) for i in ids)
                or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 3
                or not isinstance(value.get('query', ''), str)
                or not isinstance(value.get('tags', []), list)
                or any(not isinstance(t, str) for t in value.get('tags', []))):
            return value
        if ids:
            # Query/tags only rank searches; they do not change an ID read.
            selected = list(dict.fromkeys(ids))[:limit]
            value.update(ids=selected, query='', tags=[], max_results=len(selected))
        else:
            value.update(query=' '.join(value.get('query', '').casefold().split()), ids=[],
                         tags=sorted(set(value.get('tags', []))), max_results=limit)
        value.setdefault('offset', 0)
    elif name == 'read_solution':
        # Equal reads share one transcript result; defaults mirror read_solution.
        value.setdefault('view', 'pipeline')
        value.setdefault('offset', 0)
        value.setdefault('limit_bytes', 6000)
    elif name == 'read_memory':
        value.setdefault('offset', 0)
        value.setdefault('limit_bytes', 4000)
    elif name == 'analyze_data':
        if value.get('kind') == 'report':
            # Explicit incompatible new-analysis fields remain errors, not silently ignored.
            value.setdefault('columns', [])
            value.setdefault('group_by', None)
            value.setdefault('include_target', False)
            if 'offset' in value or 'limit_bytes' in value:
                value.setdefault('offset', 0)
                value.setdefault('limit_bytes', 4000)
        columns = value.get('columns')
        if isinstance(columns, list) and all(isinstance(c, str) for c in columns):
            value['columns'] = sorted(set(columns))
        if value.get('report_id') is None:
            value.pop('report_id', None)
    return value


def report_read(artifact_id):
    return {'kind': 'report', 'report_id': artifact_id, 'columns': [], 'group_by': None, 'include_target': False}
