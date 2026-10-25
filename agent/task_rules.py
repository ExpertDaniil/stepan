"""Small rules for explicit task declarations, never guesses from feature names."""
import re

from agent.errors import AgentError


def plain_lines(text):
    """Keep meaning and original evidence while removing Markdown decoration."""
    for raw in text.splitlines():
        line = re.sub(r'\*\*|__|`', '', raw).strip()
        line = re.sub(r'(?<!\w)[*_](?=\w)|(?<=\w)[*_](?!\w)', '', line)
        line = re.sub(r'^[\s#>*|+-]+', '', line).strip()
        line = re.sub(r'\s*\|\s*', ': ', line).strip(': ')
        yield raw.strip(), line


def metric_text(text):
    lines = [line for _, line in plain_lines(text) if line]
    result = []
    label = r'(?:основная\s+метрика|метрика|primary\s+metric|metric)'
    for index, line in enumerate(lines):
        if re.match(label+r'\s*:', line, re.I):
            result.append(line)
        elif re.fullmatch(label, line, re.I) and index+1 < len(lines):
            result.append(lines[index+1])
    return '\n'.join(result)


METRIC_PATTERNS = {'rmse': r'\brmse\b', 'mae': r'\bmae\b', 'r2': r'\br2\b|\br\^?2\b|\br-squared\b',
    'f1_macro': r'macro[ _-]*f1|f1[ _-]*macro|average\s*=\s*["\']macro', 'f1_weighted': r'weighted[ _-]*f1|f1[ _-]*weighted',
    'accuracy': r'\baccuracy\b', 'log_loss': r'log[ _-]*loss', 'roc_auc': r'roc[ _-]*auc'}


def _mentioned(text):
    found = []
    for metric, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text, re.I)
        if match:
            found.append((match.start(), metric))
    return [metric for _, metric in sorted(found)]


def infer_metric(text):
    """The single primary metric declared in prose, or None when it stays ambiguous.

    Precedence: an explicit "metric:" label, a line that calls a metric primary or
    main, a line about the metric that names exactly one, then the whole text when
    it names exactly one. Several metrics without a primary marker stay unresolved,
    and so do explicit labels that each name a different single metric.
    """
    labels = [_mentioned(line) for line in metric_text(text).splitlines()]
    decisive = {found[0] for found in labels if len(found) == 1}
    if len(decisive) > 1:
        return None
    labelled = [metric for found in labels for metric in found]
    if len(set(labelled)) == 1:
        return labelled[0]
    lines = [line for _, line in plain_lines(text) if line]
    for line in lines:
        # The metric nearest to "primary/main" in the first such line is primary:
        # "Macro F1 выбрана основной, потому что Accuracy..." names Macro F1.
        marker = re.search(r'основн|главн|primary|main\b', line, re.I)
        if not marker:
            continue
        spans = [(min(abs(m.start()-marker.start()), abs(m.end()-marker.start())), metric)
                 for metric, pattern in METRIC_PATTERNS.items() for m in re.finditer(pattern, line, re.I)]
        if spans:
            return min(spans)[1]
    about = [_mentioned(line) for line in lines if re.search(r'метрик|metric', line, re.I)]
    about = [found for found in about if found]
    if about and len(set(about[0])) == 1:
        return about[0][0]
    overall = _mentioned(text)
    return overall[0] if len(set(overall)) == 1 else None


def validation_requirement(text, columns, dictionary=()):
    """Resolve only declared deployment and declared column roles.

    A timestamp or a repeated value alone cannot impose a validation strategy.
    Unresolved explicit constraints survive provider failure instead of becoming
    an unrelated random split.
    """
    lines = list(plain_lines(text))
    affirmative=[(raw,line) for raw,line in lines if not re.search(r'\bnot\b|\bне\b',line,re.I)]
    future = [(raw, line) for raw, line in affirmative if re.search(
        r'\bfuture (?:observations|periods?|data)\b|\bnext (?:period|year)\b|'
        r'будущ\w* (?:наблюден|период|данн)|следующ\w* (?:период|год)', line, re.I)]
    unseen = [(raw, line) for raw, line in affirmative if re.search(
        r'\b(?:new|unseen) (?:entities|objects|groups)\b|нов\w* (?:объект|групп)|'
        r'ранее не (?:встречавш|наблюдавш)', line, re.I)]
    explicit = [(raw, line) for raw, line in lines if re.match(
        r'(?:validation|проверка качества|схема валидации)\s*:', line, re.I)]
    modes = set()
    for _, line in explicit:
        modes.update(re.findall(r'\b(?:temporal_holdout|group_holdout|random_holdout|stratified_holdout)\b', line))
    if len(modes) > 1 or (future and unseen):
        raise AgentError('VALIDATION_SCENARIO_UNRESOLVED', 'Conflicting or combined deployment constraints need an explicit supported protocol')
    mode = next(iter(modes), 'temporal_holdout' if future else 'group_holdout' if unseen else None)
    if mode is None:
        return None
    if (future and mode != 'temporal_holdout') or (unseen and mode != 'group_holdout'):
        raise AgentError('VALIDATION_SCENARIO_UNRESOLVED', 'Declared split conflicts with the explicit deployment scenario')
    quote = (explicit or future or unseen)[0][0]
    if mode in {'random_holdout', 'stratified_holdout'}:
        return {'strategy': mode, 'quote': quote, 'column': None, 'deployment': 'random_observations'}
    names = set()
    role = 'time' if mode == 'temporal_holdout' else 'entity'
    label = (r'(?:time|period|timestamp)\s+column|временн\w*\s+столбец|столбец\s+времени' if role == 'time'
             else r'(?:entity|group)\s+column|столбец\s+(?:объекта|группы)')
    for _, line in lines:
        match = re.match(r'(?:'+label+r')\s*:\s*([^,;: ]+)\s*$', line, re.I)
        if match:
            names.add(match[1])
    roles = {'time', 'period', 'timestamp', 'temporal'} if role == 'time' else {'entity', 'group', 'entity_id', 'group_id'}
    names.update(row.get('feature') for row in dictionary if str(row.get('role', '')).casefold() in roles)
    if len(names) > 1 or any(name not in columns for name in names):
        raise AgentError('VALIDATION_SCENARIO_UNRESOLVED', 'Validation column declarations conflict or name an unavailable input')
    return {'strategy': mode, 'quote': quote, 'column': next(iter(names), None),
            'deployment': 'future_observations' if role == 'time' else 'new_entities'}


def bind_requirement(proposal, requirement):
    if requirement is None:
        return proposal
    params = proposal['parameters']
    column = params.get('column')
    if proposal['strategy'] != requirement['strategy'] or (requirement['column'] and column != requirement['column']):
        raise AgentError('INVALID_SPLIT', 'Validation proposal contradicts the explicit deployment contract')
    if requirement['strategy'] in {'temporal_holdout', 'group_holdout'} and not column:
        raise AgentError('VALIDATION_SCENARIO_UNRESOLVED', 'Deployment is known but its validation column is unresolved')
    scenario = {'deployment': requirement['deployment'], 'quote': requirement['quote']}
    if scenario['deployment'] == 'future_observations':
        scenario['time_column'] = column
    elif scenario['deployment'] == 'new_entities':
        scenario['entity_columns'] = [column]
    return {**proposal, 'parameters': {**params, 'scenario': scenario}}
