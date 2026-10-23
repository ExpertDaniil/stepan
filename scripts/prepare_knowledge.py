"""Build the offline library from the supplied AutoMind archive. No network/LLM calls.

The allowlist is an explicit initial tabular selection, not an inference from the
upstream noisy task tags. All other records retain an exclusion reason.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REGRESSION = {5, 54, 59, 86, 104, 138, 151, 158, 162, 163, 166, 174, 180,
              190, 222, 224, 239, 245, 248, 252, 297, 300, 313, 335, 346, 348,
              349, 363, 387, 398, 446}
MULTICLASS = {75, 78, 100, 107, 207, 214, 229, 309, 354, 368, 396}
GENERAL_TRANSFER = {2}  # Binary source; encoding methods transfer, its AUC results do not.
STRUCTURED = {5, 104, 138, 190, 245, 297, 300, 335, 346, 349, 354, 368}
TOPICS = {
    'constant_features': r'constant|zero.variance|zero.standard',
    'missing': r'missing|imput|nan\b',
    'categorical': r'categorical|one.hot|ordinal.encod|catboost',
    'frequency_encoding': r'frequency.encod|count.encod|value_counts',
    'rare_categories': r'rare|unseen|high.cardinal|unknown.categor',
    'log_features': r'log1p|log\(x\s*\+\s*1|log\(.*features',
    'target_transform': r'target.transform|log\(loss|log\(target|box.cox|log1p',
    'interactions': r'interaction|feature.combination',
    'aggregates': r'groupby|aggregation|aggregate|row.wise|row.sum|row.mean',
    'temporal': r'rolling|time.series|lag.feature|day.of.week|weekend',
    'validation': r'cross.validation|cross.validate|k.fold|groupkfold|validation',
    'leakage': r'leak|target.encod',
    'imbalance': r'imbalanc|class.weight|weighted.loss',
    'error_analysis': r'residual|outlier|error.analysis',
    'linear': r'logistic|ridge|linear.model',
    'boosting': r'xgboost|lightgbm|catboost|gradient.boost',
    'ensemble': r'blend|ensemble|stacking|stacked',
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def build(archive_path, destination):
    with ZipFile(archive_path) as archive:
        prefix = 'automind-kaggle-knowledge/'
        original = archive.read(prefix+'tagged_competition_solutions.jsonl')
        upstream = json.loads(archive.read(prefix+'source_manifest.json'))
        if upstream['source_sha256'] != 'cf4e5182d6db75c6940d40a4936fbf2b7d93647f79d7b4bba8ff924dbef56512':
            raise ValueError('The reviewed record allowlist applies only to the pinned AutoMind corpus')
        if hashlib.sha256(original).hexdigest() != upstream['source_sha256']:
            raise ValueError('Archive source hash does not match its manifest')
        competitions = [json.loads(line) for line in original.splitlines() if line.strip()]
        license_text = archive.read(prefix+'UPSTREAM_LICENSE.txt').decode('utf-8')
    destination.mkdir(parents=True, exist_ok=True)
    selection, accepted, posts = [], [], []
    seen = set()
    for number, competition in enumerate(competitions, 1):
        kinds = (['regression'] if number in REGRESSION else
                 ['multiclass_classification'] if number in MULTICLASS else
                 ['regression', 'multiclass_classification'] if number in GENERAL_TRANSFER else [])
        scope = 'general_method_transfer' if number in GENERAL_TRANSFER else 'tabular_with_structure' if number in STRUCTURED else 'tabular'
        decision = {'source_record': number, 'title': competition['title'], 'accepted': bool(kinds),
                    'task_kinds': kinds, 'scope': scope if kinds else None,
                    'reason': 'explicit_initial_tabular_selection' if kinds else 'outside_initial_reviewed_scope',
                    'original_tags': competition['tags']}
        selection.append(decision)
        if not kinds:
            continue
        intro = competition['introduction']
        compact = re.sub(r'\s+', ' ', intro).lower()
        metric = ('rmsle' if 'root mean squared logarithmic' in compact else
                  'log_loss' if 'logarithmic loss' in compact else
                  'accuracy' if 'accuracy' in compact else
                  'mae' if 'mean absolute error' in compact else
                  'rmse' if 'root mean squared error' in compact else
                  'r2' if 'coefficient of determination' in compact else 'unknown')
        if number in GENERAL_TRANSFER:
            metric = 'binary_auc'
        accepted.append({**decision, 'url': competition['url'], 'description': intro,
                         'source_metric': metric})
        for title, text in competition['solutions'].items():
            if not isinstance(text, str) or not text.strip():
                continue
            fingerprint = hashlib.sha256((upstream['commit']+'\n'+competition['url']+'\n'+title+'\n'+text).encode()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            links = list(dict.fromkeys(u.rstrip(').,;') for u in re.findall(r'https?://[^\s<>]+', text)))
            tags = [tag for tag, pattern in TOPICS.items() if re.search(pattern, text, re.I)]
            posts.append({'id': 'kg-'+fingerprint[:24], 'source_record': number, 'title': title,
                          'task_kinds': kinds, 'tags': tags, 'text': text, 'links': links,
                          'status': 'link_stub' if len(text.strip()) < 300 and links else 'source_writeup_unverified',
                          'code_status': 'code_like_text_only' if re.search(r'(?m)^\s*(?:import |from \w[\w.]* import |def \w+\()', text) else 'no_extracted_implementation',
                          'source_sha256': hashlib.sha256(text.encode()).hexdigest()})
    write_json(destination/'competitions.json', accepted)
    write_json(destination/'selection.json', selection)
    with (destination/'sources.jsonl').open('w', encoding='utf-8') as stream:
        for post in posts:
            stream.write(json.dumps(post, ensure_ascii=False, allow_nan=False)+'\n')
    # Methods and examples are owned guidance, never attributed to upstream authors.
    from tools.knowledge_content import method_cards
    cards = method_cards()
    for card in cards:
        matches = [p for p in posts if set(card['tags']) & set(p['tags']) and p['status'] != 'link_stub']
        matches.sort(key=lambda p: (-len(set(card['tags']) & set(p['tags'])), p['source_record'], p['id']))
        records = set(); references = []
        for post in matches:
            if post['source_record'] not in records:
                references.append(post['id']);records.add(post['source_record'])
            if len(references) == 4:
                break
        card['related_source_ids'] = references
    write_json(destination/'cards.json', cards)
    (destination/'UPSTREAM_LICENSE.txt').write_text(license_text, encoding='utf-8')
    names = ['competitions.json', 'selection.json', 'sources.jsonl', 'cards.json', 'UPSTREAM_LICENSE.txt']
    manifest = {'version': 'knowledge-v1', 'upstream': upstream,
                'selection_policy': 'Explicit reviewed task/modality scope; original tags are not authoritative. Not every post has been fact-checked.',
                'counts': {'competitions': len(accepted), 'posts': len(posts), 'methods': len(cards)},
                'files': {name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in names},
                'rights_note': 'Upstream repository MIT notice retained. It does not establish separate rights to third-party write-ups. Implementation examples are project-authored, not recovered Kaggle notebooks.',
                'verification': 'No local execution of the example pipelines was performed.'}
    write_json(destination/'manifest.json', manifest)
    return manifest['counts']


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'recipes/knowledge')
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output)))
