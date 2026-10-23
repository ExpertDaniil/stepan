"""Check active documentation, local references and real analysis-query examples.

Mermaid fences are checked structurally. For a full external render, install
Mermaid CLI in a separate development environment and pass --mermaid-cli PATH.
This script never installs packages or changes task runtime dependencies.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.contracts import Schemas
from tools.notebooks import NOTEBOOK_SCHEMA


ACTIVE = [ROOT / 'README.md', *(ROOT / 'docs' / name for name in (
    'PROJECT_GUIDE.md', 'ARCHITECTURE_REVIEW.md', 'IMPLEMENTATION.md',
    'GRAPH.md', 'LOGGING.md', 'OPEN-RESEARCH.md', 'HISTORY.md',
    'PLATFORM.md', 'research/notebooks/README.md', 'KNOWLEDGE.md',
    'RESEARCH-DECISION-FIXES.md', 'CONTEXT-TOOL-FIXES.md', 'VALIDATION-ARCHITECTURE.md'))]


def heading_ids(body):
    result, counts = set(), {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', body, re.M):
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        duplicate = counts.get(slug, 0)
        counts[slug] = duplicate + 1
        result.add(slug + ('-' + str(duplicate) if duplicate else ''))
    return result


def check(mermaid_cli=None):
    schemas, errors, diagrams = Schemas(), [], []
    links = examples = notebook_examples = 0
    for document in ACTIVE:
        body = document.read_text(encoding='utf-8')
        fences = list(re.finditer(r'^```([^\n]*)\n(.*?)^```\s*$', body, re.M | re.S))
        without_code = re.sub(r'^```[^\n]*\n.*?^```\s*$', '', body, flags=re.M | re.S)
        if re.search(r'^```', without_code, re.M):
            errors.append(f'{document.relative_to(ROOT)}: unclosed code fence')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', without_code):
            parsed = urlsplit(target.strip('<>'))
            if parsed.scheme or parsed.netloc:
                continue
            links += 1
            path = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
            if not path.is_relative_to(ROOT) or not path.exists():
                errors.append(f'{document.relative_to(ROOT)}: missing local link {target}')
            elif parsed.fragment and path.suffix == '.md':
                if unquote(parsed.fragment) not in heading_ids(path.read_text(encoding='utf-8')):
                    errors.append(f'{document.relative_to(ROOT)}: missing heading {target}')
        for match in fences:
            language, content = match.group(1).strip(), match.group(2).strip()
            if language == 'mermaid':
                if not re.match(r'^(flowchart\s+(TD|TB|LR|RL|BT)|sequenceDiagram|stateDiagram-v2)\b', content):
                    errors.append(f'{document.relative_to(ROOT)}: unsupported Mermaid declaration')
                diagrams.append((document.name, content))
            elif language == 'json' and document.name == 'PROJECT_GUIDE.md':
                try:
                    value = json.loads(content)
                    if isinstance(value, dict) and value.get('kind') in {'search', 'read'}:
                        schemas._check(value, NOTEBOOK_SCHEMA, '$.notebook')
                        permitted = ({'kind', 'keywords', 'offset', 'limit'} if value['kind'] == 'search'
                                     else {'kind', 'approach_id', 'cell_index', 'offset', 'max_chars'})
                        if set(value) - permitted:
                            raise ValueError('Notebook query mixes search/read fields')
                        if value['kind'] == 'read' and not {'approach_id', 'cell_index'} <= set(value):
                            raise ValueError('Notebook read needs approach_id and cell_index')
                        notebook_examples += 1
                    else:
                        schemas.validate(value, 'AnalysisQuery')
                        examples += 1
                except Exception as exc:
                    errors.append(f'{document.name}: invalid query example: {exc}')
    if mermaid_cli:
        with tempfile.TemporaryDirectory(prefix='stepan-doc-diagrams-') as folder:
            for number, (name, source) in enumerate(diagrams, 1):
                incoming = Path(folder) / f'{number}.mmd'
                output = incoming.with_suffix('.svg')
                incoming.write_text(source, encoding='utf-8')
                try:
                    subprocess.run([str(mermaid_cli), '-i', str(incoming), '-o', str(output)],
                                   check=True, capture_output=True, text=True, timeout=60)
                    if not output.is_file() or '<svg' not in output.read_text(encoding='utf-8'):
                        raise ValueError('renderer did not produce SVG')
                except Exception as exc:
                    errors.append(f'{name}: Mermaid render {number}: {exc}')
    return {'documents': len(ACTIVE), 'local_links': links, 'analysis_examples': examples,
            'notebook_examples': notebook_examples,
            'mermaid_blocks': len(diagrams), 'full_mermaid_render': bool(mermaid_cli), 'errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mermaid-cli', type=Path,
                        help='Optional externally installed mmdc executable; no implicit installation')
    result = check(parser.parse_args().mermaid_cli)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['errors']))
