"""Collect public Hugging Face repository metadata and inspect tokenizer JSONs.

Run discovery first, then measure and history. No weights or repository code run.
Bulk observations and cached artifacts belong in the ignored workspace directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import re
from pathlib import Path
import tempfile
import threading
import time

from llm_tokenizer_statistics import measure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / 'workspace/hf-tokenizer-catalog'
QUERIES = [('text-generation', 10000), ('image-text-to-text', 1000),
           ('text2text-generation', 500), ('any-to-any', 200)]
FIELDS = ['createdAt', 'sha', 'downloads', 'likes', 'pipeline_tag', 'library_name',
          'tags', 'config', 'cardData', 'safetensors', 'gated', 'siblings', 'gguf']
LOCAL = threading.local()
LOCK = threading.Lock()
BLOB_LOCKS = {}


def session():
    import requests

    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers['User-Agent'] = 'llm-course-tokenizer-catalog/1.0'
    return LOCAL.session


def request(url, method='GET', **kwargs):
    import requests

    for attempt in range(5):
        try:
            response = session().request(method, url, timeout=(15, 90), **kwargs)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            reset = re.search(r't=(\d+)', response.headers.get('RateLimit', ''))
            wait = int(response.headers.get('Retry-After', reset.group(1) if reset else '60'))
            print(json.dumps({'rate_limit_wait_seconds': wait, 'url': url}), flush=True)
            time.sleep(wait + 1)
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return response
    raise RuntimeError(f'Repeated HTTP failure: {url}')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as temporary:
        temporary.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    Path(temporary.name).replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def discover(cache):
    models = {}
    query_summary = []
    for task, limit in QUERIES:
        count, page = 0, 0
        url = 'https://huggingface.co/api/models'
        params = {'pipeline_tag': task, 'sort': 'downloads', 'direction': -1,
                  'limit': min(1000, limit), 'expand': FIELDS}
        while url and count < limit:
            path = cache / 'pages' / f'{task}-{page:03d}.json.gz'
            if path.exists():
                document = json.loads(gzip.decompress(path.read_bytes()))
            else:
                response = request(url, params=params)
                response.raise_for_status()
                document = {'request_url': response.url, 'collected_at': datetime.now(timezone.utc).isoformat(),
                            'next_url': response.links.get('next', {}).get('url'), 'models': response.json()}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps(document).encode()))
            for raw in document['models']:
                if count >= limit:
                    break
                count += 1
                model = dict(raw)
                # The chat template is not required for registry discovery.
                model['config'] = {k: v for k, v in model.get('config', {}).items() if k != 'tokenizer_config'}
                model['selection_task'] = task
                model['selection_rank_by_downloads'] = count
                model['collected_at'] = document['collected_at']
                models.setdefault(model['id'], model)
            print(json.dumps({'task': task, 'collected': count, 'distinct_total': len(models)}), flush=True)
            url, params, page = document['next_url'], None, page + 1
        query_summary.append({'pipeline_tag': task, 'requested_limit': limit, 'returned': count})
    query_distinct_repositories = len(models)
    # Keep the original curated examples even if their task/rank falls outside discovery.
    curated = read_json(ROOT / 'docs/model-tokenizer-statistics/models.json')
    supplement_errors = []
    for row in curated:
        if row['weights_access'] != 'open_weights' or row['model_id'] in models:
            continue
        path = cache / 'supplements' / (row['model_id'].replace('/', '--') + '.json')
        model = read_json(path)
        if not model:
            response = request('https://huggingface.co/api/models/' + row['model_id'],
                               params={'expand': FIELDS})
            if response.ok:
                model = response.json()
            else:
                supplement_errors.append({'model_id': row['model_id'], 'http_status': response.status_code})
        if model:
            model['config'] = {k: v for k, v in model.get('config', {}).items() if k != 'tokenizer_config'}
            model.update(selection_task='curated_supplement', selection_rank_by_downloads=None,
                         collected_at=model.get('collected_at') or datetime.now(timezone.utc).isoformat())
            save_json(path, model)
            models[model['id']] = model
    output = sorted(models.values(), key=lambda row: (-(row.get('downloads') or 0), row['id']))
    save_json(cache / 'discovery.json', output)
    save_json(cache / 'discovery-summary.json', {'queries': query_summary, 'distinct_repositories': len(output),
              'distinct_query_repositories': query_distinct_repositories,
              'curated_supplement_repositories': len(output) - query_distinct_repositories,
              'sampling': 'Top current 30-day downloads per listed task, plus the earlier curated open-weight sample.',
              'curated_supplement_errors': supplement_errors})


def blob_lock(key):
    with LOCK:
        return BLOB_LOCKS.setdefault(key, threading.Lock())


def inspect_tokenizer(model, cache):
    repo, revision = model['id'], model.get('sha')
    result_path = cache / 'measurements' / (repo.replace('/', '--') + '.json')
    if result_path.exists():
        return read_json(result_path)
    result = {'model_id': repo, 'revision': revision, 'status': 'not_inspected'}
    try:
        names = {item['rfilename'] for item in model.get('siblings', [])}
        if not revision:
            result['status'] = 'missing_revision'
        elif 'tokenizer.json' not in names:
            result['status'] = 'no_root_tokenizer_json'
        elif model.get('gated'):
            result['status'] = 'gated'
        else:
            url = f'https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json'
            head = request(url, method='HEAD', allow_redirects=False)
            if head.status_code not in (200, 302, 307):
                result.update(status=f'http_{head.status_code}', source_url=url)
            else:
                etag = (head.headers.get('X-Linked-ETag') or head.headers.get('ETag') or '').strip('"')
                if len(etag) not in (40, 64) or any(c not in '0123456789abcdef' for c in etag):
                    raise ValueError('Missing or unsupported content identifier')
                # Exact file bytes are shared only when the Hub content ID agrees.
                folder = cache / 'tokenizers' / etag
                primary = folder / 'tokenizer.json'
                with blob_lock(etag):
                    if not primary.exists():
                        response = request(url)
                        response.raise_for_status()
                        payload = response.content
                        if len(payload) > 100_000_000:
                            raise ValueError('Tokenizer file exceeds 100MB collection limit')
                        digest = hashlib.sha256(payload).hexdigest()
                        if len(etag) == 64 and digest != etag:
                            raise ValueError('LFS checksum mismatch')
                        if len(etag) == 40:
                            git_digest = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
                            if git_digest != etag:
                                raise ValueError('Git blob checksum mismatch')
                        folder.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as temporary:
                            temporary.write(payload)
                        Path(temporary.name).replace(primary)
                    stats_path = folder / 'statistics.json'
                    if stats_path.exists():
                        measured = read_json(stats_path)
                    else:
                        measured = measure(folder)
                        measured['sha256'] = hashlib.sha256(primary.read_bytes()).hexdigest()
                        measured['bytes'] = primary.stat().st_size
                        save_json(stats_path, measured)
                result.update(status='measured_tokenizer_json', source_url=url,
                              local_path=str(folder), tokenizer_group_sha256=measured['sha256'],
                              count_scope='serialized_tokenizer_json_only', **measured)
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    save_json(result_path, result)
    return result


def measure_catalog(cache, limit, workers):
    models = read_json(cache / 'discovery.json')
    # Presence of the actual root artifact determines eligibility, not inferred lineage.
    eligible = [m for m in models if 'tokenizer.json' in {s['rfilename'] for s in m.get('siblings', [])} and not m.get('gated')]
    selected = eligible[:limit] if limit else eligible
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(inspect_tokenizer, model, cache) for model in selected]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            counts[result['status']] += 1
            if number % 50 == 0 or number == len(selected):
                print(json.dumps({'processed': number, 'selected': len(selected), 'statuses': dict(counts)}), flush=True)
    save_json(cache / 'measurement-summary.json', {'eligible': len(eligible), 'attempted': len(selected), 'statuses': dict(counts),
              'selection': 'Descending 30-day downloads among ungated repositories with a root tokenizer.json.'})


def vocabulary_fingerprint(document):
    """Hash the token-to-ID mapping, independently of JSON formatting/merges."""
    serialized = document['model']['vocab']
    mapping = ({int(index): token for token, index in serialized.items()}
               if isinstance(serialized, dict)
               else {index: item[0] for index, item in enumerate(serialized)})
    if len(mapping) != len(serialized):
        raise ValueError('Conflicting base-vocabulary token IDs')
    for token in document.get('added_tokens', []):
        index = int(token['id'])
        if index in mapping and mapping[index] != token['content']:
            raise ValueError(f'Conflicting token at ID {index}')
        mapping[index] = token['content']
    payload = json.dumps(sorted(mapping.items()), ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprint_catalog(cache):
    by_artifact = {}
    for index, path in enumerate((cache / 'tokenizers').glob('*/statistics.json'), 1):
        stats = read_json(path)
        if 'vocabulary_sha256' not in stats:
            stats['vocabulary_sha256'] = vocabulary_fingerprint(read_json(path.with_name('tokenizer.json')))
            save_json(path, stats)
        by_artifact[stats['sha256']] = stats['vocabulary_sha256']
        if index % 200 == 0:
            print(json.dumps({'vocabularies_fingerprinted': index}), flush=True)
    for path in (cache / 'measurements').glob('*.json'):
        result = read_json(path)
        if result.get('sha256') in by_artifact:
            result['vocabulary_sha256'] = by_artifact[result['sha256']]
            save_json(path, result)
    print(json.dumps({'artifact_files': len(by_artifact), 'unique_token_to_id_mappings': len(set(by_artifact.values()))}), flush=True)


def repo_history(model, cache):
    repo = model['id']
    path = cache / 'histories' / (repo.replace('/', '--') + '.json')
    if path.exists():
        return read_json(path)
    commits = []
    url = f'https://huggingface.co/api/models/{repo}/commits/{model.get("sha") or "main"}'
    try:
        pages = 0
        while url:
            response = request(url)
            response.raise_for_status()
            commits.extend(response.json())
            url = response.links.get('next', {}).get('url')
            pages += 1
            if pages >= 50 and url:
                raise ValueError('History exceeds 50 pages; earliest commit remains unverified')
        earliest = min(commits, key=lambda item: item['date'])
        result = {'model_id': repo, 'status': 'complete', 'commit_count': len(commits), 'earliest_commit': earliest,
                  'source_url': f'https://huggingface.co/{repo}/commit/{earliest["id"]}', 'commits': commits}
    except Exception as exc:
        result = {'model_id': repo, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    save_json(path, result)
    return result


def history_catalog(cache, workers):
    models = read_json(cache / 'discovery.json')
    curated_ids = {row['model_id'] for row in read_json(ROOT / 'docs/model-tokenizer-statistics/models.json')}
    selected = [model for model in models if model['id'] in curated_ids
                or (model.get('createdAt') or '').startswith('2022-03-02')]
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(repo_history, model, cache) for model in selected]
        for number, future in enumerate(as_completed(futures), 1):
            counts[future.result()['status']] += 1
            if number % 20 == 0 or number == len(selected):
                print(json.dumps({'histories_processed': number, 'selected': len(selected), 'statuses': dict(counts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['discover', 'measure', 'history', 'fingerprint'])
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--limit', type=int, default=0, help='Eligible repositories to inspect; zero means all eligible')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'discover':
        discover(args.cache_dir)
    elif args.command == 'measure':
        measure_catalog(args.cache_dir, args.limit, args.workers)
    elif args.command == 'fingerprint':
        fingerprint_catalog(args.cache_dir)
    else:
        history_catalog(args.cache_dir, args.workers)


if __name__ == '__main__':
    main()
