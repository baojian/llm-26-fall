"""Export the broad repository catalog and enrich curated date provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

from collect_hf_tokenizer_catalog import DEFAULT_CACHE, ROOT, read_json, save_json


DATE_CORRECTIONS = {
    'openai-community/openai-gpt': ('2018-06-11', 'https://openai.com/index/language-unsupervised/'),
    'openai/davinci': ('2020-06-11', 'https://openai.com/index/openai-api/'),
    'google/gemini-1.0-pro': ('2023-12-06', 'https://blog.google/technology/ai/google-gemini-ai/'),
}


def earliest_record(model, history):
    """Return a recorded repository timestamp, never a claim of public access."""
    candidates = []
    created = model.get('createdAt')
    # HF did not retain creation dates before this date. Do not use the floor.
    if created and not created.startswith('2022-03-02'):
        candidates.append((created, 'hf_repository_creation', 'https://huggingface.co/api/models/' + model['id']))
    if history.get('status') == 'complete':
        candidates.append((history['earliest_commit']['date'], 'earliest_hf_commit', history['source_url']))
    return min(candidates, key=lambda value: value[0]) if candidates else (None, None, None)


def lineage(model):
    card = model.get('cardData') or {}
    base = card.get('base_model') or []
    parents = set([base] if isinstance(base, str) else base if isinstance(base, list) else [])
    relations = set()
    for tag in model.get('tags') or []:
        if not tag.startswith('base_model:'):
            continue
        pieces = tag.split(':', 2)
        if len(pieces) == 2:
            parents.add(pieces[1])
        else:
            relations.add(pieces[1])
            parents.add(pieces[2])
    relation = card.get('base_model_relation')
    if relation:
        relations.update([relation] if isinstance(relation, str) else relation)
    tags = set(model.get('tags') or [])
    evidence = 'model card base_model / base_model_relation and Hub base_model tags'
    if model.get('library_name') in ('peft', 'adapter-transformers'):
        relations.add('adapter')
    if model.get('library_name') in ('gguf', 'GGUF', 'llama.cpp') or model.get('gguf'):
        relations.add('quantized_or_gguf_conversion')
    if 'mergekit' in tags:
        relations.add('merge')
    return sorted(str(item) for item in parents if item), sorted(relations) or ['not_declared'], evidence


def root_models(identifier, parent_map, trail=()):
    if identifier in trail:
        return [], True
    parents = parent_map.get(identifier, [])
    if not parents:
        return [identifier], False
    result, cycle = set(), False
    for parent in parents:
        roots, found_cycle = root_models(parent, parent_map, trail + (identifier,))
        result.update(roots)
        cycle |= found_cycle
    return sorted(result), cycle


def enrich_curated(rows, models, histories):
    for row in rows:
        identifier = row['model_id']
        if identifier in DATE_CORRECTIONS:
            date, source = DATE_CORRECTIONS[identifier]
            row.update(release_date=date, release_date_precision='day', release_date_source_url=source)
            if identifier == 'openai-community/openai-gpt':
                row['notes'] = 'Historical baseline. Release day follows the dated OpenAI announcement. Corpus size is reported in books/words, not converted to tokens.'
        model = models.get(identifier, {})
        stamp, kind, source = earliest_record(model, histories.get(identifier, {}))
        row.update(hf_repository_created_at=model.get('createdAt'),
                   earliest_repository_record_at=stamp, earliest_repository_record_kind=kind,
                   earliest_repository_record_source_url=source)
        if row['release_date_precision'] == 'day':
            row.update(timeline_date=row['release_date'], timeline_date_basis='documented_release',
                       timeline_date_source_url=row['release_date_source_url'])
        elif stamp:
            row.update(timeline_date=stamp[:10], timeline_date_basis=kind, timeline_date_source_url=source)
        else:
            row.update(timeline_date=None, timeline_date_basis=None, timeline_date_source_url=None)
    return rows


def flatten_for_csv(rows):
    return [{key: json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, (list, dict, bool)) else value
             for key, value in row.items()} for row in rows]


def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(flatten_for_csv(rows))
    return output.getvalue()


def export(cache, update_curated=False):
    models = read_json(cache / 'discovery.json')
    model_map = {model['id']: model for model in models}
    histories = {row['model_id']: row for path in (cache / 'histories').glob('*.json') if (row := read_json(path))}
    measurements = {row['model_id']: row for path in (cache / 'measurements').glob('*.json') if (row := read_json(path))}
    curated = enrich_curated(read_json(ROOT / 'docs/model-tokenizer-statistics/models.json'), model_map, histories)
    curated_map = {row['model_id']: row for row in curated}
    parent_map = {model['id']: lineage(model)[0] for model in models}
    group_members = defaultdict(list)
    vocab_members = defaultdict(list)
    for result in measurements.values():
        if result['status'] == 'measured_tokenizer_json':
            group_members[result['tokenizer_group_sha256']].append(result['model_id'])
            if result.get('vocabulary_sha256'):
                vocab_members[result['vocabulary_sha256']].append(result['model_id'])
    rows = []
    for model in models:
        identifier = model['id']
        original = curated_map.get(identifier, {})
        measurement = measurements.get(identifier, {})
        stamp, kind, source = earliest_record(model, histories.get(identifier, {}))
        parents, relations, evidence = lineage(model)
        roots, cycle = root_models(identifier, parent_map)
        names = {item['rfilename'] for item in model.get('siblings') or []}
        status = measurement.get('status') or ('gated' if model.get('gated') else 'not_yet_inspected' if 'tokenizer.json' in names else 'no_root_tokenizer_json')
        measured = status == 'measured_tokenizer_json'
        scope = 'serialized_tokenizer_json_only' if measured else None
        # Use a directly collected original-format tokenizer when JSON is absent/gated.
        if not measured and original.get('tokenizer_status') == 'downloaded':
            measurement = original
            measured, status, scope = True, 'measured_curated_artifact', original['tokenizer_count_scope']
        local_path = measurement.get('local_path') or measurement.get('tokenizer_local_path')
        if local_path and Path(local_path).is_absolute():
            local_path = 'workspace/hf-tokenizer-catalog/' + str(Path(local_path).resolve().relative_to(cache.resolve()))
        timeline_date = original.get('timeline_date') or (stamp[:10] if stamp else None)
        timeline_basis = original.get('timeline_date_basis') or kind
        group = measurement.get('tokenizer_group_sha256') if status == 'measured_tokenizer_json' else None
        origin_groups = sorted({measurements[parent]['tokenizer_group_sha256'] for parent in parents
                                if measurements.get(parent, {}).get('status') == 'measured_tokenizer_json'})
        config = model.get('config') or {}
        card = model.get('cardData') or {}
        row = {
            'model_id': identifier, 'model_name': original.get('model_name') or identifier.split('/')[-1],
            'developer': identifier.split('/')[0],
            'weights_access': original.get('weights_access', 'hub_repository_unverified'),
            'model_source_url': f'https://huggingface.co/{identifier}',
            'architecture': '; '.join(config.get('architectures') or []) or original.get('architecture'),
            'config_model_type': config.get('model_type'), 'config_vocab_size': original.get('config_vocab_size'),
            'parameters_total': (model.get('safetensors') or {}).get('total'),
            'pipeline_tag': model.get('pipeline_tag'), 'library_name': model.get('library_name'),
            'license': card.get('license'), 'gated': model.get('gated'),
            'release_date': original.get('release_date'), 'release_date_precision': original.get('release_date_precision'),
            'release_date_source_url': original.get('release_date_source_url'),
            'hf_repository_created_at': model.get('createdAt'), 'earliest_repository_record_at': stamp,
            'earliest_repository_record_kind': kind, 'earliest_repository_record_source_url': source,
            'timeline_date': timeline_date, 'timeline_date_basis': timeline_basis,
            'timeline_date_source_url': original.get('timeline_date_source_url') or source,
            'downloads_30d': model.get('downloads'), 'likes': model.get('likes'),
            'selection_task': model['selection_task'], 'selection_rank_by_downloads': model['selection_rank_by_downloads'],
            'declared_base_models': parents, 'declared_lineage_roots': roots,
            'lineage_cycle_detected': cycle, 'variant_types': relations, 'lineage_evidence': evidence,
            'revision': model.get('sha'), 'tokenizer_status': status,
            'tokenizer_count_scope': scope, 'tokenizer_local_path': local_path if measured else None,
            'tokenizer_vocab_size': measurement.get('tokenizer_vocab_size') if measured else None,
            'tokenizer_base_vocab_size': measurement.get('tokenizer_base_vocab_size') if measured else None,
            'tokenizer_added_vocab_size': measurement.get('tokenizer_added_vocab_size') if measured else None,
            'tokenizer_special_token_count': measurement.get('tokenizer_special_token_count') if measured else None,
            'tokenizer_id_capacity': measurement.get('tokenizer_id_capacity') if measured else None,
            'tokenizer_algorithm': measurement.get('tokenizer_algorithm') if measured else None,
            'tokenizer_source_url': measurement.get('source_url') or measurement.get('tokenizer_source_url'),
            'tokenizer_artifact_sha256': (measurement.get('sha256') or measurement.get('tokenizer_sha256')) if measured else None,
            'shared_tokenizer_json_sha256': group,
            'shared_tokenizer_json_repositories': len(group_members[group]) if group else None,
            'shared_vocabulary_sha256': measurement.get('vocabulary_sha256'),
            'shared_vocabulary_repositories': len(vocab_members[measurement['vocabulary_sha256']]) if measurement.get('vocabulary_sha256') else None,
            'declared_base_tokenizer_json_sha256': origin_groups,
            'pretraining_dataset_tokens': original.get('pretraining_dataset_tokens'),
            'pretraining_tokens_processed': original.get('pretraining_tokens_processed'),
            'pretraining_count_qualifier': original.get('pretraining_count_qualifier'),
            'pretraining_count_notes': original.get('pretraining_count_notes'),
            'pretraining_source_url': original.get('pretraining_source_url'),
            'collected_at': datetime.fromisoformat(model['collected_at']).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'notes': measurement.get('error') or original.get('notes'),
        }
        rows.append(row)
    # Closed-model records remain available without pretending the Hub lists them.
    for original in curated:
        if original['weights_access'] != 'closed_weights':
            continue
        row = dict.fromkeys(rows[0])
        for key in row:
            if key in original:
                row[key] = original[key]
        row.update(selection_task='curated_closed_model', declared_base_models=[], declared_lineage_roots=[],
                   variant_types=['closed_model'], lineage_cycle_detected=False,
                   declared_base_tokenizer_json_sha256=[], tokenizer_artifact_sha256=original['tokenizer_sha256'])
        rows.append(row)
    if len({row['model_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate model identifiers')
    for row in rows:
        if row['timeline_date']:
            datetime.fromisoformat(row['timeline_date'])
        if row['tokenizer_vocab_size'] is not None:
            assert row['tokenizer_vocab_size'] == row['tokenizer_base_vocab_size'] + row['tokenizer_added_vocab_size']
            assert row['tokenizer_vocab_size'] <= row['tokenizer_id_capacity']
    save_json(cache / 'model-catalog.json', rows)
    (cache / 'model-catalog.csv').write_text(csv_text(rows), newline='')
    groups = []
    by_id = {row['model_id']: row for row in rows}
    for digest, identifiers in group_members.items():
        dated = [by_id[i] for i in identifiers if by_id[i]['timeline_date']]
        representative = min(dated, key=lambda row: (row['timeline_date'], row['model_id'])) if dated else by_id[identifiers[0]]
        groups.append({'tokenizer_json_sha256': digest, 'repository_count': len(identifiers),
                       'vocabulary_size': representative['tokenizer_vocab_size'],
                       'earliest_observed_date_in_sample': representative['timeline_date'],
                       'representative_model_id': representative['model_id'], 'model_ids': sorted(identifiers)})
    groups.sort(key=lambda row: (-row['repository_count'], row['tokenizer_json_sha256']))
    save_json(cache / 'tokenizer-groups.json', groups)
    vocab_groups = []
    for digest, identifiers in vocab_members.items():
        dated = [by_id[i] for i in identifiers if by_id[i]['timeline_date']]
        representative = min(dated, key=lambda row: (row['timeline_date'], row['model_id'])) if dated else by_id[identifiers[0]]
        vocab_groups.append({'vocabulary_sha256': digest, 'repository_count': len(identifiers),
                             'vocabulary_size': representative['tokenizer_vocab_size'],
                             'earliest_observed_date_in_sample': representative['timeline_date'],
                             'representative_model_id': representative['model_id'], 'model_ids': sorted(identifiers)})
    vocab_groups.sort(key=lambda row: (-row['repository_count'], row['vocabulary_sha256']))
    save_json(cache / 'vocabulary-groups.json', vocab_groups)
    if update_curated:
        save_json(ROOT / 'docs/model-tokenizer-statistics/models.json', curated)
        (ROOT / 'docs/model-tokenizer-statistics/models.csv').write_text(csv_text(curated), newline='')
    save_json(cache / 'curated-models-with-dates.json', curated)
    summary = {'catalog_rows': len(rows), 'hf_repositories': len(models),
               'closed_model_rows': sum(row['weights_access'] == 'closed_weights' for row in rows),
               'measured_rows': sum(row['tokenizer_vocab_size'] is not None for row in rows),
               'measured_tokenizer_json_rows': sum(row['tokenizer_status'] == 'measured_tokenizer_json' for row in rows),
               'unique_tokenizer_json_files': len(groups),
               'unique_token_to_id_mappings': len(vocab_groups),
               'tokenizer_status_counts': dict(Counter(row['tokenizer_status'] for row in rows)),
               'timeline_date_basis_counts': dict(Counter(row['timeline_date_basis'] for row in rows)),
               'variant_type_counts_nonexclusive': dict(Counter(kind for row in rows for kind in row['variant_types'])),
               'curated_timeline_coverage': sum(row['timeline_date'] is not None for row in curated),
               'json_sha256': hashlib.sha256((cache / 'model-catalog.json').read_bytes()).hexdigest()}
    save_json(cache / 'catalog-summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--update-curated', action='store_true')
    arguments = parser.parse_args()
    export(arguments.cache_dir, arguments.update_curated)
