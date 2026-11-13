"""Validate run evidence without loading models or grading model performance.

Completed records must describe the current, fixed main run. Partial and failed
records remain separately attributed evidence; they never satisfy a missing
completed file. This module uses only the Python standard library.
"""
import hashlib
import json
import math
from pathlib import Path

RECORDS = ('training.json', 'evaluation-dev.json', 'evaluation-test.json')
MODEL = 'openai-community/gpt2'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
# Hash of json.dumps(published_main_config, sort_keys=True), independent of
# whitespace. Changing a local main.json must not rename a smaller run "main".
EXPECTED_MAIN_CONFIG_SHA256 = '776e29a9fdf275d151161ccfb930739f06cbd44e5ce3d031743813245223e59b'
STAGES = ('baseline', 'sft', 'ppo')
TRAIN_STAGES = ('sft', 'reward', 'ppo')
METRICS = ('accuracy', 'valid_label_rate', 'mean_reward', 'mean_response_tokens')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, label):
    _require(isinstance(value, dict), f'{label} must be a JSON object')
    return value


def _number(value, label, minimum=None):
    _require(type(value) in (int, float) and math.isfinite(value), f'{label} must be a finite number')
    if minimum is not None:
        _require(value >= minimum, f'{label} must be at least {minimum}')


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f'{label} must be an integer >= {minimum}')


def _keys(value, keys, label):
    _object(value, label)
    missing = set(keys) - value.keys()
    _require(not missing, f'{label} is missing {", ".join(sorted(missing))}')


def _finite_json(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_json(child, f'{label}.{key}')
    elif isinstance(value, list):
        for child in value:
            _finite_json(child, label)
    elif isinstance(value, float):
        _require(math.isfinite(value), f'{label} contains a non-finite number')


def read_json(path):
    """Read a present record strictly; malformed files are never called missing."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f'{path.name} must be an ordinary file')
    try:
        value = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{path.name}: malformed JSON ({exc}); repair the file or retain the original error as evidence') from exc
    _object(value, path.name)
    _finite_json(value, path.name)
    return value


def _context(root):
    root = Path(root)
    cfg = read_json(root / 'configs/main.json')
    manifest = read_json(root / 'data/manifest.json')
    datasets = {}
    for split in ('train', 'dev', 'test'):
        path = root / f'data/{split}.jsonl'
        info = manifest['files'][path.name]
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'],
                 f'supplied {path.name} checksum differs from data/manifest.json')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        _require(len(rows) == info['examples'], f'supplied {path.name} has an unexpected row count')
        datasets[split] = {row['id']: row for row in rows}
        _require(len(datasets[split]) == len(rows), f'supplied {path.name} has duplicate IDs')
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('a3.py', 'run.py')}
    return cfg, manifest, datasets, sources


def _provenance(record, context, complete, warnings, label):
    _keys(record, ('provenance',), label)
    p = record['provenance']
    required = ('config', 'model', 'model_revision', 'source_hashes', 'data', 'device', 'dtype', 'fingerprint')
    _keys(p, required, f'{label}.provenance')
    _object(p['config'], f'{label}.config')
    _keys(p['source_hashes'], ('a3.py', 'run.py'), f'{label}.source_hashes')
    _require(isinstance(p['device'], str) and p['device'], f'{label}.device must be a string')
    identity = {'config': p['config'], 'model': p['model'], 'model_revision': p['model_revision'],
                'sources': p['source_hashes'], 'data': p['data'],
                'device_type': p['device'].split(':')[0], 'dtype': p['dtype']}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    _require(p['fingerprint'] == fingerprint, f'{label}: fingerprint does not match its config/model/source/data/device identity')
    cfg, manifest, _, sources = context
    differences = []
    for matches, description in (
        (p['source_hashes'] == sources, 'source hashes differ from the packaged a3.py/run.py'),
        (p['config'] == cfg and hashlib.sha256(json.dumps(p['config'], sort_keys=True).encode()).hexdigest() == EXPECTED_MAIN_CONFIG_SHA256,
         'configuration differs from the published main recipe (smoke/calibration/custom runs are not full evidence)'),
        (p['data'] == manifest, 'dataset identity differs from the supplied data manifest'),
        (p['model'] == MODEL and p['model_revision'] == REVISION, 'model/revision differs from the pinned GPT-2'),
        (p['device'].split(':')[0] == 'cuda' and p['dtype'] == 'float32', 'device/precision differs from the main CUDA float32 recipe'),
        (p.get('calibration_only') is False, 'calibration_only is not explicitly false'),
    ):
        if not matches:
            differences.append(description)
    if complete:
        _require(not differences, f'{label}: {"; ".join(differences)}; use records from the current fixed main run')
        _keys(p, ('stage_seeds', 'timestamp', 'timezone', 'command', 'git_revision', 'git_dirty',
                  'python', 'torch', 'transformers', 'hardware', 'budget_wall_minutes',
                  'artifact_directory', 'model_cache'), f'{label}.provenance')
    elif differences:
        warnings.append(f'{label}: retained as partial/failed evidence only: {"; ".join(differences)}.')
    seed = p['config'].get('seed')
    _integer(seed, f'{label}.config.seed')
    expected_seeds = {'sft': seed, 'reward': seed + 1, 'ppo': seed + 2}
    if 'stage_seeds' in p:
        _require(p['stage_seeds'] == expected_seeds, f'{label}: stage_seeds disagree with the recorded configuration')
    elif not complete:
        warnings.append(f'{label}: legacy evidence has no stage_seeds; its seed attribution is incomplete.')
    return p


def _attempts(record, complete, warnings, label):
    if label.startswith('failed-attempt-'):
        # Standalone failure logs use the provenance timestamp as their start
        # and share attempt_id with a training/evaluation attempt when present.
        if 'attempt_id' not in record:
            warnings.append(f'{label}: legacy failure has no attempt_id; check the report to avoid counting its cost twice.')
        _number(record.get('elapsed_seconds'), f'{label}.elapsed_seconds', 0)
        _keys(record, ('completed_at',), label)
        return
    attempts = record.get('attempts')
    if attempts is None and not complete:
        warnings.append(f'{label}: no attempt list; failed/resumed runtime accounting may be incomplete.')
        return
    _require(isinstance(attempts, list) and (attempts or not complete), f'{label}.attempts must be a nonempty list for completed evidence')
    for i, attempt in enumerate(attempts):
        item = f'{label}.attempts[{i}]'
        _object(attempt, item)
        needed = ('attempt_id', 'started_at', 'status')
        if not complete and not set(needed) <= attempt.keys():
            warnings.append(f'{item}: legacy attempt metadata is incomplete; preserve its recorded cost in the report.')
            continue
        _keys(attempt, needed, item)
        _require(attempt['status'] in ('running', 'partial', 'failed', 'completed'), f'{item}: invalid status')
        if attempt['status'] in ('failed', 'completed'):
            _keys(attempt, ('completed_at', 'wall_seconds'), item)
        if 'wall_seconds' in attempt:
            _number(attempt['wall_seconds'], f'{item}.wall_seconds', 0)
    if complete:
        _require(attempts[-1]['status'] == 'completed', f'{label}: the final attempt is not completed')


def _training(record, cfg, context, complete, label):
    expected = {'sft': cfg['sft_steps'], 'reward': cfg['rm_steps'], 'ppo': cfg['ppo_batches']}
    if complete:
        _keys(record, (*TRAIN_STAGES, 'completed_stages', 'stage_seconds', 'tokens',
                       'parameter_counts', 'frozen_role_checks'), label)
    completed = record.get('completed_stages', [])
    _require(isinstance(completed, list) and completed == list(TRAIN_STAGES[:len(completed)]),
             f'{label}.completed_stages must be an ordered prefix of sft/reward/ppo')
    if complete:
        _require(completed == list(TRAIN_STAGES), f'{label}: all three training stages must be completed')
    for stage in TRAIN_STAGES:
        if stage not in record:
            continue
        entries = record[stage]
        _require(isinstance(entries, list), f'{label}.{stage} must be a list')
        _require(len(entries) <= expected[stage], f'{label}.{stage}: too many updates for its configuration')
        if complete or stage in completed:
            _require(len(entries) == expected[stage], f'{label}.{stage}: expected {expected[stage]} completed updates/batches, found {len(entries)}')
        for i, entry in enumerate(entries):
            item = f'{label}.{stage}[{i}]'
            if stage in ('sft', 'reward'):
                _keys(entry, ('step', 'loss', 'input_tokens', 'examples', 'scored_tokens'), item)
                _integer(entry['step'], f'{item}.step', 1)
                _integer(entry['examples'], f'{item}.examples', 1)
                _require(entry['step'] == i + 1, f'{item}: steps must be consecutive starting at 1')
                _require(entry['examples'] == cfg['batch_size'], f'{item}: batch size differs from configuration')
                _number(entry['loss'], f'{item}.loss')
                _integer(entry['input_tokens'], f'{item}.input_tokens', 1)
                if stage == 'sft':
                    _integer(entry['scored_tokens'], f'{item}.scored_tokens', 1)
            else:
                _keys(entry, ('batch', 'ids', 'sequences', 'response_tokens', 'epochs', 'rollouts',
                              'mean_reward', 'behavior_recompute_max_error', 'sampled_log_ratio_mean',
                              'old_probabilities_unchanged'), item)
                _integer(entry['batch'], f'{item}.batch', 1)
                _integer(entry['sequences'], f'{item}.sequences', 1)
                _require(entry['batch'] == i + 1, f'{item}: batches must be consecutive starting at 1')
                _require(entry['sequences'] == cfg['rollout_batch_size'], f'{item}: rollout size differs from configuration')
                ids = entry['ids']
                _require(isinstance(ids, list) and len(ids) == entry['sequences'] and len(set(ids)) == len(ids), f'{item}: missing/duplicate rollout IDs')
                if complete or record['provenance']['data'] == context[1]:
                    _require(set(ids) <= context[2]['train'].keys(), f'{item}: rollout IDs are not training IDs')
                _integer(entry['response_tokens'], f'{item}.response_tokens', entry['sequences'])
                epochs = entry['epochs']
                _require(isinstance(epochs, list) and len(epochs) == cfg['ppo_epochs'], f'{item}: missing PPO optimizer epochs')
                for epoch_index, epoch in enumerate(epochs):
                    _keys(epoch, ('epoch', 'minibatches'), item)
                    _require(epoch['epoch'] == epoch_index + 1 and isinstance(epoch['minibatches'], list) and epoch['minibatches'], f'{item}: invalid PPO epoch/minibatches')
                    expected_microbatches = math.ceil(cfg['rollout_batch_size'] / cfg['micro_batch_size'])
                    _require(len(epoch['minibatches']) == expected_microbatches, f'{item}: missing PPO microbatch diagnostics')
                    for diagnostic in epoch['minibatches']:
                        _keys(diagnostic, ('policy_loss', 'value_loss', 'clip_fraction'), f'{item}.minibatches')
                        for key in ('policy_loss', 'value_loss', 'clip_fraction'):
                            _number(diagnostic[key], f'{item}.{key}')
                _require(isinstance(entry['rollouts'], list) and entry['rollouts'], f'{item}: missing rollout trace')
                for trace in entry['rollouts']:
                    _keys(trace, ('id', 'sample', 'response_token_ids', 'reward_score', 'old_logp',
                                  'reference_logp', 'old_values', 'rewards', 'advantages', 'returns'), f'{item}.rollouts')
                    _require(trace['id'] in ids and isinstance(trace['sample'], str), f'{item}: invalid rollout trace ID/sample')
                    token_ids = trace['response_token_ids']
                    _require(isinstance(token_ids, list) and 1 <= len(token_ids) <= cfg['max_new_tokens'], f'{item}: invalid rollout token list')
                    for token_id in token_ids:
                        _integer(token_id, f'{item}.response_token_ids')
                    _number(trace['reward_score'], f'{item}.reward_score')
                    for key in ('old_logp', 'reference_logp', 'old_values', 'rewards', 'advantages', 'returns'):
                        _require(isinstance(trace[key], list) and len(trace[key]) == len(token_ids), f'{item}: {key} does not align with rollout tokens')
                        for value in trace[key]:
                            _number(value, f'{item}.{key}')
                _require(type(entry['old_probabilities_unchanged']) is bool, f'{item}: old_probabilities_unchanged must be boolean')
                for key in ('mean_reward', 'behavior_recompute_max_error', 'sampled_log_ratio_mean'):
                    _number(entry[key], f'{item}.{key}')
    if complete:
        for stage in TRAIN_STAGES:
            _number(record['stage_seconds'].get(stage), f'{label}.stage_seconds.{stage}', 0)
        for stage in ('sft', 'reward'):
            counts = record['parameter_counts'].get(stage)
            _keys(counts, ('backbone_total', 'backbone_trainable'), f'{label}.parameter_counts.{stage}')
            _require(counts['backbone_total'] == counts['backbone_trainable'] == 124439808,
                     f'{label}: main recipe must train all GPT-2 backbone parameters')
        frozen = record['frozen_role_checks']
        _keys(frozen, ('before', 'after', 'unchanged'), f'{label}.frozen_role_checks')
        for side in ('before', 'after'):
            _keys(frozen[side], ('reference', 'reward', 'reward_head'), f'{label}.frozen_role_checks.{side}')
        _require(type(frozen['unchanged']) is bool, f'{label}: frozen-role check must be boolean')
        totals = {'sft_input': sum(r['input_tokens'] for r in record['sft']),
                  'sft_scored': sum(r['scored_tokens'] for r in record['sft']),
                  'reward_input': sum(r['input_tokens'] for r in record['reward']),
                  'ppo_sampled_response': sum(r['response_tokens'] for r in record['ppo']),
                  'ppo_sampled_sequences': sum(r['sequences'] for r in record['ppo'])}
        _require(record['tokens'] == totals, f'{label}: token/sequence totals disagree with the stage records')


def _metrics(value, label):
    _keys(value, METRICS, label)
    for key in METRICS:
        _number(value[key], f'{label}.{key}')
    for key in ('accuracy', 'valid_label_rate'):
        _require(0 <= value[key] <= 1, f'{label}.{key} must be between 0 and 1')
    _require(value['mean_response_tokens'] >= 0, f'{label}.mean_response_tokens must be nonnegative')


def _paired(value, label):
    _keys(value, ('families', 'examples', 'before_accuracy', 'after_accuracy', 'delta', 'improved', 'regressed', 'ci95'), label)
    for key in ('families', 'examples'):
        _integer(value[key], f'{label}.{key}', 1)
    for key in ('before_accuracy', 'after_accuracy', 'delta'):
        _number(value[key], f'{label}.{key}')
    for key in ('improved', 'regressed'):
        _integer(value[key], f'{label}.{key}')
    _require(isinstance(value['ci95'], list) and len(value['ci95']) == 2, f'{label}.ci95 must contain two bounds')
    for bound in value['ci95']:
        _number(bound, f'{label}.ci95')


def _evaluation(record, split, cfg, context, complete, label):
    _keys(record, ('split', 'quick'), label)
    _require(record['split'] == split, f'{label}: split does not match its filename')
    _require(type(record['quick']) is bool, f'{label}.quick must be boolean')
    if complete:
        _require(record['quick'] is False, f'{label}: quick evaluation cannot be submitted as full evidence')
        _keys(record, (*STAGES, 'reward_model', 'metrics', 'slice_metrics', 'slices', 'baseline_to_sft',
                       'sft_to_ppo', 'completed_stages'), label)
        _require(record['completed_stages'] == ['reward', *STAGES], f'{label}: reward and all three policy evaluations must be completed')
    supplied = context[2][split]
    same_data = record['provenance']['data'] == context[1]
    metrics = record.get('metrics', {})
    _object(metrics, f'{label}.metrics')
    for stage in STAGES:
        if stage not in record:
            continue
        predictions = record[stage]
        _require(isinstance(predictions, list), f'{label}.{stage} must be a list')
        seen = set()
        for row in predictions:
            _keys(row, ('id', 'family', 'slice', 'condition', 'attribute', 'target', 'prediction',
                        'response_tokens', 'format_ok', 'reward_score'), f'{label}.{stage} prediction')
            _require(isinstance(row['id'], str) and row['id'] not in seen, f'{label}.{stage}: missing/duplicate prediction ID')
            seen.add(row['id'])
            if same_data:
                _require(row['id'] in supplied, f'{label}.{stage}: unknown {split} ID {row["id"]}')
                _require(all(row[key] == supplied[row['id']][key] for key in ('family', 'slice', 'condition', 'attribute', 'target')),
                         f'{label}.{stage}: metadata/target differs from supplied data for {row["id"]}')
            _require(isinstance(row['prediction'], str), f'{label}.{stage}: prediction must be text')
            _require(type(row['format_ok']) is bool, f'{label}.{stage}: format_ok must be boolean')
            _integer(row['response_tokens'], f'{label}.{stage}.response_tokens', 1)
            _require(row['response_tokens'] <= cfg['max_new_tokens'], f'{label}.{stage}: response exceeds configured token cap')
            _number(row['reward_score'], f'{label}.{stage}.reward_score')
        if complete:
            _require(seen == supplied.keys(), f'{label}.{stage}: full {split} requires exactly all {len(supplied)} supplied IDs; found {len(seen)}')
            _require(stage in metrics, f'{label}.metrics is missing {stage}')
        if stage in metrics:
            _metrics(metrics[stage], f'{label}.metrics.{stage}')
    if 'reward_model' in record:
        reward = record['reward_model']
        _keys(reward, ('pairs',), f'{label}.reward_model')
        pairs = reward['pairs']
        _require(isinstance(pairs, list), f'{label}.reward_model.pairs must be a list')
        for pair in pairs:
            _require(isinstance(pair, (list, tuple)) and len(pair) == 2, f'{label}: each reward pair must have two scores')
            for value in pair:
                _number(value, f'{label}.reward_model.pairs')
        if complete:
            _require(len(pairs) == len(supplied), f'{label}: reward pairs must cover the full split')
            _keys(reward, ('accuracy', 'ties', 'pair_ids'), f'{label}.reward_model')
        if 'accuracy' in reward:
            _number(reward['accuracy'], f'{label}.reward_model.accuracy', 0)
            _require(reward['accuracy'] <= 1, f'{label}: reward pair accuracy must be <= 1')
        if 'ties' in reward:
            _integer(reward['ties'], f'{label}.reward_model.ties')
            _require(reward['ties'] <= len(pairs), f'{label}: more ties than reward pairs')
        if 'pair_ids' in reward:
            ids = reward['pair_ids']
            _require(isinstance(ids, list) and len(ids) == len(pairs) and all(isinstance(i, str) for i in ids)
                     and len(set(ids)) == len(ids), f'{label}: reward pair IDs must match pair count without duplicates')
            if same_data:
                _require(set(ids) <= supplied.keys(), f'{label}: unknown reward pair ID')
    if complete:
        for key in ('baseline_to_sft', 'sft_to_ppo'):
            _paired(record[key], f'{label}.{key}')
        groups = {f'{field}:{row[field]}' for row in supplied.values() for field in ('slice', 'condition', 'attribute')}
        for field in ('slices', 'slice_metrics'):
            _object(record[field], f'{label}.{field}')
            _require(groups <= record[field].keys(), f'{label}.{field}: missing present/absent, condition or attribute groups')
        for group in groups:
            _paired(record['slices'][group], f'{label}.slices.{group}')
            for stage in STAGES:
                _metrics(record['slice_metrics'][group].get(stage), f'{label}.slice_metrics.{group}.{stage}')


def _validate(record, kind, context, complete, warnings):
    _object(record, kind)
    _finite_json(record, kind)
    status = record.get('status')
    if complete:
        _require(status == 'completed', f'{kind}: completed evidence needs status="completed"')
        _keys(record, ('wall_seconds', 'completed_at', 'peak_process_rss_bytes',
                       'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'), kind)
        _number(record['wall_seconds'], f'{kind}.wall_seconds', 0)
        for field in ('peak_process_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
            if record[field] is not None:
                _integer(record[field], f'{kind}.{field}')
    elif status is None:
        warnings.append(f'{kind}: legacy record has no explicit status; retained as incomplete evidence only.')
    else:
        _require(status in ('partial', 'failed', 'completed'), f'{kind}: unknown record status')
    try:
        p = _provenance(record, context, complete, warnings, kind)
        _attempts(record, complete, warnings, kind)
        if kind.startswith('training'):
            _training(record, p['config'], context, complete, kind)
        elif kind.startswith('evaluation-'):
            split = kind.split('-')[1].split('.')[0]
            _require(split in ('dev', 'test'), f'{kind}: unknown evaluation split')
            _evaluation(record, split, p['config'], context, complete, kind)
        elif kind.startswith('failed-attempt-'):
            _keys(record, ('error',), kind)
            _require(isinstance(record['error'], str) and record['error'].strip(), f'{kind}: failure needs an error explanation')
            _require(status in (None, 'failed'), f'{kind}: failed attempt cannot be marked completed')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'{kind}: malformed record structure ({exc}); retain the original and repair the recorded fields') from exc
    return p['fingerprint']


def validate_record(record, kind, root, *, complete=True):
    """Validate one named run record; return attribution warnings for partials."""
    warnings = []
    _validate(record, kind, _context(root), complete, warnings)
    return warnings


def collect_results(run, root, partial_reason=None):
    """Collect strict completed records and explicitly separate partial evidence."""
    run = Path(run)
    context = None
    reason = partial_reason.strip() if partial_reason else ''
    results = {'schema_version': 1, 'records': {}, 'unavailable_evidence': {},
               'partial_records': {}, 'failed_attempts': {}, 'warnings': []}
    def validate(record, name, complete):
        nonlocal context
        if context is None:
            context = _context(root)
        return _validate(record, name, context, complete, results['warnings'])

    fingerprints = set()
    for name in RECORDS:
        path = run / name
        if path.exists() or path.is_symlink():
            record = read_json(path)
            incomplete = record.get('status') in ('partial', 'failed')
            fingerprint = validate(record, name, not incomplete)
            if not incomplete:
                fingerprints.add(fingerprint)
                results['records'][name] = record
                continue
            results['partial_records'][name] = record
        if not reason:
            raise ValueError(f'{name} is missing or explicitly incomplete; use --partial-reason "specific explanation" and retain any partial/failed records')
        results['unavailable_evidence'][name] = reason
    _require(len(fingerprints) <= 1, 'completed records have different fingerprints; do not combine different runs/configurations/sources')
    partial_names = ['training-partial.json'] + [f'evaluation-{split}{quick}-partial.json'
                     for split in ('dev', 'test') for quick in ('', '-quick')]
    for name in partial_names:
        path = run / name
        if path.exists() or path.is_symlink():
            record = read_json(path)
            validate(record, name, False)
            results['partial_records'][name] = record
    for path in sorted(run.glob('failed-attempt-*.json')):
        record = read_json(path)
        validate(record, path.name, False)
        results['failed_attempts'][path.name] = record
    return results
