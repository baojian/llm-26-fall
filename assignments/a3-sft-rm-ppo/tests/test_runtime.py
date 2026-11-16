"""Offline runner checks; these do not require implementations of the eight tasks."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import run
import validate_results


class ProbeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    eos_token = '<eos>'

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[3, 4], [5, 6, 7, 8]]}


@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=64, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=2))
    directory = tmp_path / 'local-model'
    model.save_pretrained(directory, safe_serialization=True)
    monkeypatch.setattr(run, 'MODEL', str(directory))
    return directory


def arguments(tmp_path, attention='eager'):
    return SimpleNamespace(cache=tmp_path / 'cache', out=tmp_path,
                           attn_implementation=attention, max_wall_minutes=2,
                           calibrate=False, resume=False, split='dev', quick=False)


@pytest.mark.parametrize('scope', ['all', 'last_block'])
def test_eager_preflight_without_student_implementations(tmp_path, monkeypatch, tiny_checkpoint, scope):
    def unfinished(*args, **kwargs):
        raise NotImplementedError('preflight must not call assessed functions')
    for name in ('response_labels', 'sft_loss', 'terminal_scores', 'preference_loss',
                 'shaped_rewards', 'gae', 'ppo_loss', 'paired_summary'):
        monkeypatch.setattr(run.a3, name, unfinished)
    cfg = run.read_config(run.ROOT / 'configs/smoke.json')
    cfg['trainable_scope'] = scope
    device = torch.device('cpu')
    result = run.preflight(arguments(tmp_path), cfg, ProbeTokenizer(), device,
                           {'fingerprint': 'synthetic-probe'}, run.Timer(device, 2))
    assert result['status'] == 'completed'
    assert set(result['checks']) == {'float32_loading', 'forward', 'backward',
                                    'adamw_step', 'cached_sampling', 'checkpoint_round_trip'}
    assert result['gradient_norm'] > 0
    assert not (tmp_path / 'training.json').exists()
    assert not list(tmp_path.glob('preflight-*'))


def test_float32_loader_retains_sdpa_default_and_eager_option(tmp_path, tiny_checkpoint):
    cfg = run.read_config(run.ROOT / 'configs/smoke.json')
    device = torch.device('cpu')
    eager = run.load_model(tmp_path, cfg, device, attn_implementation='eager')
    assert eager.config._attn_implementation == 'eager'
    assert all(p.dtype == torch.float32 for p in eager.parameters())
    if getattr(GPT2LMHeadModel, '_supports_sdpa', False):
        default = run.load_model(tmp_path, cfg, device)
        assert default.config._attn_implementation == 'sdpa'
    else:
        # GPT-2 in Transformers 4.40.2 has no SDPA implementation. Selection is
        # explicit: no silent fallback or suppression of unrelated load errors.
        with pytest.raises(ValueError, match='scaled_dot_product_attention'):
            run.load_model(tmp_path, cfg, device)


def test_backend_is_bound_to_resume_evaluation_and_packaging(tmp_path):
    cfg = run.read_config(run.ROOT / 'configs/smoke.json')
    device = torch.device('cpu')
    eager = run.provenance(arguments(tmp_path, 'eager'), cfg, device)
    sdpa = run.provenance(arguments(tmp_path, 'sdpa'), cfg, device)
    assert eager['fingerprint'] != sdpa['fingerprint']
    assert eager['numpy'] and eager['attn_implementation'] == 'eager'
    context = validate_results._context(run.ROOT)
    validate_results._provenance({'provenance': eager}, context, False, [], 'probe')
    changed = copy.deepcopy(eager)
    changed['attn_implementation'] = 'sdpa'
    with pytest.raises(ValueError, match='fingerprint'):
        validate_results._provenance({'provenance': changed}, context, False, [], 'probe')
    for name in ('training.json', 'training-partial.json'):
        run.write_json(tmp_path / name, {'provenance': eager})
    args = arguments(tmp_path, 'sdpa')
    args.resume = True
    with pytest.raises(ValueError, match='attention backend'):
        run.train(args, cfg, None, [], device, sdpa, run.Timer(device, 2))
    with pytest.raises(ValueError, match='attention backend'):
        run.evaluate(args, cfg, None, [], device, sdpa, run.Timer(device, 2))


def test_original_records_without_backend_keep_their_fingerprint(tmp_path):
    cfg = run.read_config(run.ROOT / 'configs/smoke.json')
    p = run.provenance(arguments(tmp_path), cfg, torch.device('cpu'))
    del p['attn_implementation']
    identity = {'config': p['config'], 'model': p['model'], 'model_revision': p['model_revision'],
                'sources': p['source_hashes'], 'data': p['data'],
                'device_type': 'cpu', 'dtype': p['dtype']}
    p['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    validate_results._provenance({'provenance': p}, validate_results._context(run.ROOT), False, [], 'legacy')


def test_preflight_cli_preserves_each_attempt(tmp_path, monkeypatch, tiny_checkpoint):
    monkeypatch.setattr(run.AutoTokenizer, 'from_pretrained', lambda *a, **kw: ProbeTokenizer())
    monkeypatch.setattr(run.sys, 'argv', [
        'run.py', 'preflight', '--device', 'cpu', '--attn-implementation', 'eager', '--out', str(tmp_path)])
    run.main()
    run.main()
    reports = list(tmp_path.glob('preflight-*.json'))
    assert len(reports) == 2
    assert all(json.loads(p.read_text())['provenance']['attn_implementation'] == 'eager' for p in reports)
    assert not (tmp_path / 'training.json').exists()
