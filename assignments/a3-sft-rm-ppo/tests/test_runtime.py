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


def arguments(tmp_path, attention='eager', disable_tf32=False):
    return SimpleNamespace(cache=tmp_path / 'cache', out=tmp_path,
                           attn_implementation=attention, max_wall_minutes=2,
                           calibrate=False, resume=False, split='dev', quick=False,
                           disable_tf32=disable_tf32)


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
                                    'adamw_step', 'cached_sampling', 'behavior_recompute', 'checkpoint_round_trip'}
    assert result['gradient_norm'] > 0
    assert result['behavior_recompute_max_error'] < 2e-5
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


@pytest.mark.parametrize('release', ['r2', 'r3'])
def test_original_records_keep_their_fingerprint(tmp_path, release):
    cfg = run.read_config(run.ROOT / 'configs/smoke.json')
    p = run.provenance(arguments(tmp_path), cfg, torch.device('cpu'))
    del p['tf32_policy']
    if release == 'r2':
        del p['attn_implementation']
    identity = {'config': p['config'], 'model': p['model'], 'model_revision': p['model_revision'],
                'sources': p['source_hashes'], 'data': p['data'],
                'device_type': 'cpu', 'dtype': p['dtype']}
    if release == 'r3':
        identity['attn_implementation'] = p['attn_implementation']
    p['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    validate_results._provenance({'provenance': p}, validate_results._context(run.ROOT), False, [], 'legacy')


@pytest.mark.parametrize('disable_tf32', [False, True])
def test_preflight_cli_preserves_each_attempt(tmp_path, monkeypatch, tiny_checkpoint, disable_tf32):
    monkeypatch.setattr(run.AutoTokenizer, 'from_pretrained', lambda *a, **kw: ProbeTokenizer())
    monkeypatch.setattr(run.sys, 'argv', [
        'run.py', 'preflight', '--device', 'cpu', '--attn-implementation', 'eager', '--out', str(tmp_path)]
        + (['--disable-tf32'] if disable_tf32 else []))
    run.main()
    run.main()
    reports = list(tmp_path.glob('preflight-*.json'))
    assert len(reports) == 2
    assert all(json.loads(p.read_text())['provenance']['attn_implementation'] == 'eager' for p in reports)
    assert all(json.loads(p.read_text())['provenance']['tf32_policy'] ==
               ('disabled' if disable_tf32 else 'config') for p in reports)
    assert not (tmp_path / 'training.json').exists()


@pytest.mark.parametrize('config_tf32,disabled,expected', [
    (True, False, True), (True, True, False), (False, False, False), (False, True, False)])
def test_precision_override_sets_both_backends_without_changing_recipe(
        tmp_path, config_tf32, disabled, expected):
    cfg = run.read_config(run.ROOT / 'configs/main.json')
    cfg['tf32'] = config_tf32
    original = copy.deepcopy(cfg)
    args = arguments(tmp_path, disable_tf32=disabled)
    # The setters are available on CPU builds; no GPU computation is claimed.
    # Restore explicitly: Torch exposes these through dynamic backend attributes.
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = not expected
        torch.backends.cudnn.allow_tf32 = not expected
        run.configure_precision(args, cfg, torch.device('cuda'))
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    assert cfg == original
    assert run.tf32_enabled(args, cfg, torch.device('cpu')) is False


def test_tf32_policy_is_bound_to_resume_evaluation_checkpoints_and_packaging(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'synthetic GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout=''))
    cfg = run.read_config(run.ROOT / 'configs/main.json')
    device = torch.device('cuda')
    default = run.provenance(arguments(tmp_path), cfg, device)
    args = arguments(tmp_path, disable_tf32=True)
    disabled = run.provenance(args, cfg, device)
    assert default['tf32'] is True and disabled['tf32'] is False
    assert default['fingerprint'] != disabled['fingerprint']
    assert default['config'] == disabled['config'] == cfg
    context = validate_results._context(run.ROOT)
    for p in (default, disabled):
        validate_results._provenance({'provenance': p}, context, True, [], 'probe')
    changed = copy.deepcopy(disabled)
    changed.update(tf32_policy='config', tf32=True)
    with pytest.raises(ValueError, match='fingerprint'):
        validate_results._provenance({'provenance': changed}, context, True, [], 'probe')
    changed = copy.deepcopy(disabled)
    changed['tf32'] = True
    with pytest.raises(ValueError, match='TF32 setting'):
        validate_results._provenance({'provenance': changed}, context, True, [], 'probe')
    for name in ('training.json', 'training-partial.json'):
        run.write_json(tmp_path / name, {'provenance': default})
    torch.save({'fingerprint': default['fingerprint']}, tmp_path / 'sft.pt')
    with pytest.raises(ValueError, match='checkpoint provenance'):
        run.verify_checkpoint(tmp_path / 'sft.pt', disabled['fingerprint'])
    args.resume = True
    timer = run.Timer(torch.device('cpu'), 2)
    with pytest.raises(ValueError, match='device/precision'):
        run.train(args, cfg, None, [], device, disabled, timer)
    with pytest.raises(ValueError, match='device/precision'):
        run.evaluate(args, cfg, None, [], device, disabled, timer)


@pytest.mark.parametrize('error', [.03650093, float('nan'), float('inf')])
def test_behavior_guard_rejects_reported_error_and_nonfinite_values(error):
    with pytest.raises(ValueError, match='disable-tf32'):
        run.check_behavior_logp(torch.tensor([[error]]), torch.zeros(1, 1), torch.ones(1, 1))


def test_behavior_guard_ignores_padding_and_accepts_small_error():
    behavior = torch.tensor([[6.71e-7, float('nan')]])
    assert run.check_behavior_logp(behavior, torch.zeros(1, 2), torch.tensor([[1., 0.]])) < 1e-6


def test_preflight_preserves_probability_mismatch_failure(tmp_path, monkeypatch, tiny_checkpoint):
    original = run.generate_responses
    def mismatched(*args, **kwargs):
        responses, logp = original(*args, **kwargs)
        return responses, logp + .03650093
    monkeypatch.setattr(run, 'generate_responses', mismatched)
    monkeypatch.setattr(run.AutoTokenizer, 'from_pretrained', lambda *a, **kw: ProbeTokenizer())
    monkeypatch.setattr(run.sys, 'argv', [
        'run.py', 'preflight', '--device', 'cpu', '--attn-implementation', 'eager', '--out', str(tmp_path)])
    with pytest.raises(ValueError, match='behavior/recomputed'):
        run.main()
    failures = list(tmp_path.glob('failed-attempt-*.json'))
    assert len(failures) == 1
    record = json.loads(failures[0].read_text())
    assert 'behavior/recomputed' in record['error']
    assert record['provenance']['tf32'] is False
    assert not list(tmp_path.glob('preflight-*.json'))
