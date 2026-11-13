"""Supplied GPT-2 SFT -> reward model -> PPO runner; objectives live in a3.py."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
try:
    import resource
except ImportError:
    resource = None

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, __version__ as transformers_version
import a3

ROOT = Path(__file__).resolve().parent
MODEL = 'openai-community/gpt2'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
CONFIG_KEYS = {
    'name', 'data_version', 'seed', 'trainable_scope', 'sft_steps', 'rm_steps',
    'batch_size', 'micro_batch_size', 'ppo_batches', 'rollout_batch_size',
    'ppo_epochs', 'eval_batch_size', 'eval_examples', 'max_input_tokens',
    'max_new_tokens', 'sft_lr', 'rm_lr', 'ppo_lr', 'weight_decay', 'beta',
    'gamma', 'lambda', 'clip', 'value_coef', 'max_grad_norm', 'tf32', 'smoke',
}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS:
        raise ValueError(f'configuration keys differ: {set(cfg) ^ CONFIG_KEYS}')
    for key in ('sft_steps', 'rm_steps', 'batch_size', 'micro_batch_size', 'ppo_batches',
                'rollout_batch_size', 'ppo_epochs', 'eval_batch_size', 'max_input_tokens', 'max_new_tokens'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['trainable_scope'] not in ('all', 'last_block'):
        raise ValueError('trainable_scope must be all or last_block')
    if cfg['max_input_tokens'] > 1024 or cfg['max_new_tokens'] >= cfg['max_input_tokens']:
        raise ValueError('invalid GPT-2 context/generation limit')
    if cfg['beta'] < 0 or not 0 <= cfg['gamma'] <= 1 or not 0 <= cfg['lambda'] <= 1:
        raise ValueError('invalid reward/GAE configuration')
    return cfg


def read_data(split, cfg):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    if manifest['version'] != cfg['data_version']:
        raise ValueError('configuration and dataset version disagree')
    path = ROOT / f'data/{split}.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][path.name]['sha256']:
        raise ValueError(f'data checksum mismatch: {path.name}')
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode_rows(tokenizer, rows, cfg):
    prompts = tokenizer([row['prompt'] for row in rows], add_special_tokens=False)['input_ids']
    chosen = tokenizer([row['chosen'] for row in rows], add_special_tokens=False)['input_ids']
    rejected = tokenizer([row['rejected'] for row in rows], add_special_tokens=False)['input_ids']
    encoded = []
    for row, p, c, r in zip(rows, prompts, chosen, rejected):
        c, r = c + [tokenizer.eos_token_id], r + [tokenizer.eos_token_id]
        # Reserve a possible scoring EOS after a response that reaches its cap.
        if not p or len(p) + cfg['max_new_tokens'] + 1 > cfg['max_input_tokens']:
            raise ValueError(f"{row['id']}: prompt plus generation/scoring reserve exceeds context; no truncation")
        if max(len(c), len(r)) > cfg['max_new_tokens']:
            raise ValueError(f"{row['id']}: target plus EOS exceeds response budget")
        encoded.append(dict(row, prompt_ids=p, chosen_ids=c, rejected_ids=r))
    return encoded


def pack_sequences(sequences, pad_id, device, left=False):
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        offset = length - len(seq) if left else 0
        ids[i, offset:offset + len(seq)] = torch.tensor(seq, device=device)
        attention[i, offset:offset + len(seq)] = 1
    return ids, attention


def batch_tokens(rows, key, eos, device):
    ids, mask = pack_sequences([r['prompt_ids'] + r[key + '_ids'] for r in rows], eos, device)
    lengths = torch.tensor([len(r['prompt_ids']) for r in rows], device=device)
    return ids, mask, lengths


def chunks(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def epoch_batches(rows, size, seed):
    rng = random.Random(seed)
    while True:
        order = list(rows)
        rng.shuffle(order)
        yield from chunks(order, size)


def load_model(cache, cfg, device, checkpoint=None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, cache_dir=cache, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, attn_implementation='sdpa')
    model.config.use_cache = False
    if cfg['trainable_scope'] == 'last_block':
        for p in model.parameters():
            p.requires_grad_(False)
        for module in (model.transformer.h[-1], model.transformer.ln_f):
            for p in module.parameters():
                p.requires_grad_(True)
    if checkpoint is not None:
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        missing, unexpected = model.load_state_dict(saved['weights'], strict=False)
        expected_missing = {name for name, p in model.named_parameters() if not p.requires_grad}
        # Tied output weights are saved once via named_parameters.
        allowed = expected_missing | {'lm_head.weight'}
        if unexpected or not set(missing) <= allowed:
            raise ValueError('checkpoint keys do not match the training scope')
    return model.to(device).eval()  # Disable dropout during collection AND updates.


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def trainable_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path, model, fingerprint, head=None):
    state = {'weights': trainable_state(model), 'fingerprint': fingerprint}
    if head is not None:
        state['head'] = {name: p.detach().cpu() for name, p in head.state_dict().items()}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def verify_checkpoint(path, fingerprint):
    if not path.is_file():
        raise ValueError(f'missing completed checkpoint: {path.name}')
    state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    if state.get('fingerprint') != fingerprint:
        raise ValueError(f'checkpoint provenance mismatch: {path.name}')


def scalar_head(model, device, checkpoint=None):
    head = nn.Linear(model.config.n_embd, 1, bias=False, device=device)
    nn.init.zeros_(head.weight)
    if checkpoint is not None:
        head.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)['head'])
    return head


def model_scores(model, head, ids, attention):
    hidden = model.transformer(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    return a3.terminal_scores(head(hidden).squeeze(-1).float(), attention)


def generate_responses(model, tokenizer, rows, cap, sample):
    """Left-padded cached generation, retaining probabilities of actual samples."""
    prompts = [row['prompt_ids'] for row in rows]
    ids, attention = pack_sequences(prompts, tokenizer.eos_token_id, model.device, left=True)
    sampling = {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0} if sample else {}
    generation = GenerationConfig(
        max_new_tokens=cap, do_sample=sample, **sampling,
        repetition_penalty=1.0, num_beams=1, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        use_cache=True, return_dict_in_generate=True, output_scores=sample)
    with torch.no_grad():
        result = model.generate(input_ids=ids, attention_mask=attention, generation_config=generation)
    generated = result.sequences[:, ids.shape[1]:]
    responses = []
    for seq in generated.tolist():
        if tokenizer.eos_token_id in seq:
            seq = seq[:seq.index(tokenizer.eos_token_id) + 1]
        responses.append(seq)
    behavior_logp = None
    if sample:
        behavior_logp = torch.stack([
            scores.float().log_softmax(-1).gather(-1, generated[:, t:t + 1]).squeeze(-1)
            for t, scores in enumerate(result.scores)], dim=1).detach()
    return responses, behavior_logp


def response_statistics(model, head, rows, responses, eos):
    """Use states BEFORE each response action; right-pad action tensors."""
    sequences = [r['prompt_ids'] + response for r, response in zip(rows, responses)]
    ids, attention = pack_sequences(sequences, eos, model.device)
    hidden = model.transformer(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    width = max(map(len, responses))
    positions = torch.tensor([len(r['prompt_ids']) - 1 for r in rows], device=model.device)[:, None]
    positions = positions + torch.arange(width, device=model.device)[None, :]
    positions = positions.clamp_max(hidden.shape[1] - 1)
    states = hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
    actions, mask = pack_sequences(responses, eos, model.device)
    logits = model.lm_head(states).float()
    logp = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
    values = head(states).squeeze(-1).float() if head is not None else torch.zeros_like(logp)
    return logp, values, mask.float()


def reward_scores(model, head, rows, responses, eos):
    sequences = []
    for row, response in zip(rows, responses):
        ending = response if response[-1] == eos else response + [eos]
        sequences.append(row['prompt_ids'] + ending)
    ids, attention = pack_sequences(sequences, eos, model.device)
    with torch.no_grad():
        return model_scores(model, head, ids, attention)


def parameter_digest(module):
    result = hashlib.sha256()
    for name, p in module.named_parameters():
        result.update(name.encode())
        result.update(p.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class Timer:
    def __init__(self, device, minutes):
        self.device = device
        self.limit = minutes * 60
        self.start = self.now()

    def now(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def check(self):
        if self.now() - self.start > self.limit:
            raise TimeoutError('Declared wall-time ceiling reached; partial records/checkpoints retained. Use --resume.')


def make_optimizer(model, cfg, rate, head=None):
    params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += list(head.parameters())
    return torch.optim.AdamW(params, lr=rate, weight_decay=cfg['weight_decay'])


def step_optimizer(model, head, optimizer, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += list(head.parameters())
    torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()


def train(args, cfg, tokenizer, rows, device, metadata, timer):
    fingerprint = metadata['fingerprint']
    partial = args.out / 'training-partial.json'
    logs = json.loads(partial.read_text()) if args.resume and partial.exists() else {
        'sft': [], 'reward': [], 'ppo': [], 'stage_seconds': {}, 'completed_stages': [],
        'provenance': metadata, 'attempts': [], 'parameter_counts': {}}
    if logs['provenance']['fingerprint'] != fingerprint:
        raise ValueError('resume requires identical config, data, model, source code, and device/precision')
    for stage in logs['completed_stages']:
        verify_checkpoint(args.out / f'{stage}.pt', fingerprint)
    logs['attempts'].append({'started_at': metadata['timestamp'], 'resume': args.resume,
                            'command': metadata.get('command'), 'git_revision': metadata.get('git_revision'),
                            'git_dirty': metadata.get('git_dirty'), 'budget_wall_minutes': metadata.get('budget_wall_minutes')})
    write_json(partial, logs)
    eos = tokenizer.eos_token_id

    for stage, steps, seed_offset, rate, checkpoint in [
        ('sft', cfg['sft_steps'], 0, cfg['sft_lr'], 'sft.pt'),
        ('reward', cfg['rm_steps'], 1, cfg['rm_lr'], 'reward.pt'),
    ]:
        if stage in logs['completed_stages']:
            if not (args.out / checkpoint).is_file():
                raise ValueError(f'missing completed checkpoint: {checkpoint}')
            continue
        if logs[stage]:
            logs.setdefault('interrupted_stages', []).append({'stage': stage, 'records': logs[stage]})
        logs[stage] = []
        torch.manual_seed(cfg['seed'] + seed_offset)
        model = load_model(args.cache, cfg, device, args.out / 'sft.pt' if stage == 'reward' else None)
        head = scalar_head(model, device) if stage == 'reward' else None
        optimizer = make_optimizer(model, cfg, rate, head)
        logs['parameter_counts'][stage] = {
            'backbone_total': sum(p.numel() for p in model.parameters()),
            'backbone_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        batches = epoch_batches(rows, cfg['batch_size'], cfg['seed'] + seed_offset)
        started = timer.now()
        for step in range(steps):
            timer.check()
            batch = next(batches)
            optimizer.zero_grad(set_to_none=True)
            loss_total, input_tokens = 0., 0
            target_tokens = sum(len(row['chosen_ids']) for row in batch)
            for micro in chunks(batch, cfg['micro_batch_size']):
                ids, attention, lengths = batch_tokens(micro, 'chosen', eos, device)
                input_tokens += int(attention.sum())
                if stage == 'sft':
                    labels = a3.response_labels(ids, attention, lengths)
                    loss = a3.sft_loss(model(input_ids=ids, attention_mask=attention, use_cache=False).logits.float(), labels)
                    weight = sum(len(row['chosen_ids']) for row in micro) / target_tokens
                else:
                    rejected, mask_r, _ = batch_tokens(micro, 'rejected', eos, device)
                    input_tokens += int(mask_r.sum())
                    loss = a3.preference_loss(model_scores(model, head, ids, attention),
                                              model_scores(model, head, rejected, mask_r))
                    weight = len(micro) / len(batch)
                (loss * weight).backward()
                loss_total += float(loss.detach()) * weight
            step_optimizer(model, head, optimizer, cfg)
            logs[stage].append({'step': step + 1, 'loss': loss_total, 'input_tokens': input_tokens,
                                'examples': len(batch), 'scored_tokens': target_tokens if stage == 'sft' else None})
            if (step + 1) % 32 == 0 or step + 1 == steps:
                print(f'{stage} {step + 1}/{steps}: loss={loss_total:.4f}', flush=True)
                write_json(partial, logs)
        logs['stage_seconds'][stage] = timer.now() - started
        save_checkpoint(args.out / checkpoint, model, fingerprint, head)
        logs['completed_stages'].append(stage)
        write_json(partial, logs)
        del model, head, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if 'ppo' not in logs['completed_stages']:
        if logs['ppo']:
            logs.setdefault('interrupted_stages', []).append({'stage': 'ppo', 'records': logs['ppo']})
        logs['ppo'] = []
        torch.manual_seed(cfg['seed'] + 2)
        reference = freeze(load_model(args.cache, cfg, device, args.out / 'sft.pt'))
        rm = freeze(load_model(args.cache, cfg, device, args.out / 'reward.pt'))
        reward_head = freeze(scalar_head(rm, device, args.out / 'reward.pt'))
        actor = load_model(args.cache, cfg, device, args.out / 'sft.pt')
        value = scalar_head(actor, device)
        optimizer = make_optimizer(actor, cfg, cfg['ppo_lr'], value)
        before = {'reference': parameter_digest(reference), 'reward': parameter_digest(rm),
                  'reward_head': parameter_digest(reward_head)}
        batches = epoch_batches(rows, cfg['rollout_batch_size'], cfg['seed'] + 2)
        started = timer.now()
        for step in range(cfg['ppo_batches']):
            timer.check()
            batch = next(batches)
            responses, behavior_logp = generate_responses(actor, tokenizer, batch, cfg['max_new_tokens'], True)
            with torch.no_grad():
                recomputed, old_values, mask = response_statistics(actor, value, batch, responses, eos)
                ref_logp, _, _ = response_statistics(reference, None, batch, responses, eos)
                scores = reward_scores(rm, reward_head, batch, responses, eos)
                old_logp = behavior_logp[:, :mask.shape[1]].clone()
                mismatch = float(((old_logp - recomputed).abs() * mask).max())
                if mismatch > .02:
                    raise ValueError(f'behavior/recomputed log probabilities disagree ({mismatch}); inspect padding/cache/precision')
                rewards = a3.shaped_rewards(old_logp, ref_logp, scores, mask, cfg['beta'])
                advantages, returns = a3.gae(rewards, old_values, mask, cfg['gamma'], cfg['lambda'])
            original_old = old_logp.clone()
            total_actions = int(mask.sum())
            epochs = []
            for epoch in range(cfg['ppo_epochs']):
                optimizer.zero_grad(set_to_none=True)
                diagnostics = []
                for start in range(0, len(batch), cfg['micro_batch_size']):
                    end = start + cfg['micro_batch_size']
                    micro_rows, micro_responses = batch[start:end], responses[start:end]
                    new_logp, new_values, micro_mask = response_statistics(actor, value, micro_rows, micro_responses, eos)
                    width = micro_mask.shape[1]
                    loss, info = a3.ppo_loss(new_logp, old_logp[start:end, :width],
                        advantages[start:end, :width], new_values, returns[start:end, :width], micro_mask,
                        clip=cfg['clip'], value_coef=cfg['value_coef'])
                    (loss * micro_mask.sum() / total_actions).backward()
                    diagnostics.append(info)
                step_optimizer(actor, value, optimizer, cfg)
                assert torch.equal(original_old, old_logp)
                epochs.append({'epoch': epoch + 1, 'minibatches': diagnostics})
            trace = []
            for i in range(min(2, len(batch))):
                size = len(responses[i])
                trace.append({'id': batch[i]['id'], 'sample': tokenizer.decode(responses[i], skip_special_tokens=True),
                              'response_token_ids': responses[i], 'reward_score': float(scores[i]),
                              'old_logp': old_logp[i, :size].tolist(), 'reference_logp': ref_logp[i, :size].tolist(),
                              'old_values': old_values[i, :size].tolist(), 'rewards': rewards[i, :size].tolist(),
                              'advantages': advantages[i, :size].tolist(), 'returns': returns[i, :size].tolist()})
            logs['ppo'].append({'batch': step + 1, 'ids': [r['id'] for r in batch],
                'sequences': len(batch), 'response_tokens': total_actions, 'epochs': epochs, 'rollouts': trace,
                'mean_reward': float(scores.mean()), 'behavior_recompute_max_error': mismatch,
                'sampled_log_ratio_mean': float(((old_logp - ref_logp) * mask).sum() / mask.sum()),
                'old_probabilities_unchanged': True})
            if (step + 1) % 8 == 0 or step + 1 == cfg['ppo_batches']:
                print(f'ppo {step + 1}/{cfg["ppo_batches"]}: reward={float(scores.mean()):.4f}', flush=True)
                write_json(partial, logs)
        logs['stage_seconds']['ppo'] = timer.now() - started
        after = {'reference': parameter_digest(reference), 'reward': parameter_digest(rm),
                 'reward_head': parameter_digest(reward_head)}
        if before != after:
            raise AssertionError('frozen reference/reward parameters changed')
        logs['frozen_role_checks'] = {'before': before, 'after': after, 'unchanged': True}
        save_checkpoint(args.out / 'ppo.pt', actor, fingerprint, value)
        logs['completed_stages'].append('ppo')
        write_json(partial, logs)
        del reference, rm, reward_head, actor, value, optimizer
        gc.collect()
    logs['tokens'] = {
        'sft_input': sum(r['input_tokens'] for r in logs['sft']),
        'sft_scored': sum(r['scored_tokens'] for r in logs['sft']),
        'reward_input': sum(r['input_tokens'] for r in logs['reward']),
        'ppo_sampled_response': sum(r['response_tokens'] for r in logs['ppo']),
        'ppo_sampled_sequences': sum(r['sequences'] for r in logs['ppo'])}
    return logs


def evaluate(args, cfg, tokenizer, rows, device, metadata, timer):
    training = json.loads((args.out / 'training.json').read_text())
    if training['provenance']['fingerprint'] != metadata['fingerprint']:
        raise ValueError('evaluation must use the frozen training config, sources, dataset, model and device/precision')
    if args.split == 'test' and (cfg['smoke'] or args.calibrate or args.quick
                                 or training['provenance'].get('calibration_only', False)):
        raise ValueError('final test requires the full frozen main configuration')
    for stage in ('sft', 'reward', 'ppo'):
        verify_checkpoint(args.out / f'{stage}.pt', metadata['fingerprint'])
    count = 64 if args.quick else cfg['eval_examples']
    rows = rows if count is None else rows[:count]
    rm = freeze(load_model(args.cache, cfg, device, args.out / 'reward.pt'))
    head = freeze(scalar_head(rm, device, args.out / 'reward.pt'))
    pairs = []
    for batch in chunks(rows, cfg['eval_batch_size']):
        timer.check()
        with torch.no_grad():
            chosen, mask_c, _ = batch_tokens(batch, 'chosen', tokenizer.eos_token_id, device)
            rejected, mask_r, _ = batch_tokens(batch, 'rejected', tokenizer.eos_token_id, device)
            pairs.extend(zip(model_scores(rm, head, chosen, mask_c).tolist(),
                             model_scores(rm, head, rejected, mask_r).tolist()))
    output = {'split': args.split, 'quick': args.quick,
              'reward_model': {'accuracy': sum(a > b for a, b in pairs) / len(pairs),
                               'ties': sum(a == b for a, b in pairs), 'pairs': pairs}}
    valid = set(json.loads((ROOT / 'data/manifest.json').read_text())['valid_answers'])
    output['metrics'] = {}
    for stage in ('baseline', 'sft', 'ppo'):
        model = freeze(load_model(args.cache, cfg, device, args.out / f'{stage}.pt' if stage != 'baseline' else None))
        predictions = []
        for batch in chunks(rows, cfg['eval_batch_size']):
            timer.check()
            responses, _ = generate_responses(model, tokenizer, batch, cfg['max_new_tokens'], False)
            rewards = reward_scores(rm, head, batch, responses, tokenizer.eos_token_id).tolist()
            for row, response, reward in zip(batch, responses, rewards):
                record = {key: row[key] for key in ('id', 'family', 'slice', 'condition', 'attribute', 'target')}
                prediction = tokenizer.decode(response, skip_special_tokens=True)
                record.update(prediction=prediction, response_tokens=len(response),
                              format_ok=prediction.strip().lower() in valid, reward_score=reward)
                predictions.append(record)
        output[stage] = predictions
        output['metrics'][stage] = {
            'accuracy': sum(r['prediction'].strip().lower() == r['target'] for r in predictions) / len(predictions),
            'valid_label_rate': sum(r['format_ok'] for r in predictions) / len(predictions),
            'mean_reward': sum(r['reward_score'] for r in predictions) / len(predictions),
            'mean_response_tokens': sum(r['response_tokens'] for r in predictions) / len(predictions)}
        del model
        gc.collect()
    output['baseline_to_sft'] = a3.paired_summary(output['baseline'], output['sft'])
    output['sft_to_ppo'] = a3.paired_summary(output['sft'], output['ppo'])
    output['slices'] = {}
    for field in ('slice', 'condition', 'attribute'):
        for name in sorted({r[field] for r in rows}):
            output['slices'][field + ':' + name] = a3.paired_summary(
                [r for r in output['sft'] if r[field] == name], [r for r in output['ppo'] if r[field] == name])
    return output


def provenance(args, cfg, device):
    def git(*parts):
        result = subprocess.run(['git', *parts], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    dirty = git('status', '--porcelain')
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('a3.py', 'run.py')}
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    identity = {'config': cfg, 'model': MODEL, 'model_revision': REVISION,
                'sources': sources, 'data': manifest, 'device_type': device.type, 'dtype': 'float32'}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    driver = None
    if device.type == 'cuda':
        try:
            query = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10)
            driver = sorted(set(query.stdout.strip().splitlines())) if query.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            driver = 'unavailable'
    hardware = {'platform': platform.platform(), 'machine': platform.machine(),
                'cuda_runtime': torch.version.cuda, 'nvidia_driver_versions': driver,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'gpu_memory_bytes': torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None}
    return {'timestamp': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'timezone': 'Asia/Shanghai',
            'command': __import__('shlex').join(sys.argv), 'git_revision': git('rev-parse', 'HEAD'),
            'git_dirty': None if dirty is None else bool(dirty), 'source_hashes': sources,
            'model': MODEL, 'model_revision': REVISION, 'data': manifest, 'config': cfg,
            'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers_version,
            'device': str(device), 'hardware': hardware, 'dtype': 'float32', 'tf32': cfg['tf32'] and device.type == 'cuda',
            'stage_seeds': {'sft': cfg['seed'], 'reward': cfg['seed'] + 1, 'ppo': cfg['seed'] + 2},
            'budget_wall_minutes': args.max_wall_minutes, 'calibration_only': args.calibrate,
            'artifact_directory': str(args.out.resolve()), 'model_cache': str(args.cache.resolve()),
            'fingerprint': fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate-data', 'train', 'evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='Short GPU timing probe; not the main assignment')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/main')
    parser.add_argument('--cache', type=Path, default=ROOT / 'outputs/model-cache')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--quick', action='store_true', help='64 dev prompts; saves separate quick evidence')
    parser.add_argument('--resume', action='store_true', help='Skip completed stages; restart an interrupted stage')
    parser.add_argument('--max-wall-minutes', type=float, default=45)
    args = parser.parse_args()
    if args.smoke and (args.config or args.calibrate):
        parser.error('--smoke selects its own configuration')
    cfg_path = args.config or ROOT / 'configs' / ('smoke.json' if args.smoke else 'main.json')
    cfg = read_config(cfg_path)
    full_cfg = dict(cfg)
    if args.calibrate:
        cfg.update(name=cfg['name'] + '-calibration', sft_steps=16, rm_steps=16, ppo_batches=2)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. CPU debugging: --smoke --device cpu')
    if args.command != 'validate-data' and device.type != 'cuda' and not cfg['smoke']:
        parser.error('The main/calibration recipe requires CUDA. Use --smoke --device cpu for wiring checks.')
    if args.max_wall_minutes <= 0:
        parser.error('--max-wall-minutes must be positive')
    if args.quick and (args.command != 'evaluate' or args.split != 'dev'):
        parser.error('--quick is only for development evaluation')
    if args.command == 'evaluate' and args.split == 'test' and (cfg['smoke'] or args.calibrate):
        parser.error('smoke/calibration runs cannot produce final test evidence')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
        torch.backends.cudnn.allow_tf32 = cfg['tf32']
        torch.cuda.reset_peak_memory_stats(device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'train':
        existing = any((args.out / name).exists() for name in ('sft.pt', 'reward.pt', 'ppo.pt', 'training-partial.json', 'training.json'))
        if existing and not args.resume:
            parser.error('output already contains a run; choose a fresh --out or use --resume')
        if args.resume and not (args.out / 'training-partial.json').exists():
            parser.error('--resume requires the original training-partial.json')
    metadata = provenance(args, cfg, device)
    timer = Timer(device, args.max_wall_minutes)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=args.cache, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    if args.command == 'validate-data':
        result = {}
        for split in ('train', 'dev', 'test'):
            rows = encode_rows(tokenizer, read_data(split, cfg), cfg)
            result[split] = {'examples': len(rows), 'families': len({r['family'] for r in rows}),
                'max_prompt_tokens': max(len(r['prompt_ids']) for r in rows),
                'max_response_tokens_with_eos': max(len(r[key]) for r in rows for key in ('chosen_ids', 'rejected_ids'))}
        filename = 'data-validation.json'
    elif args.command == 'train':
        rows = encode_rows(tokenizer, read_data('train', cfg), cfg)
        try:
            result = train(args, cfg, tokenizer, rows, device, metadata, timer)
        except Exception as exc:
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%f%z')
            write_json(args.out / f'failed-attempt-{stamp}.json', {'provenance': metadata,
                'error': str(exc), 'elapsed_seconds': timer.now() - timer.start,
                'completed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
            raise
        if args.calibrate:
            result['projected_main_loop_seconds'] = {
                stage: result['stage_seconds'][stage] * full_cfg[key] / cfg[key]
                for stage, key in [('sft', 'sft_steps'), ('reward', 'rm_steps'), ('ppo', 'ppo_batches')]}
            result['projection_warning'] = 'Short-run extrapolation excludes loading/checkpoint/evaluation overhead; confirm with one full GPU pilot.'
        filename = 'training.json'
    else:
        rows = encode_rows(tokenizer, read_data(args.split, cfg), cfg)
        result = evaluate(args, cfg, tokenizer, rows, device, metadata, timer)
        filename = f'evaluation-{args.split}' + ('-quick' if args.quick else '') + '.json'
    result['wall_seconds'] = timer.now() - timer.start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    result['peak_process_rss_bytes'] = None if rss is None else rss if platform.system() == 'Darwin' else rss * 1024
    result['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    result['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    result['completed_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    if args.command == 'train':
        result['attempts'][-1].update(completed_at=result['completed_at'], wall_seconds=result['wall_seconds'])
        write_json(args.out / 'training-partial.json', result)
    result.setdefault('provenance', metadata)
    write_json(args.out / filename, result)
    print(f'{filename}: {result["wall_seconds"]:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
