"""Supplied A2 experiment driver. See README.md; run from this handout directory."""
import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from a2.data_policy import deduplicate
from a2.generate import generate
from a2.model import ModelConfig, TinyLM
from a2.train import learning_rate, load_checkpoint, save_checkpoint, train_step
from support import (HERE, ResourceMeter, configure, dump, environment, evaluate, load_data,
                     sample_windows, sha256, token_stream)


def model_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_condition(config_path, condition, out, stop_after=None, resume=False):
    config = json.loads(Path(config_path).read_text())
    config["model"]["norm"] = "post" if condition == "post" else "pre"
    configure(config["seed"], config["threads"])
    execution = environment()
    meter = ResourceMeter()
    manifest, tokenizer, documents = load_data()
    source = documents["train"]
    audit = {}
    if condition == "dedup":
        source, audit = deduplicate(source)
    stream = token_stream(source, tokenizer)
    dev = token_stream(documents["dev"], tokenizer)
    model = TinyLM(ModelConfig(**config["model"]))
    initial_hash = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["max_lr"], weight_decay=config["weight_decay"])
    sampler = torch.Generator().manual_seed(config["seed"] + 1)
    checkpoint_config = {**config, "condition": condition, "data_manifest_sha256": sha256(HERE / "data/manifest.json")}
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    step = 0
    history = []
    sessions = []
    dev_history = [{"step": 0, **evaluate(model, dev)}]
    if resume:
        step = load_checkpoint(checkpoint, model, optimizer, checkpoint_config, sampler)
        previous = json.loads((out / "run.json").read_text())
        if previous["completed_steps"] != step:
            raise ValueError("checkpoint/log update count mismatch")
        history, dev_history = previous["history"], previous["dev"]
        sessions = previous.get("sessions", [{"environment": previous["environment"], "resources": previous["resources"]}])
    elif checkpoint.exists():
        raise FileExistsError(f"{checkpoint} exists; use --resume or a new output directory")
    end = config["steps"] if stop_after is None else min(stop_after, config["steps"])
    if end < step:
        raise ValueError("stop-after must not precede the resumed update")
    for index in range(step, end):
        rate = learning_rate(index, config["steps"], config["warmup_steps"], config["max_lr"], config["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        batch = sample_windows(stream, config["batch_size"], config["model"]["context"], sampler)
        metrics = train_step(model, optimizer, batch, config["max_grad_norm"])
        history.append({"step": index + 1, "lr": rate, **metrics})
        meter.sample()
        if (index + 1) % config["eval_every"] == 0 or index + 1 == end:
            measured = {"step": index + 1, **evaluate(model, dev)}
            dev_history.append(measured)
            print(f"{condition} {index+1}/{config['steps']} train={metrics['loss']:.4f} dev={measured['loss']:.4f}", flush=True)
    save_checkpoint(checkpoint, model, optimizer, end, checkpoint_config, sampler)
    sessions.append({"environment": execution, "resources": meter.result()})
    resources = {"seconds": sum(session["resources"]["seconds"] for session in sessions),
                 "peak_rss_bytes": max(session["resources"]["peak_rss_bytes"] for session in sessions),
                 "memory_method": meter.result()["memory_method"]}
    result = {"condition": condition, "config": config, "environment": execution,
              "data_manifest_sha256": sha256(HERE / "data/manifest.json"),
              "tokenizer_sha256": manifest["files"]["tokenizer.json"]["sha256"],
              "initial_model_sha256": initial_hash, "final_model_sha256": model_hash(model),
              "parameters": sum(p.numel() for p in model.parameters()), "completed_steps": end,
              "tokens_processed": end * config["batch_size"] * config["model"]["context"],
              "retained_documents": len(source), "corpus_tokens": len(stream), "removed_to_retained": audit,
              "history": history, "dev": dev_history, "resources": resources, "sessions": sessions,
              "artifact_directory": str(out.resolve()), "checkpoint_sha256": sha256(checkpoint),
              "resumed": resume}
    dump(out / "run.json", result)
    return result


def tiny_setup():
    configure(103, 1)
    config = ModelConfig(vocab_size=7, context=4, width=16, heads=4, layers=1, ff_width=32)
    model = TinyLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    corpus = torch.tensor([0, 1, 3, 4, 6, 0, 2, 3, 5, 6] * 8)
    sampler = torch.Generator().manual_seed(104)
    contract = {"model": dataclasses.asdict(config), "steps": 100, "warmup": 10, "seed": 103}
    return model, optimizer, corpus, sampler, contract


def restart_worker(stage, out):
    model, optimizer, corpus, sampler, contract = tiny_setup()
    start, end = 0, 50 if stage == "first" else 100
    if stage == "resumed":
        start = load_checkpoint(out / "half.pt", model, optimizer, contract, sampler)
    elif stage == "weights-only":
        # Supplied negative control: same future batches, weights restored, optimizer lost.
        saved = torch.load(out / "half.pt", weights_only=True)
        model.load_state_dict(saved["model"])
        sampler.set_state(saved["sampler_rng"])
        start = saved["step"]
    losses = []
    for step in range(start, end):
        optimizer.param_groups[0]["lr"] = learning_rate(step, 100, 10, 0.01)
        losses.append(train_step(model, optimizer, sample_windows(corpus, 2, 4, sampler), 1.0)["loss"])
    if stage == "first":
        save_checkpoint(out / "half.pt", model, optimizer, end, contract, sampler)
    torch.save(model.state_dict(), out / f"{stage}.pt")
    dump(out / f"{stage}.json", losses)


def restart_check(out):
    out.mkdir(parents=True, exist_ok=True)
    for stage in ("full", "first", "resumed", "weights-only"):
        subprocess.run([sys.executable, str(HERE / "run.py"), "restart-worker", "--stage", stage, "--out", str(out)], check=True)
    full = torch.load(out / "full.pt", weights_only=True)
    resumed = torch.load(out / "resumed.pt", weights_only=True)
    wrong = torch.load(out / "weights-only.pt", weights_only=True)
    losses = json.loads((out / "full.json").read_text())[50:]
    later = json.loads((out / "resumed.json").read_text())
    result = {"continuous_updates": 100, "split_updates": [50, 50], "fresh_process": True,
              "max_parameter_error": max(float((full[k] - resumed[k]).abs().max()) for k in full),
              "max_loss_error": max(abs(a - b) for a, b in zip(losses, later)),
              "weights_only_parameter_error": max(float((full[k] - wrong[k]).abs().max()) for k in full),
              "atol": 1e-7, "environment": environment(), "artifact_directory": str(out.resolve())}
    dump(out / "summary.json", result)
    return result


def diagnostics(out):
    import a2.model as model_module
    model, optimizer, _, _, _ = tiny_setup()
    ids = torch.tensor([[0, 1, 3, 4], [0, 2, 3, 5]])
    changed = ids.clone()
    changed[:, 2:] = 6
    model.eval()
    with torch.no_grad():
        prefix_error = float((model(ids)[:, :2] - model(changed)[:, :2]).abs().max())
    captured = []
    def capture(module, inputs, output):
        output.retain_grad()
        captured.append(output)
    handle = model.token.register_forward_hook(capture)
    model(ids)[0, 0, 0].backward()
    future_gradient = float(captured[0].grad[0, 1:].abs().max())
    handle.remove()
    original = model_module.scaled_dot_product_attention
    def leaking_attention(q, k, v, mask):
        return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v
    try:
        model_module.scaled_dot_product_attention = leaking_attention
        with torch.no_grad():
            leaking_error = float((model(ids)[:, :2] - model(changed)[:, :2]).abs().max())
    finally:
        model_module.scaled_dot_product_attention = original
    windows = torch.tensor([[0, 1, 3, 4, 6], [0, 2, 3, 5, 6]])
    fitting = [train_step(model, optimizer, windows, 1.0)["loss"] for _ in range(200)]
    result = {"prefix_error": prefix_error, "future_gradient": future_gradient,
              "unmasked_prefix_error": leaking_error, "tiny_fit_losses": fitting,
              "attainable_loss_floor": math.log(2) / 4,
              "floor_reason": "Two equally likely first targets after BOS; other three targets are deterministic.",
              "environment": environment()}
    dump(out / "diagnostics.json", result)
    return result


def decode_records(out):
    manifest, tokenizer, _ = load_data()
    run = json.loads((out / "baseline/run.json").read_text())
    model = TinyLM(ModelConfig(**run["config"]["model"]))
    model.load_state_dict(torch.load(out / "baseline/checkpoint.pt", weights_only=True)["model"])
    records = []
    for prompt in ("Once upon a time", "The little bird", "Lily opened the door"):
        for mode, temperature, top_p in (("greedy", 1.0, 1.0), ("sample", 0.7, 0.9), ("sample", 1.0, 1.0)):
            seed = 29
            generated = generate(model, [1] + tokenizer.encode(prompt).ids, 64, manifest["tokenizer"]["eos_id"],
                                 mode=mode, temperature=temperature, top_p=top_p,
                                 generator=torch.Generator().manual_seed(seed))
            ids = generated["tokens"]
            bigrams = list(zip(ids, ids[1:]))
            records.append({"prompt": prompt, "mode": mode, "temperature": temperature, "top_p": top_p,
                            "seed": seed, **generated, "text": tokenizer.decode(ids),
                            "repeated_bigram_fraction": 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
                            "checkpoint_sha256": run["checkpoint_sha256"]})
    return records


def finalize(out, destination):
    runs = {name: json.loads((out / name / "run.json").read_text()) for name in ("baseline", "post", "dedup")}
    baseline = runs["baseline"]
    for name, run in runs.items():
        expected = json.loads(json.dumps(baseline["config"]))
        expected["model"]["norm"] = "post" if name == "post" else "pre"
        if run["config"] != expected or run["completed_steps"] != run["config"]["steps"]:
            raise ValueError("conditions must finish the same configuration and update budget")
        for key in ("initial_model_sha256", "data_manifest_sha256", "tokenizer_sha256", "tokens_processed"):
            if run[key] != baseline[key]:
                raise ValueError(f"unmatched comparison: {key}")
        if sha256(out / name / "checkpoint.pt") != run["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from recorded run")
    final_path = out / "final-evaluation.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        if final["checkpoint_sha256"] != {name: runs[name]["checkpoint_sha256"] for name in ("baseline", "dedup")}:
            raise ValueError("final evaluation already belongs to different checkpoints")
    else:
        _, tokenizer, documents = load_data(("test",))
        test = token_stream(documents["test"], tokenizer)
        final = {"checkpoint_sha256": {}, "losses": {}, "environment": environment()}
        for name in ("baseline", "dedup"):
            model = TinyLM(ModelConfig(**runs[name]["config"]["model"]))
            model.load_state_dict(torch.load(out / name / "checkpoint.pt", weights_only=True)["model"])
            final["losses"][name] = evaluate(model, test)
            final["checkpoint_sha256"][name] = runs[name]["checkpoint_sha256"]
        dump(final_path, final)
    result = {"schema_version": 1, "diagnostics": json.loads((out / "diagnostics.json").read_text()),
              "restart": json.loads((out / "restart/summary.json").read_text()), "runs": runs,
              "decoding": decode_records(out), "final_evaluation": final}
    dump(destination, result)
    print(f"Wrote {destination}; final scores must not guide further choices.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "suite", "diagnostics", "restart", "restart-worker", "finalize"])
    parser.add_argument("--config", type=Path, default=HERE / "configs/cpu.json")
    parser.add_argument("--condition", choices=["baseline", "post", "dedup"], default="baseline")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["full", "first", "resumed", "weights-only"])
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.action == "train":
        train_condition(args.config, args.condition, args.out / args.condition, args.stop_after, args.resume)
    elif args.action == "suite":
        for condition in ("baseline", "post", "dedup"):
            train_condition(args.config, condition, args.out / condition)
        diagnostics(args.out)
        restart_check(args.out / "restart")
        print("Development suite complete. Inspect controls, then run finalize once.")
    elif args.action == "diagnostics":
        diagnostics(args.out)
    elif args.action == "restart":
        restart_check(args.out / "restart")
    elif args.action == "restart-worker":
        if args.stage is None:
            parser.error("restart-worker requires --stage")
        restart_worker(args.stage, args.out)
    else:
        configure(20261009, 2)
        finalize(args.out, args.results)


if __name__ == "__main__":
    main()
