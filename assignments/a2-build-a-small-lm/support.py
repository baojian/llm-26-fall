"""Supplied data, evaluation, environment, and logging helpers; no assessed solutions."""
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_data(splits=("train", "dev")):
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    for filename, spec in manifest["files"].items():
        if sha256(HERE / "data" / filename) != spec["sha256"]:
            raise ValueError(f"data/tokenizer checksum mismatch: {filename}")
    tokenizer = Tokenizer.from_file(str(HERE / "data/tokenizer.json"))
    documents = {name: [json.loads(line) for line in (HERE / f"data/{name}.jsonl").read_text().splitlines()]
                 for name in splits}
    return manifest, tokenizer, documents


def token_stream(documents, tokenizer):
    """Concatenate [BOS] text [EOS] for each doc, sorted by ID. Cross-doc context is allowed."""
    ids = []
    for row in sorted(documents, key=lambda item: item["id"]):
        ids += [1] + tokenizer.encode(row["text"], add_special_tokens=False).ids + [2]
    return torch.tensor(ids, dtype=torch.long)


def sample_windows(stream, batch_size, context, generator):
    """Sample full windows uniformly over valid start positions, with replacement; no padding."""
    if stream.numel() <= context:
        raise ValueError("training stream must contain at least context+1 tokens")
    starts = torch.randint(stream.numel() - context, (batch_size,), generator=generator)
    return stream[starts[:, None] + torch.arange(context + 1)]


@torch.no_grad()
def evaluate(model, stream):
    """Score every next token exactly once in disjoint context blocks, including the short tail.

    Context resets for each block, using the same protocol for dev and final.
    This is token-weighted mean loss (nats), not a mean of unequal batch means.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, tokens = 0.0, 0
    try:
        for start in range(0, len(stream) - 1, model.config.context):
            window = stream[start:start + model.config.context + 1].to(device)
            logits = model(window[:-1].unsqueeze(0))[0]
            total += F.cross_entropy(logits, window[1:], reduction="sum").item()
            tokens += len(window) - 1
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("evaluation requires at least two tokens")
    return {"loss": total / tokens, "tokens": tokens}


def configure(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def environment():
    def git(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(HERE), *arguments], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "torch": str(torch.__version__), "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "threads": torch.get_num_threads(),
            "command": [sys.executable, *sys.argv], "git_revision": git("rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "packages": {name: version(name) for name in ("torch", "tokenizers", "numpy", "psutil")},
            "source_sha256": {str(path.relative_to(HERE)): sha256(path) for path in
                              [HERE / "run.py", HERE / "support.py", *sorted((HERE / "a2").glob("*.py"))]}}


class ResourceMeter:
    """Wall time and sampled process RSS. Sampling excludes unobserved inter-step peaks."""
    def __init__(self):
        self.started = time.perf_counter()
        self.peak = 0
        self.sample()

    def sample(self):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)

    def result(self):
        self.sample()
        return {"seconds": time.perf_counter() - self.started, "peak_rss_bytes": self.peak,
                "memory_method": "psutil process RSS sampled at setup, updates, evaluation, and finish; sampled peak"}
