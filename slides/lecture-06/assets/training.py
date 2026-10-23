"""Small, inspectable training utilities for Lecture 06's supplied toy model."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from toy_lm import DecoderLM, NOA_DOCUMENTS, VOCABULARY


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def encode(text):
    return [VOCABULARY.index(word) for word in text.split()] + [11]


def shifted(document):
    """One document, with EOS already appended; never cross a document boundary."""
    if len(document) < 2 or len(document) > 11:
        raise ValueError("A document needs 2–11 tokens including EOS.")
    window = torch.tensor([document], dtype=torch.long)
    return window[:, :-1].contiguous(), window[:, 1:].contiguous()


def lr_at(update, total=40, warmup=4, maximum=0.02, minimum_ratio=0.1):
    """Zero-based update schedule; the last applied update reaches the minimum."""
    if not 0 < warmup < total or not 0 <= update < total:
        raise ValueError("Require 0 < warmup < total and 0 <= update < total.")
    if not math.isfinite(maximum) or maximum <= 0 or not 0 <= minimum_ratio <= 1:
        raise ValueError("Invalid learning-rate limits.")
    if update < warmup:
        return maximum * (update + 1) / warmup
    progress = (update - warmup + 1) / (total - warmup)
    return maximum * (minimum_ratio + (1 - minimum_ratio)
                      * (1 + math.cos(math.pi * progress)) / 2)


@torch.no_grad()
def evaluate(model, documents):
    """Token-weighted NLL; restore the caller's training/evaluation mode."""
    if not documents:
        raise ValueError("Evaluation needs at least one document.")
    was_training = model.training
    model.eval()
    total_nll, tokens = 0.0, 0
    try:
        for document in documents:
            x, target = shifted(document)
            total_nll += F.cross_entropy(model(x).flatten(0, 1), target.flatten(),
                                        reduction="sum").item()
            tokens += target.numel()
    finally:
        model.train(was_training)
    return {"nll": total_nll / tokens, "tokens": tokens}


@dataclass(frozen=True)
class RunConfig:
    updates: int = 40
    warmup: int = 4
    maximum_lr: float = 0.02
    minimum_ratio: float = 0.1
    clip: float = 1.0
    seed: int = 7
    sampler_seed: int = 17
    threads: int = 2


class ToyRun:
    """A CPU run with a single randomly chosen document per optimizer update."""

    def __init__(self, config=RunConfig(), documents=None):
        lr_at(0, config.updates, config.warmup, config.maximum_lr, config.minimum_ratio)
        if config.clip <= 0 or not math.isfinite(config.clip) or config.threads < 1:
            raise ValueError("Invalid clipping threshold or CPU thread count.")
        self.config = config
        self.documents = NOA_DOCUMENTS if documents is None else documents
        if not self.documents:
            raise ValueError("Training needs documents.")
        for document in self.documents:
            shifted(document)
        torch.set_num_threads(config.threads)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = DecoderLM().cpu().float()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.maximum_lr, betas=(0.9, 0.999),
            eps=1e-8, weight_decay=0.0,
        )
        self.sampler = torch.Generator(device="cpu").manual_seed(config.sampler_seed)
        self.completed = 0
        self.tokens = 0

    def advance(self):
        if self.completed >= self.config.updates:
            raise ValueError("This run has completed its planned updates.")
        batch_id = int(torch.randint(len(self.documents), (), generator=self.sampler))
        x, target = shifted(self.documents[batch_id])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(x).flatten(0, 1), target.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.clip, error_if_nonfinite=True,
        )
        rate = lr_at(self.completed, self.config.updates, self.config.warmup,
                     self.config.maximum_lr, self.config.minimum_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()
        record = {"update": self.completed, "batch_id": batch_id,
                  "lr": rate, "nll": loss.item(), "grad_norm": norm.item()}
        self.completed += 1
        self.tokens += target.numel()
        record.update(completed=self.completed, tokens=self.tokens)
        return record

    def identity(self):
        return {"schema": 1, "torch": str(torch.__version__), "device": "cpu",
                "dtype": "float32", "config": asdict(self.config),
                "data_sha256": digest(self.documents),
                "vocabulary_sha256": digest(VOCABULARY),
                "model": "l05-noa-pre-ln-4608-v1"}

    def save(self, path):
        path = Path(path)
        payload = {"identity": self.identity(), "model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "completed": self.completed,
                   "tokens": self.tokens, "sampler_rng": self.sampler.get_state(),
                   "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, path, *, reset_optimizer=False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["identity"] != self.identity():
            raise ValueError("Checkpoint config, data, model, device, or version differs.")
        if not 0 <= payload["completed"] <= self.config.updates:
            raise ValueError("Invalid checkpoint update count.")
        self.model.load_state_dict(payload["model"])
        if not reset_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.completed, self.tokens = payload["completed"], payload["tokens"]
        # Restore after constructors and immediately before the next random draw.
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        self.sampler.set_state(payload["sampler_rng"])


def distribution(logits, temperature=1.0, top_p=1.0):
    """CPU probability vector: temperature, inclusive nucleus, renormalization."""
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("Supply a finite nonempty vector of logits.")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require finite temperature > 0 and 0 < top_p <= 1.")
    probs = torch.softmax(logits.double().cpu() / temperature, dim=0)
    if top_p == 1:
        return probs
    order = torch.argsort(probs, descending=True, stable=True)
    cumulative = probs[order].cumsum(0)
    end = min(int(torch.searchsorted(cumulative, top_p)) + 1, probs.numel())
    result = torch.zeros_like(probs)
    retained = order[:end]
    result[retained] = probs[retained]
    return result / result.sum()


@torch.no_grad()
def generate(model, prompt, *, budget=12, mode="sample", temperature=1.0,
             top_p=1.0, seed=29):
    if not prompt or budget < 0 or mode not in {"greedy", "sample"}:
        raise ValueError("Require a prompt, nonnegative budget, and a known mode.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sequence, new_tokens = list(prompt), []
    was_training = model.training
    model.eval()
    try:
        for _ in range(budget):
            # Learned positions restart at zero for each cropped context.
            x = torch.tensor([sequence[-model.max_length:]], dtype=torch.long)
            logits = model(x)[0, -1]
            if mode == "greedy":
                token = int(logits.argmax())  # first/smallest ID breaks ties
            else:
                probs = distribution(logits, temperature, top_p)
                token = int(torch.multinomial(probs, 1, generator=generator))
            sequence.append(token)
            new_tokens.append(token)
            if token == 11:
                return {"ids": new_tokens, "stop": "eos"}
    finally:
        model.train(was_training)
    return {"ids": new_tokens, "stop": "length"}
