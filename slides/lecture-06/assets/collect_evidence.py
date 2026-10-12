"""Collect the small CPU experiments used by Lecture 06.

Write a NEW ignored output directory. build_assets.py turns selected evidence
into the published teaching figures; checkpoints and raw traces stay local.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch.nn import functional as F

from toy_lm import DecoderLM, NOA_DOCUMENTS, VOCABULARY
from training import RunConfig, ToyRun, digest, encode, evaluate, generate
from resume_demo import run_comparison

HERE = Path(__file__).resolve().parent


def collect(destination):
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    fit_model = DecoderLM()
    windows = torch.tensor(NOA_DOCUMENTS)
    x, target = windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
    optimizer = torch.optim.AdamW(fit_model.parameters(), lr=0.02,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    fitting = []
    fit_start = time.perf_counter()
    for step in range(201):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fit_model(x).flatten(0, 1), target.flatten())
        fitting.append({"completed": step, "nll": loss.item()})
        if step < 200:
            loss.backward()
            optimizer.step()
    fit_seconds = time.perf_counter() - fit_start
    torch.save(fit_model.state_dict(), destination / "fitted-model.pt")

    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    assert len(set(corpus["train"] + corpus["dev"])) == 24
    train_docs, dev_docs = ([encode(s) for s in corpus[part]] for part in ["train", "dev"])
    run = ToyRun(RunConfig(updates=200, warmup=20), train_docs)
    evaluation, steps = [], []
    dev_start = time.perf_counter()
    for completed in range(201):
        if completed % 10 == 0:
            evaluation.append({"completed": completed, "tokens": run.tokens,
                               "train": evaluate(run.model, train_docs),
                               "dev": evaluate(run.model, dev_docs)})
        if completed < 200:
            steps.append(run.advance())
    development_seconds = time.perf_counter() - dev_start
    run.save(destination / "development-model.pt")

    prompt = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    generations = []
    for mode, temperature, top_p in [("greedy", 1, 1), ("sample", .7, .9), ("sample", 1, 1)]:
        result = generate(fit_model, prompt, mode=mode, temperature=temperature,
                          top_p=top_p, budget=12, seed=29)
        generations.append({"mode": mode, "temperature": temperature, "top_p": top_p,
                            "seed": 29, "budget": 12, **result,
                            "text": " ".join(VOCABULARY[i] for i in result["ids"])})
    with torch.no_grad():
        logits = fit_model(torch.tensor([prompt]))[0, -1].tolist()
    resume = run_comparison(destination / "restart")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ["toy_lm.py", "training.py", "resume_demo.py", "collect_evidence.py", "corpus.json"]}
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "revision": revision, "working_tree": "uncommitted Lecture 06 sources identified by hashes",
        "sources_sha256": sources, "python": platform.python_version(),
        "torch": str(torch.__version__), "platform": platform.platform(), "cpu_threads": 2,
        "parameters": sum(p.numel() for p in fit_model.parameters()),
        "fit": {"curve": fitting, "seconds": fit_seconds, "updates": 200,
                "batch": 2, "length": 10, "tokens": 4000, "seed": 7,
                "lr": .02, "weight_decay": 0, "clip": None,
                "data_sha256": digest(NOA_DOCUMENTS)},
        "development": {"config": asdict(run.config), "curve": evaluation, "steps": steps,
                        "seconds": development_seconds, "tokens": run.tokens,
                        "train_documents": 16, "dev_documents": 8,
                        "train_sha256": digest(train_docs), "dev_sha256": digest(dev_docs),
                        "vocabulary_sha256": digest(VOCABULARY)},
        "sampling": {"prompt": prompt, "checkpoint": "200-update full-batch fit",
                     "checkpoint_sha256": hashlib.sha256((destination / "fitted-model.pt").read_bytes()).hexdigest(),
                     "next_logits": logits, "generations": generations},
        "resume": resume,
        "wall_seconds": time.perf_counter() - started,
    }
    (destination / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": result["parameters"], "fitting_final_nll": fitting[-1]["nll"],
                      "development_final": evaluation[-1], "sampling": generations,
                      "resume": resume["comparisons"], "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args().out)
