"""Train the Aegis-EHR mini mesh on the synthetic corpus.

Smoke-test friendly: with --max_steps 100 this should finish in <60s on CPU.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(requested: str):
    try:
        import torch
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested
    except Exception:
        return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_steps", type=int, default=100,
                    help="Stop after this many optimizer steps (smoke-test).")
    ap.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--offline", action="store_true",
                    help="Force MiniBERT fallback (no HF download).")
    ap.add_argument("--site_holdout", type=str, default=None,
                    help="Leave-one-site-out: Site-A | Site-B | Site-C")
    ap.add_argument("--max_seq_len", type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)

    # Persist hparams up front so even if torch is missing we record the run.
    hparams = vars(args).copy()
    hparams["resolved_device"] = device
    (out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    # Lazy imports so the file doesn't crash if torch isn't installed.
    try:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except Exception as e:
        print(f"[WARN] torch/tqdm not available ({e}); writing stub log only.")
        with (out_dir / "train_log.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "train_loss", "val_loss",
                        "note [SIMULATED if torch missing]"])
            w.writerow([0, 0, "", "", "torch not installed; no training done"])
        return

    from model import AegisMiniMesh
    from dataset import AegisSyntheticDataset

    # Build datasets
    train_ds = AegisSyntheticDataset(split="train", site_holdout=args.site_holdout,
                                     max_length=args.max_seq_len)
    val_ds = AegisSyntheticDataset(split="val", site_holdout=args.site_holdout,
                                   max_length=args.max_seq_len)
    if len(train_ds) == 0:
        print("[ERROR] No training data found. Run data/generate_data.py first.")
        return

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=train_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=val_ds.collate_fn) if len(val_ds) else None

    model = AegisMiniMesh(offline=args.offline).to(device)
    print(f"Model parameters: {model.num_parameters():,} on {device} "
          f"(backbone={model.backbone_kind})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    # Equal head weights by default.
    head_w = {"schema": 1.0, "reasoning": 1.0, "temporal": 1.0,
              "pharma": 1.0, "governance": 0.1}

    log_path = out_dir / "train_log.csv"
    f = log_path.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "epoch", "train_loss", "val_loss"])

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_dl, desc=f"epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            schema_y = batch["schema_y"].to(device)
            temporal_y = batch["temporal_y"].to(device)
            pharma_y = batch["pharma_y"].to(device)
            spi_y = batch["spi_y"].to(device)

            out = model(input_ids, attn)
            l_s = bce(out["schema_logits"], schema_y)
            l_r = mse(torch.sigmoid(out["reasoning_logits"]), spi_y)
            l_t = bce(out["temporal_logits"], temporal_y)
            l_p = bce(out["pharma_logits"], pharma_y)
            # Governance head: regularise norm of features (proxy task)
            l_g = (out["governance_logits"] ** 2).mean()

            loss = (head_w["schema"] * l_s +
                    head_w["reasoning"] * l_r +
                    head_w["temporal"] * l_t +
                    head_w["pharma"] * l_p +
                    head_w["governance"] * l_g)

            optim.zero_grad()
            loss.backward()
            optim.step()

            val_loss = ""
            if step % 10 == 0 and val_dl is not None:
                model.eval()
                with torch.no_grad():
                    vb = next(iter(val_dl))
                    vo = model(vb["input_ids"].to(device),
                               vb["attention_mask"].to(device))
                    v = (bce(vo["schema_logits"], vb["schema_y"].to(device)) +
                         bce(vo["temporal_logits"], vb["temporal_y"].to(device)))
                    val_loss = float(v.item())
                model.train()

            writer.writerow([step, epoch, float(loss.item()), val_loss])
            f.flush()
            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    f.close()
    elapsed = time.time() - t0

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "hparams": hparams,
        "steps": step,
    }, ckpt_path)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Steps: {step}, elapsed: {elapsed:.1f}s")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
