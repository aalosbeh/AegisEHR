"""Aegis-EHR synthetic dataset loader.

Reads data/synthetic_corpus.jsonl produced by data/generate_data.py.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_corpus.jsonl"

NUM_RESOURCES = 16
NUM_AXIOMS = 27
NUM_PHARMA = 3


def _simple_tokenize(text: str, max_length: int) -> List[int]:
    """Deterministic ASCII byte-hash tokenizer (no external deps)."""
    text = text.lower()
    ids: List[int] = []
    for word in text.split():
        h = 0
        for ch in word:
            h = (h * 131 + ord(ch)) & 0x7FFF
        # Keep in distilbert vocab range, avoid 0/100/101/102 (special toks)
        ids.append(200 + (h % 29000))
        if len(ids) >= max_length:
            break
    return ids[:max_length]


class _HashTokenizer:
    """Fallback tokenizer when transformers is unavailable."""

    pad_token_id = 0

    def __call__(self, text: str, max_length: int = 512, truncation: bool = True,
                 padding: str = "max_length", return_tensors: Optional[str] = None):
        ids = _simple_tokenize(text, max_length)
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [0] * pad
            attn = attn + [0] * pad
        result = {"input_ids": ids, "attention_mask": attn}
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result


def get_tokenizer():
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception:
        return _HashTokenizer()


class AegisSyntheticDataset(Dataset):
    """Synthetic clinical-note dataset.

    Each record has:
        note_text             str
        site                  one of {Site-A, Site-B, Site-C}
        fhir_bundle           dict (synthetic FHIR R4-style)
        entities              list[str]
        temporal_axioms_violated  list[int] in [0, 27)
        pharma_alerts         list[int] in [0, 3)
    """

    def __init__(self, jsonl_path: Optional[Path] = None,
                 site_holdout: Optional[str] = None,
                 split: str = "train",
                 train_ratio: float = 0.8,
                 max_length: int = 512,
                 tokenizer: Any = None):
        self.path = Path(jsonl_path) if jsonl_path else DATA_PATH
        self.max_length = max_length
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.records: List[Dict[str, Any]] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.records.append(json.loads(line))

        # leave-one-site-out
        if site_holdout is not None:
            if split == "test":
                self.records = [r for r in self.records if r["site"] == site_holdout]
            else:
                self.records = [r for r in self.records if r["site"] != site_holdout]
        else:
            n = len(self.records)
            cut = int(n * train_ratio)
            if split == "train":
                self.records = self.records[:cut]
            elif split in ("val", "test"):
                self.records = self.records[cut:]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return rec

    # ------------------------------------------------------------------
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attn = [], []
        schema_y = torch.zeros(len(batch), NUM_RESOURCES)
        temporal_y = torch.zeros(len(batch), NUM_AXIOMS)
        pharma_y = torch.zeros(len(batch), NUM_PHARMA)
        spi_y = torch.zeros(len(batch), 1)

        for i, rec in enumerate(batch):
            enc = self.tokenizer(
                rec["note_text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            ids = enc["input_ids"]
            am = enc["attention_mask"]
            # Some tokenizers return 2D when return_tensors='pt'; ensure flat list.
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist()
                am = am.squeeze(0).tolist()
            input_ids.append(ids)
            attn.append(am)

            # schema labels: presence of each FHIR resource type
            for r in rec["fhir_bundle"].get("entry", []):
                rt = r.get("resourceType_idx")
                if rt is not None and 0 <= rt < NUM_RESOURCES:
                    schema_y[i, rt] = 1.0

            for ax in rec.get("temporal_axioms_violated", []):
                if 0 <= ax < NUM_AXIOMS:
                    temporal_y[i, ax] = 1.0

            for p in rec.get("pharma_alerts", []):
                if 0 <= p < NUM_PHARMA:
                    pharma_y[i, p] = 1.0

            spi_y[i, 0] = float(rec.get("spi_target", 0.9))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "schema_y": schema_y,
            "temporal_y": temporal_y,
            "pharma_y": pharma_y,
            "spi_y": spi_y,
        }


if __name__ == "__main__":
    ds = AegisSyntheticDataset()
    print(f"Loaded {len(ds)} train records from {ds.path}")
    if len(ds):
        sample = ds.collate_fn([ds[0]])
        print({k: tuple(v.shape) for k, v in sample.items()})
