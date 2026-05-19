"""Aegis-EHR mini mesh model.

Five lightweight task-specific heads on top of a shared encoder
(distilbert-base-uncased ~66M params, well under the <3B paper limit).
If transformers cannot be downloaded (offline), a MiniBERT fallback with
random initialization is used for smoke testing.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MiniBERT(nn.Module):
    """Small transformer encoder used when transformers is unavailable.

    NOT trained -- only for smoke-testing the pipeline.
    """

    def __init__(self, vocab_size: int = 30522, hidden_size: int = 128,
                 num_layers: int = 2, num_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = None
        if attention_mask is not None:
            mask = (attention_mask == 0)  # True where padded
        h = self.encoder(x, src_key_padding_mask=mask)
        return h  # (B, T, H)


def _try_load_distilbert():
    """Try to load distilbert-base-uncased. Return (model, hidden_size) or None."""
    try:
        from transformers import AutoModel  # type: ignore
        m = AutoModel.from_pretrained("distilbert-base-uncased")
        return m, m.config.hidden_size
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-agent heads
# ---------------------------------------------------------------------------
class SchemaHead(nn.Module):
    """a_S: emits FHIR R4 resource graph (multi-label over resource types)."""

    def __init__(self, hidden_size: int, num_resources: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_resources),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class ReasoningHead(nn.Module):
    """a_R: produces an SPI score in [0,1]."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class TemporalHead(nn.Module):
    """a_T: multi-label classification over 27 temporal axioms."""

    def __init__(self, hidden_size: int, num_axioms: int = 27):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_axioms),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class PharmaHead(nn.Module):
    """a_P: binary DDI/allergy/contraindication alert."""

    def __init__(self, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class GovernanceHead(nn.Module):
    """a_G: emits a feature vector summarising distribution drift signals."""

    def __init__(self, hidden_size: int, feat_dim: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, feat_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Aegis mini mesh
# ---------------------------------------------------------------------------
class AegisMiniMesh(nn.Module):
    """Shared encoder + five task heads.

    Parameter budget: roughly 66M (DistilBERT) + a few hundred K from heads,
    well under the 500M soft cap mentioned in the paper for the mini variant.
    """

    def __init__(self, offline: bool = False, hidden_size: int = 128,
                 vocab_size: int = 30522, max_len: int = 512):
        super().__init__()
        self.offline = offline
        backbone = None if offline else _try_load_distilbert()
        if backbone is None:
            # CPU-degraded / offline mode
            self.backbone_kind = "mini"
            self.backbone = MiniBERT(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=2,
                num_heads=4,
                max_len=max_len,
            )
            h = hidden_size
        else:
            self.backbone_kind = "distilbert"
            self.backbone, h = backbone

        self.hidden_size = h
        self.schema_head = SchemaHead(h)
        self.reason_head = ReasoningHead(h)
        self.temporal_head = TemporalHead(h)
        self.pharma_head = PharmaHead(h)
        self.gov_head = GovernanceHead(h)

    # ------------------------------------------------------------------
    def _encode(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.backbone_kind == "distilbert":
            out = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
            h = out.last_hidden_state  # (B, T, H)
        else:
            h = self.backbone(input_ids, attention_mask)
        # mean-pool over non-padding tokens
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        pooled = self._encode(input_ids, attention_mask)
        return {
            "features": pooled,
            "schema_logits": self.schema_head(pooled),
            "reasoning_logits": self.reason_head(pooled),
            "temporal_logits": self.temporal_head(pooled),
            "pharma_logits": self.pharma_head(pooled),
            "governance_logits": self.gov_head(pooled),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def temperature_scale(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Divide logits by temperature T before softmax/sigmoid.

    Used for the calibration stage; the paper reports ECE 0.041 after
    temperature scaling.
    """
    T = max(float(T), 1e-3)
    return logits / T


def joint_confidence(per_agent_sigma: Dict[str, float],
                     weights: Optional[Dict[str, float]] = None) -> float:
    """Noisy-OR joint confidence: 1 - prod(1 - w_a * sigma_a)."""
    if weights is None:
        weights = {k: 1.0 for k in per_agent_sigma}
    prod = 1.0
    for k, sigma in per_agent_sigma.items():
        w = float(weights.get(k, 1.0))
        s = min(max(float(sigma), 0.0), 1.0)
        prod *= (1.0 - w * s)
    return 1.0 - prod


def abstention_decision(sigma_joint: float,
                        emit_thresh: float = 0.70,
                        warn_thresh: float = 0.50) -> str:
    if sigma_joint >= emit_thresh:
        return "emit"
    if sigma_joint >= warn_thresh:
        return "warn"
    return "abstain"


if __name__ == "__main__":
    # Smoke test: instantiate in offline mode and check shapes.
    model = AegisMiniMesh(offline=True)
    x = torch.randint(0, 30522, (2, 32))
    mask = torch.ones_like(x)
    out = model(x, mask)
    print({k: tuple(v.shape) for k, v in out.items()})
    print(f"Params: {model.num_parameters():,}")
