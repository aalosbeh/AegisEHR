"""Gradio demo for Aegis-EHR with a green theme.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

CHECKPOINT = PROJECT_ROOT / "results" / "checkpoint.pt"

GREEN = {
    "primary":   "#2E8B57",
    "secondary": "#4CAF50",
    "light":     "#A5D6A7",
    "dark":      "#1B5E20",
}


def _try_load_model():
    try:
        import torch
        from model import AegisMiniMesh
        if CHECKPOINT.exists():
            sd = torch.load(CHECKPOINT, map_location="cpu")
            offline = sd.get("hparams", {}).get("offline", True)
            model = AegisMiniMesh(offline=offline)
            model.load_state_dict(sd["model_state"], strict=False)
            model.eval()
            return model
    except Exception as e:
        print(f"[gui] model load failed -> simulation mode ({e})")
    return None


_MODEL = None


def _simulate_agents(note_text: str):
    """Deterministic-ish per-agent confidences seeded by note hash."""
    seed = sum(ord(c) for c in note_text) or 1
    r = random.Random(seed)
    sigmas = {
        "Schema (a_S)":      round(min(0.99, 0.82 + 0.18 * r.random()), 3),
        "Reasoning (a_R)":   round(min(0.99, 0.78 + 0.20 * r.random()), 3),
        "Temporal (a_T)":    round(min(0.99, 0.72 + 0.25 * r.random()), 3),
        "Pharma (a_P)":      round(min(0.99, 0.68 + 0.30 * r.random()), 3),
        "Governance (a_G)":  round(min(0.99, 0.75 + 0.22 * r.random()), 3),
    }
    prod = 1.0
    for s in sigmas.values():
        prod *= (1.0 - 0.85 * s)
    sigma_joint = round(1.0 - prod, 3)
    return sigmas, sigma_joint


def _fake_fhir_bundle(note_text: str) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "note_excerpt": note_text[:120] + ("..." if len(note_text) > 120 else ""),
        "entry": [
            {"resourceType": "Patient", "id": "pat-001"},
            {"resourceType": "Encounter", "id": "enc-001"},
            {"resourceType": "Condition", "code": "38341003",
             "display": "Hypertension"},
            {"resourceType": "MedicationRequest",
             "medication": "lisinopril 10 mg"},
        ],
        "note": "[SIMULATED] -- this bundle is illustrative only.",
    }


def predict(note_text: str):
    if not note_text or not note_text.strip():
        return ("Please enter a clinical note.", {}, 0.0, "abstain (no input)")

    sigmas, sigma_joint = _simulate_agents(note_text)

    if sigma_joint >= 0.70:
        decision = f"EMIT (sigma_joint = {sigma_joint} >= 0.70)"
    elif sigma_joint >= 0.50:
        decision = f"WARN (sigma_joint = {sigma_joint} in [0.50, 0.70))"
    else:
        decision = f"ABSTAIN (sigma_joint = {sigma_joint} < 0.50)"

    bundle = _fake_fhir_bundle(note_text)
    bundle_text = json.dumps(bundle, indent=2)
    return bundle_text, sigmas, sigma_joint, decision


def build_interface():
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
    )

    css = f"""
    .gradio-container {{ background: #F1F8E9; }}
    h1, h2 {{ color: {GREEN['dark']}; }}
    .gr-button-primary {{ background: {GREEN['primary']} !important;
                          border-color: {GREEN['dark']} !important; }}
    .footer-attrib {{ text-align:center; color:{GREEN['dark']};
                      font-size:0.9rem; padding:0.6rem; }}
    """

    with gr.Blocks(theme=theme, css=css, title="Aegis-EHR Demo") as demo:
        gr.Markdown(
            f"# Aegis-EHR -- Decentralized Agentic Mesh\n"
            f"Paste a clinical note. The five-agent mesh "
            f"(Schema / Reasoning / Temporal / Pharma / Governance) returns a "
            f"simulated FHIR R4 bundle, per-agent confidences, and an "
            f"abstention decision."
        )
        with gr.Row():
            with gr.Column(scale=1):
                note = gr.Textbox(
                    label="Clinical note",
                    placeholder="e.g. 65 y/o M with HTN, DM2, presents with "
                                "chest pain x 2 days. Meds: lisinopril 10 mg, "
                                "metformin 500 mg. NKDA.",
                    lines=10,
                )
                btn = gr.Button("Run mesh", variant="primary")
            with gr.Column(scale=1):
                decision = gr.Textbox(label="Abstention decision")
                joint = gr.Number(label="Joint confidence (noisy-OR)",
                                  interactive=False)
                sigmas_out = gr.Label(label="Per-agent confidence")
        with gr.Row():
            fhir = gr.Code(label="FHIR R4 bundle [SIMULATED]",
                           language="json", lines=20)

        btn.click(predict,
                  inputs=note,
                  outputs=[fhir, sigmas_out, joint, decision])

        gr.HTML(
            "<div class='footer-attrib'>"
            "Aegis-EHR research demo &middot; "
            "<b>Anas Alsobeh, Utah Valley University</b> &middot; 2026"
            "</div>"
        )
    return demo


if __name__ == "__main__":
    _MODEL = _try_load_model()
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
