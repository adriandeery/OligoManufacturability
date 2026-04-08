"""
OligoScore — Oligonucleotide Manufacturability Predictor

Streamlit inference page. This is what you'd demo in the interview.

Layout:
  1. Sequence input (text box + example sequences)
  2. Score gauge (0-100 with color coding)
  3. Per-nucleotide attribution heatmap
  4. Feature breakdown table
  5. Modification suggestions (Claude API or fallback)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import torch
import os

from model import OligoScorer
from features import compute_all_features, heuristic_score
from explain import explain_sequence
from suggest import suggest_modifications
from config import MODEL_DIR


# --- Page config ---
st.set_page_config(
    page_title="OligoScore",
    page_icon="🧬",
    layout="wide",
)

st.title("OligoScore — Manufacturability Predictor")
st.markdown(
    "Predict synthesis difficulty for oligonucleotide sequences using "
    "a pretrained Nucleotide Transformer backbone with a trained scoring head. "
    "Per-nucleotide attributions show *which positions* drive the score."
)


# --- Load model (cached) ---
@st.cache_resource
def load_model():
    """Load the trained model. Cached so it only loads once."""
    model = OligoScorer(device="cpu")
    model_path = MODEL_DIR / "scorer_head.pt"
    if model_path.exists():
        model.load(model_path)
        return model, True
    else:
        st.warning(
            "No trained model found. Run `python train.py` first. "
            "Showing heuristic scores only."
        )
        return model, False


model, model_trained = load_model()


# --- Sidebar: example sequences ---
st.sidebar.header("Example Sequences")
examples = {
    # --- Known easy: balanced GC, no runs, short ---
    "Mipomersen-like ASO (20-mer)": "GCCTCAGTCTGCTTCGCACC",
    # Mipomersen (Kynamro) targets ApoB-100 mRNA. FDA-approved 2013.
    # Well-characterised synthesis — 20-mer, 65% GC (slightly high),
    # no polyG runs. This is a known-manufacturable gapmer design.

    # --- Known difficult: polyG run ---
    "Telomeric repeat (polyG, hard)": "TTAGGGTTAGGGTTAGGGTTAGGG",
    # Human telomeric repeat (TTAGGG)x4. Known to form G-quadruplex
    # structures during synthesis. Every oligo vendor flags these.
    # The GGG runs cause coupling failures and n-1 deletions.

    # --- Known moderate: high GC ---
    "CpG island probe (high GC)": "GCGGCGCGCGATCGCGCGGC",
    # CpG island targeting — 90% GC. Common in epigenetics research.
    # Self-aggregation and secondary structure make this hard to
    # synthesise at scale, but short length partially compensates.

    # --- Known easy: standard PCR primer ---
    "Standard PCR primer (easy)": "ATGCTAGCTAACGTACGATC",
    # Typical 20-mer primer. 50% GC, no runs, no hairpins.
    # Any contract manufacturer would make this at >90% purity.

    # --- Known difficult: long oligo ---
    "Gene fragment (60-mer, hard)": "ATGAAAGCGATTATTGGTCTGGGTGCTTATGAGAATCTGTACTTCCAATCCGATAAAGCG",
    # 60-mer gene fragment for Gibson assembly. At 99% coupling,
    # cumulative yield ~55%. Requires careful HPLC or PAGE purification.
    # The length is the primary challenge, not sequence composition.

    # --- Known difficult: self-complementary ---
    "Palindromic RE site (hairpin)": "ATCGATATCGATAGCTATCGATATCGAT",
    # Contains multiple EcoRV-like palindromes (GATATC). The repeated
    # self-complementary motif forms intramolecular hairpins during
    # synthesis, blocking coupling at the stem positions.

    # --- Cancer therapeutic: Danvatirsen (AZD9150) ---
    "Danvatirsen STAT3 ASO (cancer)": "GCTGAGCTCCCGGCGCCGCC",
    # Anti-STAT3 ASO (AstraZeneca). Phase II for DLBCL, NSCLC, and
    # head/neck cancers. 16-mer cEt gapmer, 3-10-3 design.
    # Base sequence shown here. High GC (80%) is the synthesis
    # challenge — real drug uses constrained ethyl (cEt) modifications.
}

selected_example = st.sidebar.selectbox(
    "Load example sequence:",
    ["(none)"] + list(examples.keys()),
)

# --- Main input ---
default_seq = ""
if selected_example != "(none)":
    default_seq = examples[selected_example]

sequence = st.text_input(
    "Enter oligonucleotide sequence (DNA: A, T, G, C):",
    value=default_seq,
    max_chars=200,
    placeholder="e.g., ATCGATCGATCG",
).upper().strip()

# Validate
valid = True
if sequence:
    invalid_chars = set(sequence) - set("ATGC")
    if invalid_chars:
        st.error(f"Invalid characters: {invalid_chars}. Only A, T, G, C allowed.")
        valid = False
    elif len(sequence) < 10:
        st.error("Sequence too short (minimum 10 nt).")
        valid = False


# --- Run prediction ---
if sequence and valid:
    st.divider()

    # Compute heuristic features (always available)
    features = compute_all_features(sequence)
    h_score = heuristic_score(sequence)

    # Layout: score on left, features on right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Manufacturability Score")

        if model_trained:
            # Model prediction
            with torch.no_grad():
                model_score = model.forward([sequence]).item()

            # Show both scores
            score_color = (
                "#2ecc71" if model_score >= 70
                else "#f39c12" if model_score >= 40
                else "#e74c3c"
            )
            st.markdown(
                f"<h1 style='color:{score_color}; text-align:center;'>"
                f"{model_score:.0f}/100</h1>",
                unsafe_allow_html=True,
            )
            st.caption(f"Heuristic baseline: {h_score:.0f}/100")

            difficulty = (
                "Easy — standard synthesis"
                if model_score >= 70
                else "Moderate — may need optimisation"
                if model_score >= 40
                else "Challenging — expect low yield"
            )
            st.info(f"**{difficulty}**")
        else:
            # Fallback to heuristic
            score_color = (
                "#2ecc71" if h_score >= 70
                else "#f39c12" if h_score >= 40
                else "#e74c3c"
            )
            st.markdown(
                f"<h1 style='color:{score_color}; text-align:center;'>"
                f"{h_score:.0f}/100</h1>",
                unsafe_allow_html=True,
            )
            st.caption("(Heuristic score — train model for ML prediction)")

    with col2:
        st.subheader("Feature Breakdown")
        feature_display = {
            "GC Content": f"{features['gc_content']:.1%}",
            "Sequence Length": f"{features['length']} nt",
            "Cumulative Yield Est.": f"{features['length_yield']:.1%}",
            "Longest Homopolymer": f"{features['max_homopolymer']:.0f} nt",
            "PolyG Run": f"{features['polyG_run']:.0f} nt",
            "PolyC Run": f"{features['polyC_run']:.0f} nt",
            "PolyA Run": f"{features['polyA_run']:.0f} nt",
            "PolyT Run": f"{features['polyT_run']:.0f} nt",
            "G-Quad Tracts (GGG+)": f"{features['g_quad_tracts']:.0f}",
            "Self-Complementarity": f"{features['self_complementarity']:.1%}",
            "Dinucleotide Complexity": f"{features['dinucleotide_complexity']:.2f}",
            "Terminal Penalty": f"{features['terminal_penalty']:.2f}",
        }
        # Two-column feature table
        fcol1, fcol2 = st.columns(2)
        items = list(feature_display.items())
        for i, (name, val) in enumerate(items):
            if i < len(items) // 2:
                fcol1.metric(name, val)
            else:
                fcol2.metric(name, val)

    # --- Attribution heatmap ---
    st.divider()
    st.subheader("Per-Nucleotide Attribution")

    if model_trained:
        method = st.radio(
            "Attribution method:",
            ["Attention Rollout", "Integrated Gradients"],
            horizontal=True,
            help="Attention Rollout: fast, shows what the model attended to. "
                 "Integrated Gradients: slower, mathematically rigorous attribution.",
        )
        method_key = (
            "attention" if method == "Attention Rollout"
            else "integrated_gradients"
        )

        with st.spinner(f"Computing {method}..."):
            explanation = explain_sequence(model, sequence, method=method_key)

        attributions = explanation["nucleotide_attributions"]

        # Colour strategy: the SCORE sets a base risk level, and
        # ATTRIBUTION modulates which positions are worse/better.
        #
        # base_risk: where on the green→red scale the sequence starts
        #   score 95 → base 0.05 (green)
        #   score 50 → base 0.50 (yellow)
        #   score 20 → base 0.80 (red)
        #
        # Each position then shifts ±spread based on its attribution:
        #   high attribution → pushed toward red (worse than average)
        #   low attribution  → pushed toward green (better than average)
        #
        # This means:
        #   Easy sequence:  all positions are green, slight variation
        #   Hard sequence:  base is orange, high-attribution positions are red
        base_risk = (100 - explanation["score"]) / 100.0
        spread = 0.20  # How much attribution can shift the colour

        # Centre attributions around 0 (subtract mean so relative diffs show)
        attr_mean = attributions.mean()
        attr_centered = attributions - attr_mean  # range roughly [-0.5, +0.5]
        # Normalise to [-1, +1]
        attr_range = max(attributions.max() - attributions.min(), 1e-9)
        attr_norm = attr_centered / (attr_range / 2)

        color_values = np.clip(base_risk + attr_norm * spread, 0.0, 1.0)

        # Heatmap visualisation
        fig, ax = plt.subplots(figsize=(max(12, len(sequence) * 0.4), 2))

        # Color map: green (0) → yellow (0.5) → red (1)
        cmap = plt.cm.RdYlGn_r

        for i, (nt, cv) in enumerate(zip(sequence, color_values)):
            color = cmap(cv)
            rect = FancyBboxPatch(
                (i, 0), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                i + 0.45, 0.45, nt,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if cv > 0.6 else "black",
            )

        ax.set_xlim(-0.1, len(sequence) + 0.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Attribution Heatmap ({method}) — "
            f"Green = low synthesis risk, Red = high synthesis risk",
            fontsize=11,
        )

        # Add position numbers every 5 nt
        for i in range(0, len(sequence), 5):
            ax.text(i + 0.45, -0.2, str(i + 1), ha="center", fontsize=7, color="gray")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Each box is one nucleotide. Colour reflects synthesis risk at that "
            "position: green = low risk, red = high risk. The heatmap is scaled "
            "by the overall score — easy sequences appear mostly green, "
            "difficult sequences highlight the problematic regions in red."
        )
    else:
        st.info(
            "Train the model (`python train.py`) to enable per-nucleotide attributions."
        )

    # --- Modification suggestions ---
    st.divider()
    st.subheader("Modification Suggestions")

    score_for_suggestions = model_score if model_trained else h_score
    attrs_for_suggestions = (
        attributions if model_trained
        else np.ones(len(sequence)) * 0.5
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_api = st.checkbox(
        "Use Claude API for suggestions",
        value=bool(api_key),
        disabled=not bool(api_key),
        help="Set ANTHROPIC_API_KEY environment variable to enable. "
             "Falls back to rule-based suggestions otherwise.",
    )

    suggestions = suggest_modifications(
        sequence=sequence,
        score=score_for_suggestions,
        features=features,
        attributions=attrs_for_suggestions,
        api_key=api_key if use_api else None,
    )

    st.markdown(suggestions)


# --- Footer ---
st.divider()
st.caption(
    "**OligoScore PoC** — Pretrained Nucleotide Transformer backbone with manufacturability scoring head. "
    "Explainability via Integrated Gradients and Attention Rollout. "
    "Modification suggestions via Claude API or rule-based fallback."
)
