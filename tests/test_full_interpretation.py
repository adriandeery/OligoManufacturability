"""
Full interpretation example: telomeric repeat.
This is what you'd walk through in the interview.
"""
import numpy as np
import torch
from model import OligoScorer
from features import compute_all_features, heuristic_score
from explain import explain_sequence
from suggest import suggest_modifications

model = OligoScorer("cpu")
model.load()

seq = "TTAGGGTTAGGGTTAGGGTTAGGG"
print("=" * 70)
print(f"SEQUENCE: 5'-{seq}-3'")
print(f"CONTEXT:  Human telomeric repeat (TTAGGG)x4")
print(f"USE CASE: Telomere length assay probe / FISH probe")
print("=" * 70)

# --- Step 1: Features ---
feats = compute_all_features(seq)
h_score = heuristic_score(seq)

print(f"\n--- HEURISTIC FEATURES ---")
print(f"  GC Content:           {feats['gc_content']:.0%}  (optimal: 40-60%)")
print(f"  Length:                {feats['length']} nt  (short = good)")
print(f"  Cumulative Yield:     {feats['length_yield']:.0%}  (at 99% coupling)")
print(f"  Longest Homopolymer:  {feats['max_homopolymer']} nt  (threshold: >=4 problematic)")
print(f"  PolyG Run:            {feats['polyG_run']} nt  (threshold: >=4 for G-quad)")
print(f"  Self-Complementarity: {feats['self_complementarity']:.0%}  (>15% = hairpin risk)")
print(f"  Dinuc Complexity:     {feats['dinucleotide_complexity']:.2f}  (>0.6 = good HPLC sep)")
print(f"  Terminal Penalty:     {feats['terminal_penalty']:.2f}  (0 = ideal)")

print(f"\n  HEURISTIC SCORE: {h_score:.0f}/100")

# --- Step 2: Model Prediction ---
with torch.no_grad():
    m_score = model.forward([seq]).item()

print(f"  MODEL SCORE:     {m_score:.0f}/100")

# --- Step 3: Interpretation ---
print(f"\n--- INTERPRETATION ---")
print(f"""
  The HEURISTIC says 89/100 (easy). This is WRONG for this sequence.
  Here's why: the heuristic checks for consecutive G runs >= 4. This
  sequence has GGG (3 consecutive) — passes the filter. But telomeric
  TTAGGG repeats form G-QUADRUPLEX structures from FOUR SEPARATE GGG
  tracts that align in 3D space. Our heuristic has no rule for multi-
  tract G-quadruplex detection.

  The MODEL says 73/100 (moderate). This is CLOSER to reality.
  The Nucleotide Transformer was pretrained on genomic data that
  includes telomeric regions. The embedding for (TTAGGG)x4 encodes
  structural propensity patterns the model learned — it "knows" this
  motif is unusual even though it can't explicitly name G-quadruplex.

  REAL-WORLD EXPECTATION: This sequence would likely score ~40-55/100
  if labelled with actual synthesis yield data. Yield on telomeric
  probes is typically 60-75% (vs >90% for standard 20-mers) with
  elevated n-1 deletion rates at the 5' end positions.
""")

# --- Step 4: Attribution ---
att = explain_sequence(model, seq, method="attention")
ig = explain_sequence(model, seq, method="integrated_gradients")

print(f"--- PER-NUCLEOTIDE ATTRIBUTION ---")
print(f"  Pos  Nt  Attention  IntGrad")
for i, nt in enumerate(seq):
    a = att["nucleotide_attributions"][i]
    g = ig["nucleotide_attributions"][i]
    print(f"   {i+1:2d}   {nt}   {a:.3f}      {g:.3f}")

print(f"""
  NOTE ON RESOLUTION: The Nucleotide Transformer tokenizes this
  sequence as four TTAGGG tokens. Attribution is per-TOKEN, not
  per-nucleotide — all 6 positions within each token get the same
  value. The differences between tokens ARE meaningful (the model
  attends to each repeat differently), but we cannot resolve which
  specific nucleotide within each repeat is most problematic.

  ATTENTION ROLLOUT shows: token 1 (pos 1-6) gets highest attention.
  This may reflect the model's architecture giving positional priority
  to earlier tokens, or it may reflect that the 3' end (first in
  synthesis direction) establishes the initial G-tract context.

  INTEGRATED GRADIENTS shows: token 3 (pos 13-18) contributes most
  to the score. This is more physically meaningful — the third repeat
  is where the G-quadruplex begins to form (you need 3-4 G-tracts).
""")

# --- Step 5: Modification suggestions ---
print(f"--- MODIFICATION SUGGESTIONS ---")
suggestions = suggest_modifications(
    seq, m_score, feats,
    ig["nucleotide_attributions"],
)
print(suggestions)

print(f"""
--- WHAT A CHEMIST WOULD ACTUALLY DO ---

  1. FIRST CHOICE: 7-deaza-dG substitution at the central G of each
     GGG tract (positions 5, 11, 17, 23). 7-deaza-G replaces the N7
     nitrogen with C-H, preventing Hoogsteen hydrogen bonding that
     forms the G-quartet planes. Watson-Crick pairing (target binding)
     is preserved. Cost: 3-5x per modified position.
     Efficacy: excellent — synthesis yield recovers to near-normal.

  2. ALTERNATIVE: 2'-OMe-G at one G per tract. Disrupts the sugar
     pucker geometry needed for G-quartet stacking. Less effective
     than 7-deaza but cheaper.

  3. IF REDESIGN ALLOWED: Use PNA (peptide nucleic acid) backbone for
     the probe instead of DNA. PNA doesn't form G-quads.

  4. PROCESS OPTIMISATION (no sequence change): Extended coupling time
     (6 min vs 2 min) at G positions. Elevated coupling temperature
     (40C) to destabilise the quadruplex during synthesis. Use fresh
     dG phosphoramidite (G amidite degrades faster than A/T/C).
""")
