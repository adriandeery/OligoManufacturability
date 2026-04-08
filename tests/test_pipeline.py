"""Full pipeline smoke test."""
from model import OligoScorer
from features import compute_all_features, heuristic_score
from explain import explain_sequence
from suggest import suggest_modifications
import numpy as np

print("=" * 60)
print("OligoScore Pipeline Test")
print("=" * 60)

# Load trained model
model = OligoScorer("cpu")
model.load()

test_seqs = {
    "Easy (balanced 20-mer)": "ATCGATCGATCGATCGATCG",
    "Hard (polyG run)":       "ATCGATGGGGGCTAGCATCG",
    "Hard (high GC)":         "GCGCGCGCGCGCGCGCGCGC",
    "Moderate (30-mer)":      "ATCGATCGATCGATCGATCGATCGATCGAT",
}

for label, seq in test_seqs.items():
    print(f"\n--- {label}: {seq} ---")

    # Features
    feats = compute_all_features(seq)
    h_score = heuristic_score(seq)

    # Model prediction
    import torch
    with torch.no_grad():
        m_score = model.forward([seq]).item()

    print(f"  Heuristic score: {h_score:.1f}")
    print(f"  Model score:     {m_score:.1f}")
    print(f"  GC: {feats['gc_content']:.1%}, MaxRun: {feats['max_homopolymer']}, "
          f"SelfComp: {feats['self_complementarity']:.1%}")

    # Explainability (attention rollout — fast)
    explanation = explain_sequence(model, seq, method="attention")
    attrs = explanation["nucleotide_attributions"]
    top3 = np.argsort(attrs)[-3:][::-1]
    print(f"  Top attributed positions: {[f'{i+1}({seq[i]})' for i in top3]}")

    # Modification suggestions (fallback, no API key)
    suggestions = suggest_modifications(seq, m_score, feats, attrs)
    first_line = suggestions.split('\n')[0]
    print(f"  Suggestion: {first_line}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
