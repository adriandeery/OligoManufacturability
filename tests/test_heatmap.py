"""Test the colour values for each example to verify visual differentiation."""
import numpy as np
import torch
from model import OligoScorer
from explain import explain_sequence

model = OligoScorer("cpu")
model.load()

examples = {
    "PCR primer (easy, M:88)": "ATGCTAGCTAACGTACGATC",
    "Mipomersen (easy, M:90)": "GCCTCAGTCTGCTTCGCACC",
    "Palindrome (mod, M:74)": "ATCGATATCGATAGCTATCGATATCGAT",
    "Telomeric (hard, M:52)": "TTAGGGTTAGGGTTAGGGTTAGGG",
}

spread = 0.20

for label, seq in examples.items():
    exp = explain_sequence(model, seq, method="attention")
    attrs = exp["nucleotide_attributions"]
    score = exp["score"]
    base_risk = (100 - score) / 100.0

    attr_mean = attrs.mean()
    attr_centered = attrs - attr_mean
    attr_range = max(attrs.max() - attrs.min(), 1e-9)
    attr_norm = attr_centered / (attr_range / 2)
    color_values = np.clip(base_risk + attr_norm * spread, 0.0, 1.0)

    print(f"{label}")
    print(f"  Score: {score:.0f}, Base risk: {base_risk:.2f}")
    print(f"  Color value range: {color_values.min():.2f} - {color_values.max():.2f}")
    # Show what this maps to: 0=green, 0.5=yellow, 1=red
    avg_cv = color_values.mean()
    color_name = "GREEN" if avg_cv < 0.33 else "YELLOW/ORANGE" if avg_cv < 0.66 else "RED"
    print(f"  Average color value: {avg_cv:.2f} ({color_name})")
    print(f"  Spread: {color_values.max() - color_values.min():.2f}")

    # Now test IG
    exp_ig = explain_sequence(model, seq, method="integrated_gradients")
    attrs_ig = exp_ig["nucleotide_attributions"]
    attr_mean_ig = attrs_ig.mean()
    attr_centered_ig = attrs_ig - attr_mean_ig
    attr_range_ig = max(attrs_ig.max() - attrs_ig.min(), 1e-9)
    attr_norm_ig = attr_centered_ig / (attr_range_ig / 2)
    cv_ig = np.clip(base_risk + attr_norm_ig * spread, 0.0, 1.0)

    print(f"  IG color range: {cv_ig.min():.2f} - {cv_ig.max():.2f} "
          f"(spread: {cv_ig.max() - cv_ig.min():.2f})")

    # Check if modes actually differ
    diff = np.abs(color_values - cv_ig).max()
    print(f"  Max difference between modes: {diff:.3f}")
    print()
