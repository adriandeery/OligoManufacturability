"""Test attribution on a heterogeneous sequence with diverse tokens."""
import numpy as np
from model import OligoScorer
from explain import explain_sequence

model = OligoScorer("cpu")
model.load()

# This sequence has mixed features: GC-rich region, AT-rich region, a GGG run
seq = "GCCTCAGTCTGCTTCGCACC"
print(f"Sequence: {seq} ({len(seq)} nt)")

att = explain_sequence(model, seq, method="attention")
ig = explain_sequence(model, seq, method="integrated_gradients")

print(f"\nPos  Nt  Attention  IG")
print(f"---  --  ---------  --")
for i, nt in enumerate(seq):
    a = att["nucleotide_attributions"][i]
    g = ig["nucleotide_attributions"][i]
    a_bar = "#" * int(a * 15)
    g_bar = "#" * int(g * 15)
    print(f" {i+1:2d}   {nt}   {a:.3f} {a_bar:15s}  {g:.3f} {g_bar}")

corr = np.corrcoef(att["nucleotide_attributions"], ig["nucleotide_attributions"])[0, 1]
print(f"\nCorrelation: {corr:.3f}")
print(f"Tokens: {att['tokens']}")
