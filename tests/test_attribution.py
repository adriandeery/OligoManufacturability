"""Compare attention rollout vs integrated gradients outputs."""
import numpy as np
from model import OligoScorer
from explain import explain_sequence

model = OligoScorer("cpu")
model.load()

seq = "TTAGGGTTAGGGTTAGGGTTAGGG"
print(f"Sequence: {seq}")
print(f"Length: {len(seq)}")

att = explain_sequence(model, seq, method="attention")
ig = explain_sequence(model, seq, method="integrated_gradients")

print(f"\nAttention Rollout attributions:")
for i, (nt, a) in enumerate(zip(seq, att["nucleotide_attributions"])):
    bar = "#" * int(a * 20)
    print(f"  {i+1:2d} {nt}: {a:.3f} {bar}")

print(f"\nIntegrated Gradients attributions:")
for i, (nt, a) in enumerate(zip(seq, ig["nucleotide_attributions"])):
    bar = "#" * int(a * 20)
    print(f"  {i+1:2d} {nt}: {a:.3f} {bar}")

# Check if they're actually different
corr = np.corrcoef(att["nucleotide_attributions"], ig["nucleotide_attributions"])[0, 1]
print(f"\nCorrelation between methods: {corr:.3f}")
print(f"Max attention: {att['nucleotide_attributions'].max():.3f}")
print(f"Max IG: {ig['nucleotide_attributions'].max():.3f}")

# Check token-level too
print(f"\nAttention tokens: {att['tokens']}")
print(f"Attention token attrs: {att['token_attributions'][:10]}")
print(f"IG tokens: {ig['tokens']}")
print(f"IG token attrs: {ig['token_attributions'][:10]}")
