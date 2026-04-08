"""Test all example sequences for consistency."""
import torch
from model import OligoScorer
from features import compute_all_features, heuristic_score

model = OligoScorer("cpu")
model.load()

examples = {
    "Mipomersen-like ASO (20-mer)": "GCCTCAGTCTGCTTCGCACC",
    "Telomeric repeat (G-quad, hard)": "TTAGGGTTAGGGTTAGGGTTAGGG",
    "CpG island probe (high GC)": "GCGGCGCGCGATCGCGCGGC",
    "Standard PCR primer (easy)": "ATGCTAGCTAACGTACGATC",
    "Gene fragment (60-mer, hard)": "ATGAAAGCGATTATTGGTCTGGGTGCTTATGAGAATCTGTACTTCCAATCCGATAAAGCG",
    "Palindromic RE site (hairpin)": "ATCGATATCGATAGCTATCGATATCGAT",
    "Danvatirsen STAT3 ASO (cancer)": "GCTGAGCTCCCGGCGCCGCC",
}

for label, seq in examples.items():
    feats = compute_all_features(seq)
    h = heuristic_score(seq)
    with torch.no_grad():
        m = model.forward([seq]).item()

    # Flag inconsistencies
    flags = []
    if h >= 70 and m < 50:
        flags.append("MISMATCH: heuristic=easy, model=hard")
    if h < 50 and m >= 70:
        flags.append("MISMATCH: heuristic=hard, model=easy")

    print(f"{label}")
    print(f"  H:{h:.0f}  M:{m:.0f}  GC:{feats['gc_content']:.0%}  "
          f"MaxRun:{feats['max_homopolymer']:.0f}  GQuad:{feats['g_quad_tracts']:.0f}  "
          f"SelfComp:{feats['self_complementarity']:.0%}  Len:{feats['length']:.0f}")
    if flags:
        for f in flags:
            print(f"  *** {f}")
    print()
