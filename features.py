"""
Heuristic feature extraction for oligonucleotide manufacturability.

Each feature maps to a real problem in solid-phase phosphoramidite synthesis.
These features serve two purposes:
  1. Generate synthetic training labels (weighted combination → score)
  2. Provide interpretable baseline features alongside the Transformer

In an interview, you should be able to explain WHY each feature matters
for synthesis yield, n-1 deletion rate, and HPLC purification.
"""

import re
import numpy as np
from typing import Dict, List, Tuple


def gc_content(seq: str) -> float:
    """
    Fraction of G and C nucleotides.

    Chemistry: G-C base pairs form 3 hydrogen bonds vs 2 for A-T.
    - Too high (>60%): oligo self-aggregates, poor solubility, secondary structure
    - Too low (<40%): weak target binding, but easier to synthesise
    - Sweet spot: 40-60% for both efficacy AND manufacturability
    """
    seq = seq.upper()
    gc = sum(1 for nt in seq if nt in "GC")
    return gc / len(seq) if seq else 0.0


def homopolymer_runs(seq: str) -> Dict[str, int]:
    """
    Longest run of each nucleotide.

    Chemistry:
    - PolyG (≥4): forms G-quadruplex structures that block coupling reagent access.
      This is the single biggest synthesis killer for ASOs.
    - PolyC (≥5): can form i-motif structures at low pH (relevant during deprotection)
    - PolyA/T (≥5): less problematic but can cause polymerase slippage in PCR probes

    Returns dict: {'A': max_run_A, 'T': max_run_T, 'G': max_run_G, 'C': max_run_C}
    """
    seq = seq.upper()
    runs = {nt: 0 for nt in "ATGC"}

    for nt in "ATGC":
        max_run = 0
        current_run = 0
        for base in seq:
            if base == nt:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        runs[nt] = max_run

    return runs


def max_homopolymer(seq: str) -> int:
    """Longest homopolymer run of any nucleotide."""
    runs = homopolymer_runs(seq)
    return max(runs.values()) if runs else 0


def self_complementarity_score(seq: str, window: int = 4) -> float:
    """
    Fraction of positions where a window is complementary to another region.

    Chemistry: Self-complementary regions form intramolecular hairpins.
    During synthesis (3'→5' direction on solid support), if the growing chain
    folds back on itself, the 5'-OH is buried and the next phosphoramidite
    can't couple. This directly increases the n-1 deletion rate.

    We check every window against every other window for complementarity.
    Returns: fraction of windows that have a complement elsewhere (0.0 to 1.0)
    """
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    seq = seq.upper()
    n = len(seq)

    if n < window * 2:
        return 0.0

    # Extract all windows
    windows = [seq[i:i + window] for i in range(n - window + 1)]

    comp_count = 0
    for i, w1 in enumerate(windows):
        # Reverse complement of w1
        rc = "".join(complement.get(nt, "N") for nt in reversed(w1))
        for j, w2 in enumerate(windows):
            # Don't compare overlapping windows
            if abs(i - j) >= window and w2 == rc:
                comp_count += 1
                break  # One match is enough for this window

    return comp_count / len(windows)


def length_penalty(seq: str) -> float:
    """
    Cumulative coupling efficiency decay.

    Chemistry: Each coupling step is ~98.5-99.5% efficient (best case).
    Cumulative yield = efficiency^(n_steps).

    At 99% per step:
      20-mer: 0.99^20 = 82% → good
      40-mer: 0.99^40 = 67% → moderate
      60-mer: 0.99^60 = 55% → challenging
      100-mer: 0.99^100 = 37% → very difficult

    Returns: estimated cumulative yield (0.0 to 1.0), assuming 99% coupling
    """
    n = len(seq)
    coupling_efficiency = 0.99
    return coupling_efficiency ** n


def dinucleotide_complexity(seq: str) -> float:
    """
    Shannon entropy of dinucleotide frequencies, normalised to 0-1.

    Chemistry: Low-complexity sequences (e.g., ATATAT, GCGCGC) produce
    n-1 deletion products with very similar hydrophobicity to the full-length
    product. This means they CO-ELUTE on HPLC, making purification extremely
    difficult. A diverse dinucleotide profile means deletions at different
    positions produce chromatographically distinct species → easier to purify.

    Higher entropy = more diverse = easier HPLC purification.
    """
    seq = seq.upper()
    if len(seq) < 2:
        return 0.0

    # Count dinucleotides
    dinucs = {}
    for i in range(len(seq) - 1):
        di = seq[i:i + 2]
        dinucs[di] = dinucs.get(di, 0) + 1

    total = sum(dinucs.values())

    # Shannon entropy
    entropy = 0.0
    for count in dinucs.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    # Normalise: max entropy for 16 possible dinucleotides = log2(16) = 4.0
    return entropy / 4.0


def terminal_penalty(seq: str) -> float:
    """
    Penalty based on 3' and 5' terminal nucleotides.

    Chemistry:
    - 3' nucleotide: pre-loaded on the controlled-pore glass (CPG) support.
      G-loaded supports have slightly lower initial loading efficiency.
    - 5' nucleotide (last coupling): G at the 5' end can form G-quartets
      with the capping failure sequences, complicating purification.

    Returns: penalty factor (0.0 = no penalty, up to 0.15)
    """
    seq = seq.upper()
    penalty = 0.0

    # 3' terminal (first base in sequence, loaded on support)
    if seq[-1] == "G":
        penalty += 0.05  # G-loaded CPG slightly less efficient

    # 5' terminal (last coupling step)
    if seq[0] == "G":
        penalty += 0.05  # G-quartet risk with failure sequences

    # Both termini G
    if seq[0] == "G" and seq[-1] == "G":
        penalty += 0.05  # Compounding risk

    return penalty


def g_quadruplex_tracts(seq: str) -> int:
    """
    Count the number of separate G-tracts (GGG+) in the sequence.

    Chemistry: A G-quadruplex requires 4 separate runs of 3+ consecutive Gs.
    These don't need to be adjacent — they fold together in 3D space,
    with the intervening bases forming loops. The classic example is the
    human telomeric repeat (TTAGGG)x4 which has exactly 4 GGG tracts.

    This catches cases that the simple polyG_run feature MISSES:
    a sequence with four GGG tracts (each only 3 Gs) passes the polyG>=4
    check but still forms a G-quadruplex.

    Returns: number of GGG+ tracts (0 = no risk, >=4 = G-quad possible)
    """
    seq = seq.upper()
    return len(re.findall(r'G{3,}', seq))


def compute_all_features(seq: str) -> Dict[str, float]:
    """
    Compute all manufacturability features for a sequence.
    Returns a dict of named features — each one is explainable.
    """
    seq = seq.upper().strip()

    runs = homopolymer_runs(seq)

    return {
        "gc_content": gc_content(seq),
        "length": len(seq),
        "length_yield": length_penalty(seq),
        "max_homopolymer": max_homopolymer(seq),
        "polyG_run": runs.get("G", 0),
        "polyC_run": runs.get("C", 0),
        "polyA_run": runs.get("A", 0),
        "polyT_run": runs.get("T", 0),
        "g_quad_tracts": g_quadruplex_tracts(seq),
        "self_complementarity": self_complementarity_score(seq),
        "dinucleotide_complexity": dinucleotide_complexity(seq),
        "terminal_penalty": terminal_penalty(seq),
    }


def heuristic_score(seq: str) -> float:
    """
    Compute a heuristic manufacturability score (0-100).

    This is our "ground truth" for synthetic training data.
    It combines all features with weights reflecting their relative
    impact on synthesis success.

    In a real system, this would be replaced by actual synthesis yield data
    from HPLC/LC-MS instruments — but the features would remain the same.
    """
    f = compute_all_features(seq)

    score = 100.0

    # --- GC content: penalise deviation from 40-60% ---
    # Weight: high. GC extremes cause cascading problems.
    gc = f["gc_content"]
    if gc < 0.40:
        score -= 15 * (0.40 - gc) / 0.40  # Up to -15 for 0% GC
    elif gc > 0.60:
        score -= 20 * (gc - 0.60) / 0.40  # Up to -20 for 100% GC (asymmetric: high GC worse)

    # --- G-quadruplex (multi-tract) ---
    # Weight: very high. G-quads from 4+ separate GGG tracts are as
    # damaging as a single long polyG run, but weren't previously caught.
    g_tracts = f["g_quad_tracts"]
    if g_tracts >= 4:
        score -= 25  # Severe: full G-quadruplex can form
    elif g_tracts == 3:
        score -= 10  # Partial: 3 tracts can form a 3-quartet stack

    # --- Homopolymer runs ---
    # Weight: very high for polyG, moderate for others.
    polyG = f["polyG_run"]
    if polyG >= 4:
        score -= 8 * (polyG - 3)  # -8 per G beyond 3 (G-quad is a synthesis killer)

    for nt in ["polyC_run", "polyA_run", "polyT_run"]:
        run_len = f[nt]
        if run_len >= 5:
            score -= 3 * (run_len - 4)  # -3 per base beyond 4

    # --- Self-complementarity ---
    # Weight: high. Hairpins directly block coupling.
    score -= 25 * f["self_complementarity"]

    # --- Length penalty ---
    # Weight: moderate. Reflects cumulative yield loss.
    score -= 20 * (1 - f["length_yield"])

    # --- Dinucleotide complexity ---
    # Weight: moderate. Affects purification, not synthesis itself.
    score -= 10 * (1 - f["dinucleotide_complexity"])

    # --- Terminal penalties ---
    # Weight: low-moderate.
    score -= 40 * f["terminal_penalty"]  # Max -6 points

    return max(0.0, min(100.0, score))


def per_position_features(seq: str) -> List[Dict[str, float]]:
    """
    Compute per-nucleotide features for position-level explainability.

    This gives us a baseline per-position signal that we can compare
    against the Transformer's attention/gradient attributions.

    Returns: list of dicts, one per nucleotide position.
    """
    seq = seq.upper()
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    n = len(seq)

    positions = []
    for i, nt in enumerate(seq):
        pos = {
            "position": i,
            "nucleotide": nt,
            "is_G": float(nt == "G"),
            "is_C": float(nt == "C"),
            "is_terminal_3": float(i == n - 1),
            "is_terminal_5": float(i == 0),
        }

        # Local homopolymer: how long is the run this position belongs to?
        run_len = 1
        j = i - 1
        while j >= 0 and seq[j] == nt:
            run_len += 1
            j -= 1
        j = i + 1
        while j < n and seq[j] == nt:
            run_len += 1
            j += 1
        pos["local_homopolymer"] = run_len

        # Local GC in a 5-nt window centred on this position
        start = max(0, i - 2)
        end = min(n, i + 3)
        window = seq[start:end]
        pos["local_gc"] = sum(1 for b in window if b in "GC") / len(window)

        positions.append(pos)

    return positions
