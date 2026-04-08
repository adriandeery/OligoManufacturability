"""
Synthetic training data generator for oligo manufacturability.

Strategy: generate sequences spanning the full difficulty spectrum,
from easy-to-synthesise (balanced GC, no runs, short) to challenging
(extreme GC, polyG runs, self-complementary, long).

We deliberately bias generation to cover edge cases that the model
needs to learn about — uniform random sequences would cluster around
50% GC and miss the hard cases.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from features import compute_all_features, heuristic_score
from config import MIN_OLIGO_LEN, MAX_OLIGO_LEN, DATA_DIR


def random_sequence(length: int, gc_bias: float = 0.5) -> str:
    """
    Generate a random oligonucleotide with controlled GC content.

    gc_bias: probability of choosing G or C at each position (0.0-1.0)
    """
    seq = []
    for _ in range(length):
        if random.random() < gc_bias:
            seq.append(random.choice("GC"))
        else:
            seq.append(random.choice("AT"))
    return "".join(seq)


def sequence_with_homopolymer(length: int, run_nt: str = "G", run_len: int = 5) -> str:
    """
    Generate a sequence with a specific homopolymer run inserted.

    This ensures we have training examples of the most problematic
    synthesis pattern: polyG runs that form G-quadruplexes.
    """
    if run_len >= length:
        return run_nt * length

    # Random position for the run
    pos = random.randint(0, length - run_len)
    prefix = random_sequence(pos)
    suffix = random_sequence(length - pos - run_len)
    return prefix + (run_nt * run_len) + suffix


def self_complementary_sequence(length: int) -> str:
    """
    Generate a sequence with a palindromic/hairpin-forming region.

    Creates a stem-loop structure: XXXXXX-loop-X'X'X'X'X'X'
    where X' is the reverse complement of X.
    """
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    stem_len = min(length // 3, 8)
    loop_len = min(4, length - 2 * stem_len)
    flank_len = length - 2 * stem_len - loop_len

    stem = random_sequence(stem_len)
    loop = random_sequence(loop_len)
    rc_stem = "".join(complement[nt] for nt in reversed(stem))
    flank = random_sequence(flank_len)

    # Place the hairpin somewhere in the sequence
    hairpin = stem + loop + rc_stem
    if flank_len > 0:
        insert = random.randint(0, flank_len)
        return flank[:insert] + hairpin + flank[insert:]
    return hairpin


def generate_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced dataset covering the manufacturability spectrum.

    Distribution strategy (each category gets roughly equal representation):
      1. Easy: balanced GC, short, no problematic features
      2. Moderate: slight GC bias or moderate length
      3. Hard - GC extreme: very high or very low GC
      4. Hard - homopolymer: contains polyG/polyC runs
      5. Hard - self-complementary: hairpin-forming (moderate)
      6. Hard - extreme palindromes: very high self-complementarity
      7. Hard - long: >50nt where cumulative yield drops
      8. Mixed difficulty: random properties
    """
    random.seed(seed)
    np.random.seed(seed)

    sequences = []
    per_category = n_samples // 8

    # 1. Easy sequences: 20-25nt, balanced GC
    for _ in range(per_category):
        length = random.randint(18, 25)
        seq = random_sequence(length, gc_bias=random.uniform(0.40, 0.55))
        sequences.append(seq)

    # 2. Moderate: 25-40nt, slightly off-centre GC
    for _ in range(per_category):
        length = random.randint(25, 40)
        seq = random_sequence(length, gc_bias=random.uniform(0.30, 0.70))
        sequences.append(seq)

    # 3. Extreme GC
    for _ in range(per_category):
        length = random.randint(18, 35)
        gc_bias = random.choice([random.uniform(0.0, 0.25), random.uniform(0.75, 1.0)])
        seq = random_sequence(length, gc_bias=gc_bias)
        sequences.append(seq)

    # 4. Homopolymer runs (mostly polyG — the worst offender)
    for _ in range(per_category):
        length = random.randint(20, 40)
        run_nt = random.choices(["G", "C", "A", "T"], weights=[0.5, 0.2, 0.15, 0.15])[0]
        run_len = random.randint(4, 8)
        seq = sequence_with_homopolymer(length, run_nt, run_len)
        sequences.append(seq)

    # 5. Self-complementary / hairpin (moderate)
    for _ in range(per_category):
        length = random.randint(20, 40)
        seq = self_complementary_sequence(length)
        sequences.append(seq)

    # 6. Extreme palindromes (very high self-complementarity)
    # These are sequences built from short repeating palindromic units
    # like restriction enzyme sites (GATATC, AGATCT, etc.)
    palindromic_units = [
        "GATATC", "AGATCT", "ATCGAT", "GTATAC", "CATATG",
        "AACGTT", "ACATGT", "AGCTAG", "ATGCAT", "GACGTC",
    ]
    for _ in range(per_category):
        n_repeats = random.randint(3, 6)
        unit = random.choice(palindromic_units)
        seq = (unit * n_repeats)[:random.randint(20, 40)]
        sequences.append(seq)

    # 7. Long sequences (yield decay)
    # (was category 6 before adding extreme palindromes)
    for _ in range(per_category):
        length = random.randint(50, MAX_OLIGO_LEN)
        seq = random_sequence(length, gc_bias=random.uniform(0.35, 0.65))
        sequences.append(seq)

    # 8. Fill remainder with fully random (uniform difficulty distribution)
    remaining = n_samples - len(sequences)
    for _ in range(remaining):
        length = random.randint(MIN_OLIGO_LEN, MAX_OLIGO_LEN)
        seq = random_sequence(length, gc_bias=random.uniform(0.1, 0.9))
        sequences.append(seq)

    # Compute features and scores
    records = []
    for seq in sequences:
        feats = compute_all_features(seq)
        score = heuristic_score(seq)
        records.append({"sequence": seq, "score": score, **feats})

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def save_dataset(n_train: int = 5000, n_val: int = 1000):
    """Generate and save train/val splits."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = generate_dataset(n_train, seed=42)
    val_df = generate_dataset(n_val, seed=99)

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)

    print(f"Train: {len(train_df)} sequences, score range: "
          f"{train_df['score'].min():.1f} - {train_df['score'].max():.1f}, "
          f"mean: {train_df['score'].mean():.1f}")
    print(f"Val:   {len(val_df)} sequences, score range: "
          f"{val_df['score'].min():.1f} - {val_df['score'].max():.1f}, "
          f"mean: {val_df['score'].mean():.1f}")

    return train_df, val_df


if __name__ == "__main__":
    save_dataset()
