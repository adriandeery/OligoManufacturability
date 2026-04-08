"""
Configuration for OligoScore PoC.

Key decisions documented here so you can explain each in an interview.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "saved_models"
DATA_DIR = PROJECT_ROOT / "data"

# --- Pretrained model ---
# Nucleotide Transformer v2 (InstaDeep/NVIDIA, 2023):
#   - 50M params, trained on multi-species genomes (3.2B tokens)
#   - Standard HuggingFace BERT architecture (no custom code / triton dep)
#   - 6-mer tokenization with learned BPE — captures dinucleotide and
#     trinucleotide context relevant to coupling efficiency and structure
#   - 512-dim hidden size — smaller than DNABERT-2 but sufficient for our
#     task head, and faster for CPU inference in a PoC
#
# Why not DNABERT-2? It requires triton (GPU-only, no Windows support).
# Nucleotide Transformer gives us the same transfer learning benefit
# with standard HuggingFace loading.
PRETRAINED_MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
EMBEDDING_DIM = 512  # NT v2 50M hidden size

# --- Task head ---
# Small MLP on top of mean-pooled Nucleotide Transformer embeddings.
# We freeze the base model and only train this head — transfer learning.
# This is appropriate because:
#   1. We have limited (synthetic) training data
#   2. The base model already encodes nucleotide context we need
#   3. Training only ~66K params keeps the model interpretable
HIDDEN_DIM = 128
DROPOUT = 0.1

# --- Training ---
LEARNING_RATE = 1e-3  # Relatively high because we're only training the head
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_SAMPLES = 5000  # Synthetic sequences for training
VAL_SAMPLES = 1000

# --- Oligo constraints ---
# Typical ASO: 18-25nt, but we support up to 100nt for probes/primers
MIN_OLIGO_LEN = 15
MAX_OLIGO_LEN = 100

# --- Manufacturability heuristics ---
# These thresholds come from synthesis chemistry knowledge:
#   - Coupling efficiency drops ~0.5-1% per step, cumulative
#   - PolyG runs (>3) cause secondary structure that blocks coupling
#   - GC content outside 40-60% affects solubility and purification
#   - Self-complementary regions form hairpins that reduce yield
OPTIMAL_GC_LOW = 0.40
OPTIMAL_GC_HIGH = 0.60
MAX_HOMOPOLYMER_RUN = 3  # Runs longer than this penalise score
SELF_COMP_WINDOW = 4  # Window for checking self-complementarity

# --- Claude API ---
# Used for: given model outputs + flagged positions, suggest chemical
# modifications (phosphorothioate, 2'-OMe, 2'-MOE, LNA) to improve yield.
# NOT used for: interpreting the model's predictions (that's intrinsic).
CLAUDE_MODEL = "claude-sonnet-4-20250514"
