# OligoScore — Oligonucleotide Manufacturability Predictor

Predicts synthesis difficulty for oligonucleotide sequences before a reagent is spent.

## Architecture

```
Sequence → DNABERT-2 (frozen, 117M params) → [CLS] embedding → MLP head (50K params) → Score (0-100)
                                                  ↓
                                          Attention Rollout / Integrated Gradients
                                                  ↓
                                     Per-nucleotide attribution heatmap
```

**Pretrained backbone**: DNABERT-2, a BERT model trained on multi-species genomes with BPE tokenization. Encodes dinucleotide context, GC patterns, and sequence motifs from billions of genomic sequences.

**Task head**: Small MLP trained on synthetic data generated from known synthesis chemistry heuristics (GC content, homopolymer runs, self-complementarity, cumulative coupling yield, dinucleotide complexity).

**Explainability**: Intrinsic — Integrated Gradients and Attention Rollout provide per-nucleotide attribution without relying on an external LLM for interpretation.

**Modification suggestions**: Claude API (Sonnet) suggests chemical modifications (PS, 2'-OMe, 2'-MOE, LNA, etc.) at positions flagged by the model. Falls back to rule-based suggestions without an API key.

## Setup

```bash
# 1. Create conda environment
conda create -n oligos python=3.10 -y
conda activate oligos

# 2. Install PyTorch (CPU is fine for this PoC)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install requirements
pip install -r requirements.txt

# 4. (Optional) Set Claude API key for modification suggestions
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
# Generate synthetic training data
python data_gen.py

# Train the scoring head (~5 min on CPU first run, <1 min subsequent)
python train.py

# Launch the inference UI
streamlit run app.py
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters and thresholds with chemistry rationale |
| `features.py` | Heuristic feature extraction — each feature maps to a synthesis chemistry problem |
| `data_gen.py` | Synthetic training data generation spanning the difficulty spectrum |
| `model.py` | DNABERT-2 embeddings + MLP scoring head |
| `train.py` | Training loop with cached embeddings and early stopping |
| `explain.py` | Integrated Gradients + Attention Rollout for per-nucleotide attribution |
| `suggest.py` | Claude API modification suggestions (with rule-based fallback) |
| `app.py` | Streamlit inference page |
