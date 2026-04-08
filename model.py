"""
OligoScore model: Nucleotide Transformer embeddings + manufacturability scoring head.

Architecture:
    Sequence → Nucleotide Transformer v2 (frozen) → mean pooling → MLP → Score (0-100)

Why this design:
    - Nucleotide Transformer v2-50M was pretrained on 3.2B tokens of
      multi-species genomic data. It encodes dinucleotide context, GC
      patterns, and sequence motifs relevant to synthesis chemistry.
    - We freeze the base model: no catastrophic forgetting, fast training,
      and we can attribute predictions back through the frozen layers.
    - The MLP head is tiny (~33K params) — trains in minutes on CPU.
    - Mean pooling (masked) gives a sequence-level embedding that weighs
      all positions equally — better than [CLS] for short oligos where
      every nucleotide matters for manufacturability.

For explainability, we need access to:
    - Per-token embeddings (for integrated gradients)
    - Attention weights (for attention rollout visualization)
    Both are available because we keep the full encoder forward pass.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path

from config import PRETRAINED_MODEL, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, MODEL_DIR


class OligoScorer(nn.Module):
    """
    Manufacturability scoring model.

    Forward pass:
        1. Tokenize sequence with Nucleotide Transformer tokenizer
        2. Pass through frozen encoder → get all hidden states + attentions
        3. Mean-pool token embeddings (excluding special tokens)
        4. MLP head: Linear → ReLU → Dropout → Linear → Sigmoid × 100
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        # Load pretrained Nucleotide Transformer
        self.tokenizer = AutoTokenizer.from_pretrained(
            PRETRAINED_MODEL, trust_remote_code=True
        )
        # Use the base model from inside the MaskedLM wrapper
        mlm_model = AutoModelForMaskedLM.from_pretrained(
            PRETRAINED_MODEL, trust_remote_code=True
        )
        # Extract the backbone encoder from the MLM wrapper
        # NT v2 wraps it as .esm; fallback to .bert for other architectures
        self.encoder = mlm_model.esm if hasattr(mlm_model, 'esm') else mlm_model.bert

        # Freeze all encoder parameters — transfer learning
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Task-specific scoring head
        # Input: 512-dim mean-pooled embedding from Nucleotide Transformer
        # Output: single scalar (manufacturability score 0-100)
        self.head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, HIDDEN_DIM),  # 512 → 128
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),               # 128 → 1
        )

        self.to(device)

    def get_embeddings(self, sequences: list[str]) -> dict:
        """
        Get encoder embeddings with attention weights.

        Uses mean pooling over non-special tokens for the sequence
        representation. This is better than [CLS] pooling for short
        oligos because every position contributes to manufacturability.

        Returns dict with:
            - pooled_embedding: [batch, 512] — mean-pooled representation
            - token_embeddings: [batch, seq_len, 512] — per-token embeddings
            - attentions: tuple of [batch, heads, seq_len, seq_len] per layer
            - input_ids: [batch, seq_len] — tokenized input
            - attention_mask: [batch, seq_len]
        """
        # Tokenize — NT expects raw nucleotide strings with spaces between tokens
        # but also works with raw sequences
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Forward through frozen encoder
        with torch.no_grad():
            outputs = self.encoder(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
            )

        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]

        # Mean pooling: average all non-padding, non-special token embeddings
        # Expand mask to embedding dim for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        return {
            "pooled_embedding": pooled,
            "token_embeddings": token_embeddings,
            "attentions": outputs.attentions,
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
        }

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """
        Predict manufacturability score for a batch of sequences.

        Returns: tensor of shape [batch] with scores in range [0, 100]
        """
        emb = self.get_embeddings(sequences)
        pooled = emb["pooled_embedding"]

        # Score through MLP head
        # Sigmoid * 100 constrains output to [0, 100] range
        score = self.head(pooled).squeeze(-1)
        score = torch.sigmoid(score) * 100.0
        return score

    def forward_from_embedding(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        """
        Score from pre-computed pooled embedding.
        Used by integrated gradients to backprop through the head only.
        """
        score = self.head(pooled_embedding).squeeze(-1)
        score = torch.sigmoid(score) * 100.0
        return score

    def save(self, path: Path = None):
        """Save only the task head (encoder is pretrained, no need to save)."""
        path = path or MODEL_DIR / "scorer_head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path)
        print(f"Saved scoring head to {path}")

    def load(self, path: Path = None):
        """Load a trained task head."""
        path = path or MODEL_DIR / "scorer_head.pt"
        self.head.load_state_dict(torch.load(path, map_location=self.device))
        self.head.eval()
        print(f"Loaded scoring head from {path}")

    def parameter_count(self) -> dict:
        """Show trainable vs frozen parameter counts."""
        frozen = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.head.parameters())
        return {
            "encoder_frozen": frozen,
            "head_trainable": trainable,
            "total": frozen + trainable,
            "percent_trainable": f"{trainable / (frozen + trainable) * 100:.2f}%",
        }
