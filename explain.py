"""
Explainability module for OligoScore.

Two complementary methods for per-nucleotide attribution:

1. Integrated Gradients (IG):
   - Gold-standard attribution method (Sundararajan et al., 2017)
   - Computes how much each dimension of the input embedding contributed
     to the final score, by integrating gradients along a path from a
     baseline (zero embedding) to the actual embedding.
   - Satisfies completeness axiom: attributions sum to (score - baseline_score)
   - We aggregate across the embedding dimension to get per-TOKEN attributions,
     then map tokens back to nucleotide positions.

2. Attention Rollout:
   - Combines multi-head, multi-layer attention into a single per-token
     importance score by recursively multiplying attention matrices.
   - Fast (no gradient computation needed).
   - Shows "what the model looked at" — complementary to IG's "what mattered."

Both return per-nucleotide scores that can be visualised as a heatmap
overlaid on the sequence — this is what biochemists will actually use.
"""

import torch
import numpy as np
from typing import Tuple, List

from model import OligoScorer


def integrated_gradients(
    model: OligoScorer,
    sequence: str,
    n_steps: int = 50,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Compute Integrated Gradients attribution for a single sequence.

    Since we use mean pooling, IG is computed over all token embeddings:
      1. Get actual token embeddings from the frozen encoder
      2. Define baseline as zero embeddings (no information)
      3. Interpolate between baseline and actual in n_steps
      4. At each step, mean-pool → score → backprop to get gradients
      5. IG = (actual - baseline) * mean(gradients)
      6. L2 norm across embedding dim → per-token attribution

    Returns:
      - attributions: array of shape [n_tokens] with per-token importance
      - tokens: list of token strings (for labelling the heatmap)
      - score: the model's predicted manufacturability score
    """
    model.eval()
    model.head.eval()

    # Get actual embeddings from frozen encoder
    encoded = model.tokenizer(
        [sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.encoder(
            **encoded,
            output_attentions=True,
            output_hidden_states=True,
        )

    # Token-level embeddings: [1, seq_len, 512]
    token_embeddings = outputs.last_hidden_state.detach().clone()
    attention_mask = encoded["attention_mask"]

    # Baseline: zero embedding (represents "no sequence information")
    baseline = torch.zeros_like(token_embeddings)

    # Interpolate and accumulate gradients
    accumulated_grads = torch.zeros_like(token_embeddings)

    for step in range(n_steps):
        alpha = step / n_steps
        interpolated = baseline + alpha * (token_embeddings - baseline)
        interpolated = interpolated.detach().requires_grad_(True)

        # Mean pool (same as model.get_embeddings)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(interpolated).float()
        sum_emb = (interpolated * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / sum_mask

        # Score through head
        score = model.forward_from_embedding(pooled)

        # Backward to get gradients w.r.t. token embeddings
        model.zero_grad()
        score.backward(retain_graph=True)

        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach()

    # Integrated gradients = (actual - baseline) * mean(gradients)
    ig = (token_embeddings.detach() - baseline) * (accumulated_grads / n_steps)

    # Aggregate across embedding dimension → per-token attribution
    # Take L2 norm across embedding dim (preserves magnitude information)
    token_attributions = torch.norm(ig.squeeze(0), dim=-1).numpy()

    # Get token strings for labelling
    tokens = model.tokenizer.convert_ids_to_tokens(
        encoded["input_ids"].squeeze(0).tolist()
    )

    # Get the actual score
    with torch.no_grad():
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / sum_mask
        actual_score = model.forward_from_embedding(pooled).item()

    # Mask out special tokens
    # Nucleotide Transformer only has <cls> at position 0 — no [SEP]/[EOS]
    token_attributions[0] = 0  # <cls>

    # Normalise to [0, 1] for visualisation
    max_attr = token_attributions.max()
    if max_attr > 0:
        token_attributions = token_attributions / max_attr

    return token_attributions, tokens, actual_score


def attention_rollout(
    model: OligoScorer,
    sequence: str,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Compute attention rollout for a single sequence.

    Method (Abnar & Zuidema, 2020):
      1. Get attention matrices from all layers: [n_layers, n_heads, seq, seq]
      2. Average across heads within each layer
      3. Add identity matrix (residual connections)
      4. Normalise rows
      5. Recursively multiply layer-by-layer
      6. Average all rows → mean attention each token receives

    For mean-pooled models we average ALL rows (not just [CLS]),
    since every token contributes equally to the pooled representation.
    """
    model.eval()

    with torch.no_grad():
        emb = model.get_embeddings([sequence])
        score = model.forward([sequence]).item()

    attentions = emb["attentions"]  # Tuple of [1, n_heads, seq, seq]
    attention_mask = emb["attention_mask"].squeeze(0).numpy()

    # Stack and average across heads
    att_matrices = []
    for layer_att in attentions:
        avg_att = layer_att.squeeze(0).mean(dim=0).cpu().numpy()
        att_matrices.append(avg_att)

    # Attention rollout
    rollout = np.eye(att_matrices[0].shape[0])
    for att in att_matrices:
        att_with_residual = 0.5 * att + 0.5 * np.eye(att.shape[0])
        att_with_residual = att_with_residual / att_with_residual.sum(axis=-1, keepdims=True)
        rollout = rollout @ att_with_residual

    # For mean-pooled models: average attention across all token rows
    # This represents "how much attention does each token receive on average?"
    token_attention = rollout.mean(axis=0)

    # Get tokens
    encoded = model.tokenizer(
        [sequence], return_tensors="pt", padding=True, truncation=True,
    )
    tokens = model.tokenizer.convert_ids_to_tokens(
        encoded["input_ids"].squeeze(0).tolist()
    )

    # Mask special tokens
    # Nucleotide Transformer only has <cls> at position 0 — no [SEP]/[EOS]
    token_attention[0] = 0  # <cls>

    # Normalise
    max_att = token_attention.max()
    if max_att > 0:
        token_attention = token_attention / max_att

    return token_attention, tokens, score


def map_tokens_to_nucleotides(
    attributions: np.ndarray,
    tokens: List[str],
    sequence: str,
) -> np.ndarray:
    """
    Map token-level attributions back to per-nucleotide attributions.

    Nucleotide Transformer uses k-mer tokenization, so one token may
    cover multiple nucleotides (e.g., "TTAGGG" is a single token).

    We assign the token's attribution to ALL nucleotides it covers
    WITHOUT dividing by token length. This avoids the artefact where
    single-character remainder tokens (e.g., trailing "C") appear
    disproportionately important simply because they aren't diluted.

    The final normalisation to [0,1] handles the scaling.

    Returns: array of shape [len(sequence)] with per-nucleotide attributions.
    """
    seq = sequence.upper()
    nuc_attributions = np.zeros(len(seq))

    seq_pos = 0

    for tok_idx, token in enumerate(tokens):
        # Skip special tokens (various tokenizer conventions)
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>",
                     "<cls>", "<eos>", "<unk>", "<mask>"):
            continue

        # Clean the token (remove BPE/k-mer markers)
        clean_token = (token.replace("##", "")
                           .replace("Ġ", "")
                           .replace("▁", "")
                           .upper())

        # Keep only nucleotide characters
        clean_token = "".join(c for c in clean_token if c in "ATGC")

        if not clean_token:
            continue

        tok_len = len(clean_token)

        # Assign the token's full attribution to all covered nucleotides.
        # We do NOT divide by tok_len — that creates artefacts where
        # short remainder tokens dominate the heatmap.
        if seq_pos + tok_len <= len(seq):
            for j in range(tok_len):
                nuc_attributions[seq_pos + j] = attributions[tok_idx]
            seq_pos += tok_len

    # Normalise to [0, 1]
    max_val = nuc_attributions.max()
    if max_val > 0:
        nuc_attributions = nuc_attributions / max_val

    return nuc_attributions


def explain_sequence(
    model: OligoScorer,
    sequence: str,
    method: str = "attention",
) -> dict:
    """
    Full explainability pipeline for a single sequence.

    Returns a dict with everything needed for visualisation:
      - score: predicted manufacturability (0-100)
      - nucleotide_attributions: per-position importance [0-1]
      - tokens: token strings
      - token_attributions: per-token importance [0-1]
      - method: which method was used
    """
    if method == "integrated_gradients":
        token_attrs, tokens, score = integrated_gradients(model, sequence)
    else:
        token_attrs, tokens, score = attention_rollout(model, sequence)

    nuc_attrs = map_tokens_to_nucleotides(token_attrs, tokens, sequence)

    return {
        "score": score,
        "sequence": sequence.upper(),
        "nucleotide_attributions": nuc_attrs,
        "tokens": tokens,
        "token_attributions": token_attrs,
        "method": method,
    }
