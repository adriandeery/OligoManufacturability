"""
Training script for OligoScore.

Key design: we pre-compute Nucleotide Transformer embeddings for the entire dataset ONCE,
then train only the MLP head on cached embeddings. This means:
  - No GPU needed (embedding extraction is a one-time cost)
  - Training is fast (~seconds per epoch on CPU)
  - We can iterate on the head architecture without re-encoding

This is a standard transfer learning pattern: freeze the foundation model,
cache representations, train a lightweight task head.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
    TRAIN_SAMPLES, VAL_SAMPLES, DATA_DIR, MODEL_DIR,
)
from model import OligoScorer
from data_gen import save_dataset


def cache_embeddings(model: OligoScorer, sequences: list[str],
                     batch_size: int = 64) -> torch.Tensor:
    """
    Pre-compute mean-pooled embeddings for all sequences.

    This is the expensive step (~2-5 min for 5K sequences on CPU).
    We do it once and cache the result.
    """
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        with torch.no_grad():
            emb = model.get_embeddings(batch)
            all_embeddings.append(emb["pooled_embedding"].cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  Encoding batch {i // batch_size + 1}/"
                  f"{len(sequences) // batch_size + 1}")

    return torch.cat(all_embeddings, dim=0)


def train(use_cached: bool = True):
    """
    Train the scoring head.

    Steps:
      1. Generate synthetic data (if not already cached)
      2. Load Nucleotide Transformer and extract embeddings (one-time cost)
      3. Train MLP head on cached embeddings
      4. Evaluate on validation set
      5. Save best model
    """
    # --- Step 1: Data ---
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"

    if not train_path.exists():
        print("Generating synthetic training data...")
        save_dataset(TRAIN_SAMPLES, VAL_SAMPLES)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # --- Step 2: Embeddings ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = OligoScorer(device=device)
    params = model.parameter_count()
    print(f"Parameters — Frozen: {params['encoder_frozen']:,}, "
          f"Trainable: {params['head_trainable']:,} "
          f"({params['percent_trainable']})")

    # Cache embeddings to disk so we don't recompute on every run
    train_emb_path = DATA_DIR / "train_embeddings.pt"
    val_emb_path = DATA_DIR / "val_embeddings.pt"

    if use_cached and train_emb_path.exists():
        print("Loading cached embeddings...")
        train_embeddings = torch.load(train_emb_path, map_location=device)
        val_embeddings = torch.load(val_emb_path, map_location=device)
    else:
        print("Extracting Nucleotide Transformer embeddings (one-time cost)...")
        train_embeddings = cache_embeddings(
            model, train_df["sequence"].tolist()
        )
        val_embeddings = cache_embeddings(
            model, val_df["sequence"].tolist()
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(train_embeddings, train_emb_path)
        torch.save(val_embeddings, val_emb_path)
        print("Embeddings cached to disk.")

    # --- Step 3: Train the head ---
    train_scores = torch.tensor(
        train_df["score"].values, dtype=torch.float32
    )
    val_scores = torch.tensor(
        val_df["score"].values, dtype=torch.float32
    )

    train_ds = TensorDataset(train_embeddings, train_scores)
    val_ds = TensorDataset(val_embeddings, val_scores)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Only optimise the head parameters
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Learning rate scheduler — reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 10

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={patience_limit})...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        # Train
        model.head.train()
        train_loss = 0.0
        for emb_batch, score_batch in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)

            pred = model.forward_from_embedding(emb_batch)
            loss = criterion(pred, score_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb_batch)

        train_loss /= len(train_ds)

        # Validate
        model.head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb_batch, score_batch in val_loader:
                emb_batch = emb_batch.to(device)
                score_batch = score_batch.to(device)
                pred = model.forward_from_embedding(emb_batch)
                loss = criterion(pred, score_batch)
                val_loss += loss.item() * len(emb_batch)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        # RMSE is more interpretable: "average error in score points"
        train_rmse = train_loss ** 0.5
        val_rmse = val_loss ** 0.5

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} — "
                  f"Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save()
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print("-" * 60)
    print(f"Best validation RMSE: {best_val_loss ** 0.5:.2f} score points")
    print(f"Model saved to {MODEL_DIR / 'scorer_head.pt'}")

    return model


if __name__ == "__main__":
    train()
