

from __future__ import annotations

import argparse
import numpy as np

from dataset import (
    DatasetConfig,
    TextDatasetModule,
    collect_skipgram_pairs,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1/(1+exp(-x)), numerically stable."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log σ(x), numerically stable."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def loss_one_pair(
    center_id: int,
    context_id: int,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
) -> float:

    v_center = input_embeddings[center_id]
    v_context = output_embeddings[context_id]
    score = float(np.dot(v_center, v_context))
    return float(-log_sigmoid(np.array(score)))


def loss_and_grad_one_pair(
    center_id: int,
    context_id: int,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:

    v_center = input_embeddings[center_id]
    v_context = output_embeddings[context_id]
    score = float(np.dot(v_center, v_context))
    loss = float(-log_sigmoid(np.array(score)))
    g = float(sigmoid(np.array([score]))[0]) - 1.0

    grad_input = np.zeros_like(input_embeddings)
    grad_output = np.zeros_like(output_embeddings)
    grad_input[center_id] = g * v_context
    grad_output[context_id] = g * v_center

    return loss, grad_input, grad_output


def loss_and_grad_neg(
    center_id: int,
    context_id: int,
    neg_sample_ids: np.ndarray,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:

    v_center = input_embeddings[center_id]
    v_context = output_embeddings[context_id]
    score_pos = float(np.dot(v_center, v_context))
    loss = float(-log_sigmoid(np.array(score_pos)))
    g_pos = float(sigmoid(np.array([score_pos]))[0]) - 1.0

    grad_input = np.zeros_like(input_embeddings)
    grad_output = np.zeros_like(output_embeddings)
    grad_input[center_id] = g_pos * v_context
    grad_output[context_id] = g_pos * v_center

    for neg_id in neg_sample_ids:
        neg_id = int(neg_id)
        if neg_id == context_id:
            continue
        v_neg = output_embeddings[neg_id]
        score_neg = float(np.dot(v_center, v_neg))
        loss -= float(log_sigmoid(np.array(-score_neg)))
        g_neg = float(sigmoid(np.array([score_neg]))[0])
        grad_output[neg_id] += g_neg * v_center
        grad_input[center_id] += g_neg * v_neg

    return loss, grad_input, grad_output


def train_step_neg(
    pairs_batch: np.ndarray,
    neg_sample_ids_batch: np.ndarray,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    learning_rate: float,
) -> float:

    N = len(pairs_batch)
    grad_in = np.zeros_like(input_embeddings)
    grad_out = np.zeros_like(output_embeddings)
    total_loss = 0.0
    for i in range(N):
        center_id = int(pairs_batch[i, 0])
        context_id = int(pairs_batch[i, 1])
        neg_ids = neg_sample_ids_batch[i]
        loss, gi, go = loss_and_grad_neg(
            center_id, context_id, neg_ids,
            input_embeddings, output_embeddings,
        )
        total_loss += loss
        grad_in += gi
        grad_out += go
    input_embeddings -= learning_rate * (grad_in / N)
    output_embeddings -= learning_rate * (grad_out / N)
    return total_loss / N


# --- defaults ---
DEFAULT_TEXT_PATH = "text8.txt"
DEFAULT_MAX_CHARS = 2_000_000
DEFAULT_MIN_COUNT = 5
DEFAULT_WINDOW_SIZE = 5
DEFAULT_EMBED_DIM = 100
DEFAULT_BATCH_SIZE = 200
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.025
DEFAULT_NEG_K = 5
DEFAULT_SEED = 42


def run_training(
    text_path: str = DEFAULT_TEXT_PATH,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_count: int = DEFAULT_MIN_COUNT,
    window_size: int = DEFAULT_WINDOW_SIZE,
    embed_dim: int = DEFAULT_EMBED_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    neg_k: int = DEFAULT_NEG_K,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)

    config = DatasetConfig(
        min_count=min_count,
        window_size=window_size,
        seed=seed,
    )
    module = TextDatasetModule(config)
    artifacts = module.build_from_text(text)
    V = artifacts.vocab_size

    pairs = collect_skipgram_pairs(
        artifacts.train_sentences,
        config.window_size,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    input_emb = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (V, embed_dim)).astype(np.float64)
    output_emb = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (V, embed_dim)).astype(np.float64)

    n_batches = (len(pairs) + batch_size - 1) // batch_size
    for epoch in range(epochs):
        perm = rng.permutation(len(pairs))
        epoch_loss = 0.0
        for b in range(n_batches):
            batch = pairs[perm[b * batch_size : (b + 1) * batch_size]]
            neg_ids = rng.choice(V, size=(len(batch), neg_k), replace=True, p=artifacts.neg_probs)
            loss = train_step_neg(batch, neg_ids, input_emb, output_emb, learning_rate)
            epoch_loss += loss
            print("Completed batch "+str(b))
        print(f"Epoch {epoch + 1}/{epochs}  mean loss: {epoch_loss / n_batches:.4f}")

    return input_emb, output_emb


def simple_test():
    return


def main() -> None:

    text_path = 'text8.txt'
    max_chars = 200_000
    epochs = 30
    run_training(text_path = text_path, max_chars = max_chars, epochs = epochs)


if __name__ == "__main__":
    main()



