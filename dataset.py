from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetConfig:
    min_count: int = 5
    window_size: int = 5
    seed: int = 42
    negative_sampling_power: float = 0.75
    max_tokens_per_sentence: int = 10000
    lowercase: bool = True


@dataclass
class DatasetArtifacts:
    word2id: dict[str, int] = field(default_factory=dict)
    id2word: dict[int, str] = field(default_factory=dict)
    counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    train_sentences: list[np.ndarray] = field(default_factory=list)
    neg_probs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    vocab_size: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size == 0 and self.word2id:
            self.vocab_size = len(self.word2id)


def _normalize(text: str, lowercase: bool) -> str:
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return text.split()


class TextDatasetModule:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def build_from_text(self, text: str) -> DatasetArtifacts:
        text = _normalize(text, self.config.lowercase)
        tokens = _tokenize(text)
        if not tokens:
            return DatasetArtifacts()

        print("Number of words in text: " + str(len(tokens)))

        counts_raw: Counter[str] = Counter(tokens)
        vocab_tokens = [w for w, c in counts_raw.items() if c >= self.config.min_count]
        vocab_tokens.sort(key=lambda w: (-counts_raw[w], w))

        word2id = {w: i for i, w in enumerate(vocab_tokens)}
        id2word = {i: w for w, i in word2id.items()}
        V = len(word2id)
        counts = np.zeros(V, dtype=np.int64)
        for w, i in word2id.items():
            counts[i] = counts_raw[w]

        train_sentences: list[np.ndarray] = []
        max_len = self.config.max_tokens_per_sentence
        for i in range(0, len(tokens), max_len):
            chunk = tokens[i : i + max_len]
            ids = [word2id[w] for w in chunk if w in word2id]
            if ids:
                train_sentences.append(np.array(ids, dtype=np.int64))

        neg_probs = np.power(counts.astype(np.float64), self.config.negative_sampling_power)
        total = neg_probs.sum()
        if total > 0:
            neg_probs /= total
        else:
            neg_probs = np.ones(V, dtype=np.float64) / V

        return DatasetArtifacts(
            word2id=word2id,
            id2word=id2word,
            counts=counts,
            train_sentences=train_sentences,
            neg_probs=neg_probs,
            vocab_size=V,
        )


def collect_skipgram_pairs(
    sentences: list[np.ndarray],
    window_size: int,
    *,
    seed: int | None = None,
) -> np.ndarray:

    pairs: list[tuple[int, int]] = []
    rng = np.random.default_rng(seed)
    for sent in sentences:
        n = len(sent)
        for t in range(n):
            center = int(sent[t])
            R = rng.integers(1, window_size + 1) if window_size >= 1 else 0
            for j in range(-R, R + 1):
                if j == 0:
                    continue
                idx = t + j
                if 0 <= idx < n:
                    pairs.append((center, int(sent[idx])))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)
