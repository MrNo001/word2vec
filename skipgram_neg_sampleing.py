import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def log_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))

def loss_one_pair(center_id: int, context_id: int, input_embeddings: np.ndarray, output_embeddings: np.ndarray) -> float:

    v_center = input_embeddings[center_id]
    v_context = output_embeddings[context_id]
    score = float(np.dot(v_center, v_context))
    return float(-log_sigmoid(np.array(score)))

arr1d = np.array([-1, -0.4, 0, 0.7, 3])
arr2d = np.array([-1, -0.4, 0, 0.7, 3])


if __name__ == '__main__':
    score = float(np.dot(arr1d, arr2d))
    print(float(-log_sigmoid(np.array(score))))
