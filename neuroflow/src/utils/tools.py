import numpy as np

def shuffle_dataset(X: np.ndarray, y: np.ndarray, seed: int = 42):
    assert len(X) == len(y), "X và y phải cùng số lượng mẫu"
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def to_one_hot(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot