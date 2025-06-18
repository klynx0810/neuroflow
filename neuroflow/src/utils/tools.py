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

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42, shuffle: bool = True):
    """
    Chia tập dữ liệu thành tập train và test, ép kiểu X về float32.

    Parameters:
    - X: Dữ liệu đầu vào (numpy array)
    - y: Nhãn (numpy array)
    - test_size: Tỷ lệ dữ liệu dành cho tập test (mặc định 0.2)
    - seed: Seed để shuffle (mặc định 42)
    - shuffle: Có muốn xáo trộn trước khi chia không

    Returns:
    - X_train, X_test, y_train, y_test
    """
    assert len(X) == len(y), "X và y phải cùng số lượng mẫu"
    if shuffle:
        X, y = shuffle_dataset(X, y, seed)

    n_test = int(len(X) * test_size)
    X_train, X_test = X[:-n_test].astype(np.float32), X[-n_test:].astype(np.float32)
    y_train, y_test = y[:-n_test], y[-n_test:]
    return X_train, X_test, y_train, y_test
