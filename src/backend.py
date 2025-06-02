try:
    import cupy as cp
    backend = cp
    is_gpu = True
except ImportError:
    import numpy as np
    backend = np
    is_gpu = False
