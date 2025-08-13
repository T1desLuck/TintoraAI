import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Фиксирует генераторы случайных чисел для воспроизводимости.
    Включает детерминированный режим cuDNN и отключает benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
