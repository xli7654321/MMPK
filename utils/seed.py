import os
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # torch < 1.9
    torch.use_deterministic_algorithms(True)  # torch >= 1.9
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return

if __name__ == '__main__':
    pass
