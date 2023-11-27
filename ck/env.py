import platform

import torch

def get_device():
    device = 'cpu'
    current_system = platform.system()
    if current_system == 'Darwin':
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = 'mps'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
            
    return torch.device(device)