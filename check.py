import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import pandas as pd
tr = pd.read_csv('data/train_split.csv')
te = pd.read_csv('data/test_split.csv')
print(f'Train: {len(tr)} | Test: {len(te)}')