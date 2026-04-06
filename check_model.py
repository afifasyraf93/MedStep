# Run this as a quick check - save as check_model.py
from modules.detection import build_model

model = build_model()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable

print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {frozen:,}")
print(f"Trainable %:          {trainable/total*100:.1f}%")