# save as check_loss.py
import os
import glob

# Check all saved checkpoints and their file sizes
# Larger file = more parameters saved = sanity check
models = glob.glob("models/*.pth")
for m in sorted(models):
    size = os.path.getsize(m) / 1024 / 1024
    mtime = os.path.getmtime(m)
    print(f"{m:<45} {size:.1f} MB")