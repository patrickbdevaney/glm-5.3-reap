import sys
sys.path.insert(0,'scripts')
import torch
if not torch.cuda.is_available():
    print("GPU ABSENT", flush=True); sys.exit(2)
try:
    torch.zeros(1024, device="cuda").sum().item()
except Exception as e:
    print(f"GPU UNUSABLE ({type(e).__name__}: {e})", flush=True); sys.exit(2)
print("GPU OK", flush=True)
