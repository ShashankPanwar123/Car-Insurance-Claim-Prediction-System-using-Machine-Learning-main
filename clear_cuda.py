# ===============================
# CUDA MEMORY CLEANER SCRIPT
# ===============================

import torch
import gc

print("🔍 Checking CUDA status...")

if torch.cuda.is_available():
    print("✅ CUDA Available")
    print("GPU:", torch.cuda.get_device_name(0))

    print("\n📊 BEFORE CLEANING:")
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

    # 🔥 Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("\n🧹 AFTER CLEANING:")
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

    print("\n✅ CUDA memory cleared successfully!")

else:
    print("❌ CUDA not available (using CPU)")