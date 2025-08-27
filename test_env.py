#!/usr/bin/env python3
"""
Simple Environment Test
"""

import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

try:
    from transformers import AutoTokenizer
    print("✅ Transformers imported successfully")
except Exception as e:
    print(f"❌ Transformers import failed: {e}")

try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth imported successfully")
except Exception as e:
    print(f"❌ Unsloth import failed: {e}")

print("Environment test completed!") 