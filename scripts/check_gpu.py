#!/usr/bin/env python3
from __future__ import annotations
import sys

try:
    import torch
except Exception as exc:
    print(f"[ERROR] Nepavyko importuoti torch: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"cuda device count: {torch.cuda.device_count()}")
    print(f"current device: {torch.cuda.current_device()}")
    print(f"device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    sys.exit(0)

print("[ERROR] GPU nepasiekiamas. Paleidimas neturėtų vykti ant CPU.", file=sys.stderr)
sys.exit(2)
