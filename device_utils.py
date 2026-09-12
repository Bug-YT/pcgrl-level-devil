"""
device_utils.py
================
Erkennt automatisch verfuegbare Hardware (CPU-Kerne, GPU via torch.cuda)
und gibt Log-Zeilen im gewuenschten Format aus, z.B.:
  [DEVICE] Using GPU - 1 x NVIDIA RTX 4090
  [DEVICE] Using CPU - 12 Cores
  [DEVICE] CPU Cores Available: 16 | Using: 4 Envs
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str          # "cuda" oder "cpu"
    gpu_count: int
    gpu_name: str | None
    cpu_cores: int
    n_envs: int


def detect_device(requested_device: str, requested_n_envs: int) -> DeviceInfo:
    cpu_cores = os.cpu_count() or 1
    n_envs = max(1, requested_n_envs)

    gpu_count = 0
    gpu_name = None
    device = "cpu"

    try:
        import torch

        if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()):
            if torch.cuda.is_available():
                device = "cuda"
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
            else:
                device = "cpu"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    return DeviceInfo(
        device=device,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        cpu_cores=cpu_cores,
        n_envs=n_envs,
    )


def log_device_info(info: DeviceInfo) -> None:
    if info.device == "cuda" and info.gpu_count > 0:
        print(f"[DEVICE] Using GPU - {info.gpu_count} x {info.gpu_name}")
    else:
        print(f"[DEVICE] Using CPU - {info.cpu_cores} Cores")
    print(f"[DEVICE] CPU Cores Available: {info.cpu_cores} | Using: {info.n_envs} Envs")
