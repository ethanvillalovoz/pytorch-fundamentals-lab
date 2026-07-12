import random

import numpy as np
import pytest
import torch

from pytorch_lab.reproducibility import resolve_device, seed_everything


def test_seed_everything_repeats_all_generators() -> None:
    seed_everything(17)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(17)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_explicit_cpu_is_supported() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_unavailable_accelerators_fail_clearly(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")
    assert resolve_device() == torch.device("cpu")


def test_automatic_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == torch.device("cuda")


def test_automatic_device_uses_mps_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == torch.device("mps")
