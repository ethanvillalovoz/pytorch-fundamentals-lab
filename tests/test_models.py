import pytest
import torch

from pytorch_lab.models import IrisMLP, MnistCNN


def test_iris_mlp_shape_and_raw_logits() -> None:
    model = IrisMLP()
    logits = model(torch.zeros(5, 4))
    assert logits.shape == (5, 3)
    assert torch.isfinite(logits).all()


def test_iris_mlp_rejects_invalid_hidden_layers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        IrisMLP(hidden_features=())
    with pytest.raises(ValueError, match="positive"):
        IrisMLP(hidden_features=(8, 0))


def test_mnist_cnn_shape() -> None:
    model = MnistCNN()
    logits = model(torch.zeros(4, 1, 28, 28))
    assert logits.shape == (4, 10)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(4))
