import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lab.training import evaluate, train_epoch


def test_evaluate_weights_metrics_by_examples() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(nn.Identity(), loader, criterion, torch.device("cpu"))
    expected = criterion(logits, labels).item()
    assert metrics.loss == pytest.approx(expected)
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.examples == 3


def test_train_epoch_updates_parameters() -> None:
    torch.manual_seed(3)
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)
    model = nn.Linear(1, 2)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert metrics.examples == 4
    assert not torch.equal(before, model.weight.detach())


def test_empty_or_invalid_batch_limit_fails_clearly() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="did not yield"):
        evaluate(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    populated = DataLoader(TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)))
    with pytest.raises(ValueError, match="positive"):
        evaluate(
            nn.Identity(),
            populated,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            max_batches=0,
        )


def test_batch_limit_bounds_work() -> None:
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    metrics = evaluate(
        nn.Identity(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        max_batches=1,
    )
    assert metrics.examples == 2
