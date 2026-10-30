"""Checkpoint metadata and optimizer-state roundtrip tests."""

import random

import numpy as np
import torch
from torch import nn

from cityscapes_segmentation.engine.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_restores_model_optimizer_epoch_and_best_miou(tmp_path):
    torch.manual_seed(7)
    model = nn.Conv2d(3, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    image = torch.randn(1, 3, 4, 4)
    target = torch.randint(0, 2, (1, 4, 4))
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(image), target).backward()
    optimizer.step()

    path = tmp_path / "best.pt"
    config = {"seed": 42, "base_channels": 64, "dev_val_cities": ["jena", "krefeld", "ulm"]}
    save_checkpoint(path, epoch=3, model=model, optimizer=optimizer, best_miou=0.375, config=config)
    assert path.is_file()

    resumed_model = nn.Conv2d(3, 2, kernel_size=1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.1)
    restored = load_checkpoint(path, model=resumed_model, optimizer=resumed_optimizer)
    assert restored.epoch == 3
    assert restored.best_miou == 0.375
    assert restored.config == config
    for original, resumed in zip(model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(original, resumed)
    torch.testing.assert_close(model(image), resumed_model(image))
    assert resumed_optimizer.param_groups[0]["lr"] == 0.003
    assert len(resumed_optimizer.state) == len(optimizer.state) > 0
    for original_state, resumed_state in zip(optimizer.state.values(), resumed_optimizer.state.values()):
        for key, value in original_state.items():
            torch.testing.assert_close(value, resumed_state[key])


def test_scheduler_and_extra_state_roundtrip_for_exact_resume(tmp_path):
    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 1, 2, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    generator_state = torch.Generator().manual_seed(42).get_state()
    numpy_state = np.random.get_state()
    extra_state = {
        "train_generator_state": generator_state,
        "python_random_state": random.getstate(),
        "numpy_random_state": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_miou=0.25,
        config={"experiment_name": "test", "torch_version": torch.__version__},
        scheduler=scheduler,
        extra_state=extra_state,
    )

    restored_model = nn.Conv2d(1, 2, kernel_size=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001, weight_decay=0.0001)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=30, eta_min=1e-5
    )
    metadata = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert metadata.epoch == 1
    assert metadata.best_miou == 0.25
    assert metadata.config == {"experiment_name": "test", "torch_version": torch.__version__}
    torch.testing.assert_close(metadata.extra_state["train_generator_state"], generator_state)
    assert metadata.extra_state["python_random_state"] == extra_state["python_random_state"]
    assert metadata.extra_state["numpy_random_state"] == extra_state["numpy_random_state"]
    torch.testing.assert_close(metadata.extra_state["torch_rng_state"], extra_state["torch_rng_state"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
