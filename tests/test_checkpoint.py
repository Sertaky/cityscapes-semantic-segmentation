"""Checkpoint metadata and optimizer-state roundtrip tests."""

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
