"""Run from the project root: python -m homework.train_detection."""

import argparse
from pathlib import Path

import torch

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model


def train(data_path="drive_data", epochs=30, batch_size=32, lr=1e-3,
          num_workers=2, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    loaders = {
        split: load_data(Path(data_path) / split, batch_size=batch_size,
                         shuffle=split == "train", num_workers=num_workers)
        for split in ("train", "val")
    }
    for split, loader in loaders.items():
        if not len(loader.dataset):
            raise ValueError(f"Empty {split} dataset")

    model = load_model("detector").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Give the sparse left/right boundaries more weight than background.
    segmentation_loss = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4.0, 4.0], device=device)
    )
    best_score = -float("inf")

    for epoch in range(epochs):
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            metric = DetectionMetric()
            total_loss = 0.0
            samples = 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    images = batch["image"].to(device)
                    labels = batch["track"].to(device)
                    depths = batch["depth"].to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits, predicted_depth = model(images)
                    depth_error = (predicted_depth - depths).abs()
                    boundary = (labels > 0).to(depth_error.dtype)
                    boundary_error = (depth_error * boundary).sum() / boundary.sum().clamp_min(1)
                    loss = segmentation_loss(logits, labels) + depth_error.mean() + boundary_error
                    if training:
                        loss.backward()
                        optimizer.step()
                    metric.add(logits.argmax(1), labels, predicted_depth, depths)
                    total_loss += loss.item() * images.size(0)
                    samples += images.size(0)
            results = metric.compute()
            print(f"Epoch {epoch + 1}/{epochs} {split}: loss={total_loss / samples:.4f} "
                  f"iou={results['iou']:.4f} accuracy={results['accuracy']:.4f} "
                  f"depth_mae={results['abs_depth_error']:.4f} "
                  f"boundary_mae={results['tp_depth_error']:.4f}", flush=True)
            score = results["iou"] - results["abs_depth_error"] - results["tp_depth_error"]
            if not training and score > best_score:
                best_score = score
                print(f"Saved {save_model(model)}", flush=True)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="drive_data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    train(**vars(parser.parse_args()))
