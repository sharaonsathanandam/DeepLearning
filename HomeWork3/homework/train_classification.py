"""Run from the project root: python -m homework.train_classification."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import load_model, save_model


def train(data_path="classification_data", epochs=20, batch_size=128, lr=1e-3,
          num_workers=2, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    loaders = {}
    for split in ("train", "val"):
        dataset = load_data(
            Path(data_path) / split,
            transform_pipeline="aug" if split == "train" else "default",
            return_dataloader=False,
        )
        if not len(dataset):
            raise ValueError(f"Empty {split} dataset")
        # Include the final partial batch in validation and training.
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=split == "train", num_workers=num_workers)

    model = load_model("classifier").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0

    for epoch in range(epochs):
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            metric = AccuracyMetric()
            total_loss = 0.0
            samples = 0
            with torch.set_grad_enabled(training):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                    metric.add(logits.argmax(1), labels)
                    total_loss += loss.item() * images.size(0)
                    samples += images.size(0)
            accuracy = metric.compute()["accuracy"]
            print(f"Epoch {epoch + 1}/{epochs} {split}: "
                  f"loss={total_loss / samples:.4f} accuracy={accuracy:.4f}", flush=True)
            if not training and accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"Saved {save_model(model)}", flush=True)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="classification_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    train(**vars(parser.parse_args()))
