import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
from data import get_train_val_loaders
from model import HybridSwinViT


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    train_loader, val_loader, classes = get_train_val_loaders()
    print("Classes:", classes, "| Device:", config.DEVICE)

    model = HybridSwinViT(len(classes), dropout=config.DROPOUT).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    for ep in range(1, config.EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        scheduler.step()
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"Epoch {ep}/{config.EPOCHS} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), config.OUT_DIR / "best_model.pth")
            print("  -> best model saved")

    (config.OUT_DIR / "history.json").write_text(json.dumps(hist))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_acc"], label="train"); ax[0].plot(hist["val_acc"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(hist["train_loss"], label="train"); ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(config.OUT_DIR / "training_curves.png", dpi=150)
    print("Done. Best val acc:", round(best, 4))


if __name__ == "__main__":
    main()
