import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import config
from data import get_test_loader
from model import HybridSwinViT
from uncertainty import mc_predict


def main():
    loader, classes = get_test_loader()
    model = HybridSwinViT(len(classes), pretrained=False, dropout=config.DROPOUT).to(config.DEVICE)
    model.load_state_dict(torch.load(config.OUT_DIR / "best_model.pth", map_location=config.DEVICE))
    model.eval()

    y_true, y_pred, ents = [], [], []
    for x, y in loader:
        x = x.to(config.DEVICE)
        mean, ent, _ = mc_predict(model, x, 1)
        y_pred += mean.argmax(1).cpu().tolist()
        ents += ent.cpu().tolist()
        y_true += y.tolist()
    y_true, y_pred, ents = np.array(y_true), np.array(y_pred), np.array(ents)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(f"Test accuracy: {acc:.4f}\n{report}")
    wrong = y_true != y_pred
    unc = {
        "mean_entropy_correct": float(ents[~wrong].mean()),
        "mean_entropy_incorrect": float(ents[wrong].mean()) if wrong.any() else None,
    }
    print("Uncertainty:", unc)
    (config.OUT_DIR / "test_report.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}\n{json.dumps(unc, indent=2)}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(config.OUT_DIR / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
