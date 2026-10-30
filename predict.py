"""Usage:
    python predict.py --image path/to/mri.jpg            (prediction + uncertainty + Grad-CAM)
    python predict.py --image path/to/mri.jpg --shap     (also SHAP)
"""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
import config
from data import eval_tf
from model import HybridSwinViT
from uncertainty import mc_predict
from explain import GradCAM, shap_map, save_explanation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--shap", action="store_true")
    args = ap.parse_args()

    classes = json.loads((config.OUT_DIR / "classes.json").read_text())
    model = HybridSwinViT(len(classes), pretrained=False, dropout=config.DROPOUT).to(config.DEVICE)
    model.load_state_dict(torch.load(config.OUT_DIR / "best_model.pth", map_location=config.DEVICE))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(config.DEVICE)
    img_rgb = np.array(img.resize((config.IMG_SIZE, config.IMG_SIZE)))

    mean, ent, std = mc_predict(model, x, config.MC_PASSES)
    idx = mean.argmax(1).item()
    conf = mean[0, idx].item()
    print(f"Prediction : {classes[idx]}")
    print(f"Confidence : {conf * 100:.2f}%")
    print(f"Uncertainty (entropy): {ent.item():.4f}  (max possible {np.log(len(classes)):.4f})")

    cam, _ = GradCAM(model)(x, idx)
    sv = None
    if args.shap:
        ds = datasets.ImageFolder(config.TRAIN_DIR, transform=eval_tf)
        sel = torch.randperm(len(ds))[:2].tolist()
        bg = torch.stack([ds[i][0] for i in sel]).to(config.DEVICE)
        sv = shap_map(model, x, bg, idx)

    out = config.OUT_DIR / "explanation.png"
    save_explanation(img_rgb, cam, sv,
                     f"{classes[idx]} | conf {conf*100:.1f}% | entropy {ent.item():.3f}", out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
