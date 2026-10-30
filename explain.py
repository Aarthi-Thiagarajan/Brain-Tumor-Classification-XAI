import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM on the last Swin block (works for timm NHWC and NLC outputs)."""

    def __init__(self, model):
        self.model = model
        self.acts, self.grads = None, None
        layer = model.swin.layers[-1].blocks[-1].norm1
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(x)
            if class_idx is None:
                class_idx = out.argmax(1).item()
            out[0, class_idx].backward()
        a, g = self.acts, self.grads
        if a.dim() == 3:  # (B, N, C) -> (B, H, W, C)
            n = int(a.shape[1] ** 0.5)
            a, g = a.reshape(1, n, n, -1), g.reshape(1, n, n, -1)
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * a).sum(-1)).unsqueeze(1)          # (1,1,H,W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx



def shap_map(model, x, background, class_idx):
    import shap
    import torch

    model.eval()

    # Use only a small background sample
    background = background[:2]
    x = x[:1]

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(x, nsamples=10)

    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    sv = torch.as_tensor(sv)

    # Handle output dimensions safely
    if sv.ndim == 5:
        if sv.shape[-1] > class_idx:
            sv = sv[..., class_idx]
        elif sv.shape[1] > class_idx:
            sv = sv[:, class_idx]

    return sv.detach().cpu().numpy().sum(axis=0)




def save_explanation(img_rgb, cam, shap_vals, title, path):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    cols = 3 if shap_vals is not None else 2
    fig, ax = plt.subplots(1, cols, figsize=(5 * cols, 5))

    # 1. Original MRI image
    ax[0].imshow(img_rgb)
    ax[0].set_title("MRI Input")
    ax[0].axis("off")

    # 2. Grad-CAM heatmap
    cam = np.asarray(cam).squeeze()
    cam = cv2.resize(
        cam.astype(np.float32),
        (img_rgb.shape[1], img_rgb.shape[0])
    )

    ax[1].imshow(img_rgb)
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")

    # 3. SHAP explanation
    if shap_vals is not None:
        sv = np.asarray(shap_vals).squeeze()

        # Convert SHAP values into a 2D heatmap
        if sv.ndim == 3:
            sv = np.sum(np.abs(sv), axis=-1)

        if sv.ndim == 2:
            sv = cv2.resize(
                sv.astype(np.float32),
                (img_rgb.shape[1], img_rgb.shape[0])
            )

            limit = np.max(np.abs(sv)) + 1e-8

            ax[2].imshow(img_rgb)
            ax[2].imshow(
                sv,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                alpha=0.55
            )
            ax[2].set_title("SHAP Explanation")
        else:
            ax[2].imshow(img_rgb)
            ax[2].set_title("SHAP: unsupported shape")

        ax[2].axis("off")

   
    fig.suptitle(title)
    plt.tight_layout()

    import os
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    print("Explanation functions are ready.")
    print("Use predict.py to test a single MRI image.")