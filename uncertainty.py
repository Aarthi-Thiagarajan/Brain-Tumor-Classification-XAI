import torch
import torch.nn as nn


def enable_mc_dropout(model):
    """Eval mode, but keep Dropout layers active for Monte-Carlo sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, x, passes=5):
    """Returns mean probs, predictive entropy, and std over MC passes."""
    enable_mc_dropout(model)
    probs = torch.stack([torch.softmax(model(x), dim=1) for _ in range(passes)])
    model.eval()
    mean = probs.mean(0)
    entropy = -(mean * torch.log(mean + 1e-12)).sum(1)
    std = probs.std(unbiased=False)
    return mean, entropy, std
