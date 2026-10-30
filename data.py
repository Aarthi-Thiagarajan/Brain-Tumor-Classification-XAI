import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import config

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_train_val_loaders():
    full_train = datasets.ImageFolder(config.TRAIN_DIR, transform=train_tf)
    full_val = datasets.ImageFolder(config.TRAIN_DIR, transform=eval_tf)
    n = len(full_train)
    g = torch.Generator().manual_seed(config.SEED)
    idx = torch.randperm(n, generator=g).tolist()
    n_val = int(n * config.VAL_SPLIT)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(Subset(full_val, val_idx), batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    classes = full_train.classes
    (config.OUT_DIR / "classes.json").write_text(json.dumps(classes))
    return train_loader, val_loader, classes


def get_test_loader():
    ds = datasets.ImageFolder(config.TEST_DIR, transform=eval_tf)
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS), ds.classes
