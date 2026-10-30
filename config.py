from pathlib import Path
import torch

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "dataset" / "Training"
TEST_DIR = ROOT / "dataset" / "Testing"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16          # laptop-la memory error vandha 8 ku kurainga
EPOCHS = 10              # time irundha 15-20 vara increase pannalam
LR = 1e-4
VAL_SPLIT = 0.2
DROPOUT = 0.3
MC_PASSES = 5           # Monte-Carlo Dropout passes (uncertainty)
SEED = 42
NUM_WORKERS = 0          # Windows-la 0 dhaan safe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
