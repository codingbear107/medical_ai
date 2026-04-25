"""
Utility functions: metrics, seeding, checkpointing, logging.
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1(y_true, y_pred, num_classes=11):
    """Compute macro F1 score."""
    return f1_score(y_true, y_pred, average='macro', labels=range(num_classes))


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=11):
    """Evaluate model and return macro F1 + accuracy."""
    model.eval()
    all_preds, all_labels = [], []

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy() if isinstance(y, torch.Tensor) else y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = macro_f1(all_labels, all_preds, num_classes)
    acc = (all_preds == all_labels).mean()
    return f1, acc


def save_checkpoint(model, path, extra=None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model_state_dict': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, path, device='cuda'):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


class EarlyStopping:
    """Early stopping based on a monitored metric (higher is better)."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # not improved


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
