"""
Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention.
Computes Fisher Information Matrix from base domain and penalizes
parameter drift during few-shot fine-tuning.
"""
import torch
import torch.nn.functional as F


def compute_fisher(model, dataloader, device, num_samples=2000):
    """Compute diagonal Fisher Information Matrix over base domain data.

    Args:
        model: trained base model
        dataloader: base domain validation loader
        device: torch device
        num_samples: max samples to use for Fisher estimation

    Returns:
        fisher: dict of {param_name: Fisher diagonal tensor}
    """
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)

    model.eval()
    count = 0

    for x, y in dataloader:
        if count >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        model.zero_grad()
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=1)

        # Sample from model's own distribution (proper Fisher)
        probs = torch.exp(log_probs)
        sampled_labels = torch.multinomial(probs, 1).squeeze(1)
        loss = F.nll_loss(log_probs, sampled_labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2) * batch_size

        count += batch_size

    # Normalize by total sample count
    for name in fisher:
        fisher[name] /= max(count, 1)

    return fisher


class EWCLoss:
    """EWC penalty loss to prevent catastrophic forgetting.

    Usage:
        ewc = EWCLoss(model, fisher, ewc_lambda=5000)
        ...
        total_loss = ce_loss + ewc.penalty(model)
    """

    def __init__(self, model, fisher, ewc_lambda=5000.0):
        self.fisher = {n: f.clone() for n, f in fisher.items()}
        self.old_params = {n: p.data.clone() for n, p in model.named_parameters()}
        self.ewc_lambda = ewc_lambda

    def penalty(self, model):
        """Compute EWC penalty: lambda * sum(F_i * (theta_i - theta_i*)^2)"""
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                diff = param - self.old_params[name]
                loss = loss + (self.fisher[name] * diff.pow(2)).sum()
        return self.ewc_lambda * loss


def save_fisher(fisher, old_params, fisher_path, params_path):
    """Save Fisher matrix and base parameters to disk."""
    torch.save(fisher, fisher_path)
    torch.save(old_params, params_path)


def load_ewc(model, fisher_path, params_path, ewc_lambda=5000.0, device='cuda'):
    """Load saved Fisher and create EWCLoss."""
    fisher = torch.load(fisher_path, map_location=device)
    # old_params not needed separately if we load from fisher_path context
    # But we saved them separately for clarity
    old_params_dict = torch.load(params_path, map_location=device)

    # Create a temporary namespace to hold old params for EWCLoss
    class _Holder:
        pass

    ewc = EWCLoss.__new__(EWCLoss)
    ewc.fisher = fisher
    ewc.old_params = old_params_dict
    ewc.ewc_lambda = ewc_lambda
    return ewc
