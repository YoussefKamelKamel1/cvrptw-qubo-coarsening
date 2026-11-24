"""Ensemble merge scorer: averages the merge probabilities of several trained scorers."""
import torch
from torch import nn


class EnsembleScorer(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    @torch.no_grad()
    def score_bundle(self, fb, device):
        probs = torch.stack([torch.sigmoid(m.score_bundle(fb, device)) for m in self.models])
        p = probs.mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))   # logit of the mean probability


def load_any(path, device="cpu"):
    """One checkpoint (GraphSAGE, GAT or GCN) from experiments/retrain_sage.py."""
    import sys
    from pathlib import Path
    from src.gnn_coarsening import load_model
    ck = torch.load(path, map_location=device)
    if ck.get("arch", "sage") == "sage":
        return load_model(path, device=device)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compare_architectures import VariantScorer
    m = VariantScorer(ck["arch"], hidden=ck.get("hidden", 64), num_layers=ck.get("num_layers", 2))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def load_spec(spec, device="cpu"):
    """'ens:a.pt,b.pt,...' -> EnsembleScorer; otherwise a single checkpoint."""
    if spec.startswith("ens:"):
        return EnsembleScorer([load_any(p, device) for p in spec[4:].split(",")]).eval()
    return load_any(spec, device)
