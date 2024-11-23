from functools import partial
from typing import NamedTuple, Tuple
import torch
import torch.nn.functional as F

# Set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Constants
EPS = 1e-8
MAX_TEMP = 100.0
MIN_TEMP = 0.01

def kl_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two log probability distributions."""
    p = torch.exp(logp)
    return torch.sum(torch.where(p > 0, p * (logp - logq), 0.0), dim=-1)

def ent_varent(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy and varentropy from log probabilities."""
    p = torch.exp(logp)
    ent = -torch.sum(p * logp, dim=-1)
    diff = logp + ent.unsqueeze(-1)
    varent = torch.sum(p * diff**2, dim=-1)
    return ent, varent

def normalize_logits(logits: torch.Tensor, noise_floor: float) -> torch.Tensor:
    """Normalize logits to log probabilities with noise floor truncation."""
    shifted = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    normalized = shifted - torch.log(torch.sum(torch.exp(shifted + EPS), dim=-1, keepdim=True))
    return torch.where(normalized < noise_floor, torch.log(torch.tensor(EPS, device=device)), normalized)

class DSState(NamedTuple):
    emwa_dir: torch.Tensor
    emwa_logp_on_supp: torch.Tensor
    emwa_temp: torch.Tensor
    emwa_ent_scaffold: torch.Tensor
    emwa_ent_naked: torch.Tensor
    emwa_varent_scaffold: torch.Tensor
    emwa_varent_naked: torch.Tensor
    token_cross_ent_scaffold: torch.Tensor
    token_cross_ent_naked: torch.Tensor
    token_cross_var_scaffold: torch.Tensor
    token_cross_var_naked: torch.Tensor
    emwa_dir_ent: torch.Tensor
    emwa_topk_ent_naked: torch.Tensor

def initialize_state(
    logits: torch.Tensor, 
    bsz: int, 
    config: 'DSConfig', 
    dtype=torch.bfloat16
) -> DSState:
    """Initialize DSState from logits."""
    _, seqlen, _ = logits.shape
    logprobs = normalize_logits(logits, config.noise_floor)
    ent, varent = ent_varent(logprobs)
    avg_ent, avg_varent = ent.mean(dim=-1), varent.mean(dim=-1)

    # Get top-k logits and indices
    topk_logits, topk_indices = torch.topk(logprobs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    topk_ent, _ = ent_varent(topk_logprobs)
    avg_topk_ent = topk_ent.mean(dim=-1)

    # Handle Dirichlet support
    logprobs_on_supp = normalize_logits(
        logits[..., config.dirichlet_support], config.noise_floor
    )
    avg_logprobs_on_supp = torch.mean(logprobs_on_supp, dim=1)

    initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
    avg_dir_ent = dirichlet_log_likelihood_from_logprob(
        logprobs_on_supp, initial_dir.unsqueeze(1)
    ).mean(dim=-1)

    # Get token cross entropy stats
    topk_token_logprobs = torch.gather(logprobs, -1, topk_indices)
    initial_cross_ent_naked = -topk_token_logprobs.mean(dim=(1, 2))
    initial_cross_var_naked = topk_token_logprobs.var(dim=(1, 2))

    # Create state with proper device placement
    state = DSState(
        emwa_dir=initial_dir.repeat(bsz, 1).to(device),
        emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, 1).to(device),
        emwa_temp=torch.ones((bsz,), dtype=dtype, device=device),
        emwa_ent_scaffold=avg_ent.repeat(bsz).to(device),
        emwa_ent_naked=avg_ent.repeat(bsz).to(device),
        emwa_varent_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_varent_naked=avg_varent.repeat(bsz).to(device),
        token_cross_ent_scaffold=avg_ent.repeat(bsz).to(device),
        token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz).to(device),
        token_cross_var_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        token_cross_var_naked=initial_cross_var_naked.repeat(bsz).to(device),
        emwa_dir_ent=avg_dir_ent.repeat(bsz).to(device),
        emwa_topk_ent_naked=avg_topk_ent.repeat(bsz).to(device),
    )
    return state