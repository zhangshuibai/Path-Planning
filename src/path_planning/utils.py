import torch
import random
import numpy as np
from typing import Optional, Tuple, Union

def topk_lowest_masking(scores: torch.Tensor, cutoff_len: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask identifying the k lowest-scoring positions in a tensor.
    
    This function selects positions with the lowest scores, where k is specified by cutoff_len.
    It's used in P2 sampling to determine which positions should be masked.
    
    Args:
        scores: Tensor of shape (batch_size, seq_len) containing scores for each position
        cutoff_len: Tensor of shape (batch_size, 1) specifying how many positions to select in each sequence
    
    Returns:
        Boolean mask tensor of shape (batch_size, seq_len) where True indicates positions with lowest scores
    """
    sorted_index = scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = scores < cutoff
    return masking

def topk_highest_masking(scores: torch.Tensor, cutoff_len: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask identifying the k highest-scoring positions in a tensor.
    
    This function selects positions with the highest scores, where k is specified by cutoff_len.
    It's the opposite of topk_lowest_masking and can be used when you want to select the most confident tokens.
    
    Args:
        scores: Tensor of shape (batch_size, seq_len) containing scores for each position
        cutoff_len: Tensor of shape (batch_size, 1) specifying how many positions to select in each sequence
    
    Returns:
        Boolean mask tensor of shape (batch_size, seq_len) where True indicates positions with highest scores
    """
    sorted_index = scores.sort(-1, descending=True)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = scores >= cutoff
    return masking

def seed_everything(seed: Optional[int] = None) -> None:
    """
    Set the seed for reproducibility across various libraries.
    
    This function sets random seeds for Python's random module, NumPy, and PyTorch
    to ensure reproducible results across runs.
    
    Args:
        seed: Integer seed value. If None, no seeding is performed.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stochastic_sample_from_categorical(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    noise_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample from a categorical distribution with temperature scaling and Gumbel noise.
    
    This function implements stochastic sampling from logits with temperature control.
    When temperature > 0, Gumbel noise is added to introduce randomness in sampling.
    
    Mathematical formulation:
    1. Convert logits to probabilities: p = softmax(logits/temperature + noise_scale * gumbel_noise)
    2. Sample from the resulting distribution
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing unnormalized log probabilities
        temperature: Temperature parameter controlling randomness (higher = more random)
        noise_scale: Scale factor for the Gumbel noise
    
    Returns:
        A tuple containing:
            - tokens: Tensor of shape (batch_size, seq_len) containing sampled token indices
            - scores: Tensor of shape (batch_size, seq_len) containing log probabilities of selected tokens
            - modified_logits: Tensor of shape (batch_size, seq_len, vocab_size) containing temperature-scaled logits
    """
    dtype = logits.dtype
    logits = logits.to(torch.float64)
    if temperature != 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float64) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel_noise
    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores.to(dtype), logits.to(dtype)

