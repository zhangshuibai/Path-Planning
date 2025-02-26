import torch
from tqdm import tqdm
from typing import Optional, Any, Callable, Union
from path_planning.utils import topk_lowest_masking, stochastic_sample_from_categorical
from path_planning.score_function import logP, random_score


@torch.inference_mode()
@torch.cuda.amp.autocast()
def p2_sampling(
    xt: torch.Tensor,
    model: Any,
    mask_id: int,
    num_steps: int,
    tau: float = 1.0,
    kappa_fn: Callable[[float], float] = lambda t: t,
    eta: float = 1.0,
    planner: Optional[Any] = None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = logP
) -> torch.Tensor:
    """
    P2 (Path Planning) sampling implementation.
    
    This function implements the P2 sampling algorithm, a guided diffusion method
    for sequence generation. It starts with a fully or partially masked sequence and
    progressively unmasks tokens based on model confidence and a scheduling function.
    
    Algorithm:
    
    1. Start with a masked sequence x_T
    2. For each step t from T to 1:
       - Forward pass: compute logits = model(x_t)
       - Sample: x_0 ~ softmax(logits/τ)
       - Compute scores using score_fn
       - Modify scores for tokens that were previously unmasked (multiplied by η)
       - Compute κ_t = kappa_fn(t/T) to determine the fraction of tokens to keep unmasked
       - Mask the lowest-scoring tokens to maintain κ_t fraction unmasked
       - Replace masked tokens with sampled ones
    3. Return the final sequence
    
    Args:
        xt: Input tensor with masked tokens, shape (batch_size, seq_len)
        model: Main model for generating logits. Should return logits when called with a sequence
        mask_id: ID of the mask token
        num_steps: Number of sampling steps
        tau: Temperature parameter for sampling. Higher values increase diversity
        kappa_fn: Function to compute kappa at each timestep t ∈ [0,1]. Determines unmasking schedule
        eta: stochasticity strength, higher values make the sampling more stochastic.
        planner: Optional planner model for guided generation
        score_fn: Scoring function for token selection (e.g., logP, random_score, diff_top2)
    
    Returns:
        Sampled sequence tensor of shape (batch_size, seq_len)
    
    References:
        - P2 Sampling: https://arxiv.org/pdf/2502.03540
    """
    dt = 1/num_steps
    fix_mask = xt != mask_id
    
    for i in tqdm(range(1, num_steps+1)):
        kappa_t = kappa_fn(i*dt)
        logits = model(xt).double()
        last_mask = xt == mask_id
        unmask_t = ~last_mask & ~fix_mask
        
        x0, logp, logits = stochastic_sample_from_categorical(logits, temperature=tau)
        if planner is not None:
            planner_logits = planner(x0).double()
            planner_logp = planner_logits.log_softmax(dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            logits[unmask_t] = planner_logits[unmask_t]
            logp[unmask_t] = planner_logp[unmask_t]
        score = score_fn(logits, x0)
        score = score.masked_fill(fix_mask, float('inf'))
        
        score[unmask_t] = score[unmask_t] * eta
        
        num_to_mask = ((~fix_mask).sum(1, keepdim=True).float() * (1 - kappa_t)).long()
        lowest_k_mask = topk_lowest_masking(score, num_to_mask)
        to_mask = lowest_k_mask
        
        xt[to_mask] = mask_id
        mask_2_x0 = last_mask & ~lowest_k_mask
        xt[mask_2_x0] = x0[mask_2_x0]
    # Fill any remaining masks
    xt[xt == mask_id] = x0[xt == mask_id]
    return xt 

from functools import partial

ancestral_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=random_score,
    eta=0
)
ancestral_sampling.__doc__ = """
Ancestral sampling using the P2 framework.

This is a specialized version of P2 sampling that uses random scores and no 
eta parameter, resulting in a pure diffusion sampling approach where token 
selection is random rather than based on model confidence.

Args:
    Same as p2_sampling, except:
    - planner is fixed to None
    - score_fn is fixed to random_score
    - eta is fixed to 0

Returns:
    Sampled sequence tensor of shape (batch_size, seq_len)
"""


greedy_ancestral_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=logP,
    eta=1,
)
greedy_ancestral_sampling.__doc__ = """
Greedy ancestral sampling using the P2 framework.

This variant uses log probabilities as scores but has no planner model.
It selects tokens based on model confidence, making it more deterministic
than pure ancestral sampling.

Args:
    Same as p2_sampling, except:
    - planner is fixed to None
    - score_fn is fixed to logP
    - eta is fixed to 1

Returns:
    Sampled sequence tensor of shape (batch_size, seq_len)
"""


dfm_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=random_score,
)
dfm_sampling.__doc__ = """
Diffusion Masked Language Model (DFM) sampling.

This is a variant of P2 sampling aligned with the DFM approach,
using random scores but no planner model.

Args:
    Same as p2_sampling, except:
    - planner is fixed to None
    - score_fn is fixed to random_score

Returns:
    Sampled sequence tensor of shape (batch_size, seq_len)
"""



