import torch
from tqdm import tqdm
from typing import Optional, Any, Callable
from path_planning.utils import topk_lowest_masking, stochastic_sample_from_categorical
from path_planning.score_function import logP, random_score


@torch.inference_mode()
@torch.cuda.amp.autocast()
def p2_sampling(
    xt: torch.Tensor,
    model: Any,
    tokenizer: Any,
    num_steps: int,
    tau: float = 1.0,
    kappa_fn: Callable = lambda t: t,
    eta: float = 1.0,
    planner: Optional[Any] = None,
    score_fn: Callable = logP
) -> torch.Tensor:
    """
    P2 (Path Planning) sampling implementation.
    
    Args:
        xt: Input tensor with masked tokens
        model: Main model for generating logits
        tokenizer: Tokenizer with mask_token_id attribute
        num_steps: Number of sampling steps
        tau: Temperature parameter for sampling
        kappa_fn: Function to compute kappa at each timestep
        eta: Scaling factor for unmask scores
        planner: Optional planner model
        score_type: Type of scoring ('confidence' or 'random')
    
    Returns:
        Sampled sequence tensor
    """
    dt = 1/num_steps
    fix_mask = xt != tokenizer.mask_token_id
    
    for i in tqdm(range(1, num_steps+1)):
        kappa_t = kappa_fn(i*dt)
        logits = model(xt).double()
        last_mask = xt == tokenizer.mask_token_id
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
        
        xt[to_mask] = tokenizer.mask_token_id
        mask_2_x0 = last_mask & ~lowest_k_mask
        xt[mask_2_x0] = x0[mask_2_x0]
    # Fill any remaining masks
    xt[xt == tokenizer.mask_token_id] = x0[xt == tokenizer.mask_token_id]
    return xt 

from functools import partial

ancestral_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=random_score,
    eta=0
)


greedy_ancestral_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=logP,
    eta=1,
)

dfm_sampling = partial(
    p2_sampling,
    planner=None,
    score_fn=random_score,
)



