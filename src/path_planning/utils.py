import torch
import random
import numpy as np

def topk_lowest_masking(scores, cutoff_len):
    sorted_index = scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = scores < cutoff
    return masking
def topk_highest_masking(scores, cutoff_len):
    sorted_index = scores.sort(-1, descending=True)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = scores >= cutoff
    return masking

def seed_everything(seed):
    """
    Set the seed for reproducibility across various libraries.
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




def stochastic_sample_from_categorical(logits, temperature=1.0, noise_scale=1.0):
    logits = logits.double()
    if temperature != 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel_noise
    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores, logits

