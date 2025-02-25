import torch


def logP(logits, x0):
    """
    Compute the log probabilities of predicted tokens from model logits.
    
    Args:
        logits: (batch_size, seq_len, vocab_size) - Raw logits from model
        x0: (batch_size, seq_len) - Token indices to compute probabilities for
        
    Returns:
        scores: (batch_size, seq_len) - Log probabilities for each token
    """
    logits = logits.double()
    logits = logits.log_softmax(dim=-1)
    scores = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    return scores






def random_score(logits, x0):
    """
    Return a random score for each token.
    """
    return torch.rand_like(x0).log()

def diff_top2(logits, x0):
    """
    Return the difference between the top 2 probabilities for each position.
    the score is the difference between the top 2 probabilities.
    the higher the diff, the more confident the model is, the better the score.
    """
    logits = logits.log_softmax(dim=-1)
    top2_logits = logits.topk(2, dim=-1).values
    diff = top2_logits[:, :, 0] - top2_logits[:, :, 1]
    return diff



