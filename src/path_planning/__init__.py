"""
Path Planning (P2) Sampling

A Python package implementing P2 (Path Planning) sampling, a guided diffusion method for sequence generation.
"""

# Core sampling functions
from .p2 import (
    p2_sampling,
    ancestral_sampling,
    greedy_ancestral_sampling,
    dfm_sampling
)

# Utility functions
from .utils import (
    seed_everything,
    topk_lowest_masking,
    topk_highest_masking,
    stochastic_sample_from_categorical
)

# Score functions
from .score_function import (
    logP,
    random_score,
    diff_top2
)

# Scheduler functions
from .scheduler import (
    linear_scheduler,
    sine_scheduler,
    geometric_scheduler,
    log_scheduler,
    poly2_scheduler,
    poly05_scheduler
)

# Define what should be available directly upon import
__all__ = [
    # Core sampling
    "p2_sampling",
    "ancestral_sampling",
    "greedy_ancestral_sampling",
    "dfm_sampling",
    
    # Utilities
    "seed_everything",
    "topk_lowest_masking",
    "topk_highest_masking",
    "stochastic_sample_from_categorical",
    
    # Score functions
    "logP",
    "random_score",
    "diff_top2",
    
    # Schedulers
    "linear_scheduler",
    "sine_scheduler",
    "geometric_scheduler",
    "log_scheduler", 
    "poly2_scheduler",
    "poly05_scheduler"
]
