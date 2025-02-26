"""
Time-dependent schedulers for P2 sampling.

The scheduler is a non-decreasing function that takes a time step t [0, 1] and returns a scalar [0, 1].
f(0) = 0, f(1) = 1

These schedulers control the rate at which tokens are unmasked during P2 sampling. Different
schedulers produce different dynamics in the sampling process, potentially affecting the quality
of the generated sequences.
"""
import math
import torch
import matplotlib.pyplot as plt
from typing import List, Callable, Union, Optional


def linear_scheduler(t: float) -> float:
    """
    Linear scheduler that increases proportionally with time.
    
    This is the simplest scheduler, where the proportion of tokens unmasked
    increases linearly with time.
    
    Mathematical formulation:
    f(t) = t
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return t


def sine_scheduler(t: float) -> float:
    """
    Sine scheduler that maps t from [0,1] to [0,1] with a smooth S-curve.
    
    This scheduler starts slow, accelerates in the middle, and then slows down
    at the end, following a sine curve. It provides a smooth transition between
    the fully masked and fully unmasked states.
    
    Mathematical formulation:
    f(t) = sin(t * π/2)
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return math.sin(t * math.pi / 2)


def geometric_scheduler(t: float) -> float:
    """
    Geometric scheduler that maps t from [0,1] to [0,1] with accelerating progress.
    
    This scheduler starts slow and accelerates, following a quadratic curve.
    It tends to keep more tokens masked at the beginning compared to linear.
    
    Mathematical formulation:
    f(t) = 1-(1-t)²
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return 1-(1-t)**2


def log_scheduler(t: float) -> float:
    """
    Logarithmic scheduler that maps t from [0,1] to [0,1].
    
    This scheduler progresses quickly at the beginning and slows down toward the end.
    It tends to unmask more tokens early in the process.
    
    Mathematical formulation:
    f(t) = log(t+1)/log(2)
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return math.log(t+1)/math.log(2)


def poly2_scheduler(t: float) -> float:
    """
    Polynomial (quadratic) scheduler that maps t from [0,1] to [0,1].
    
    This scheduler starts slow and accelerates, following a quadratic curve.
    It's similar to the geometric scheduler but with different dynamics.
    
    Mathematical formulation:
    f(t) = t²
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return t**2


def poly05_scheduler(t: float) -> float:
    """
    Polynomial (square root) scheduler that maps t from [0,1] to [0,1].
    
    This scheduler starts fast and slows down, following a square root curve.
    It tends to unmask more tokens early in the process.
    
    Mathematical formulation:
    f(t) = t^0.5
    
    Args:
        t: Time step in [0, 1]
        
    Returns:
        Float in [0, 1] representing progress of the unmasking process
    """
    return t**0.5


if __name__ == "__main__":
    def plot_schedulers(t_max: int = 100) -> None:
        """
        Plot all scheduler functions for comparison.
        
        Args:
            t_max: Number of time steps to plot
        """
        t = torch.linspace(0, 1, t_max)
        plt.plot(t, [linear_scheduler(float(i)) for i in t], label='linear')
        plt.plot(t, [sine_scheduler(float(i)) for i in t], label='sine')
        plt.plot(t, [geometric_scheduler(float(i)) for i in t], label='geometric')
        plt.plot(t, [poly2_scheduler(float(i)) for i in t], label='poly2')
        plt.plot(t, [poly05_scheduler(float(i)) for i in t], label='poly0.5')
        plt.plot(t, [log_scheduler(float(i)) for i in t], label='log')
        plt.legend()
        plt.savefig('schedulers.png')

    plot_schedulers()
