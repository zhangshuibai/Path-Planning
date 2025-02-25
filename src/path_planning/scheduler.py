"""
Time-dependent schedulers for P2 sampling.

The scheduler is a non-decreasing function that takes a time step t [0, 1] and returns a scalar [0, 1].
f(0) = 0, f(1) = 1

"""
import math
import torch
import matplotlib.pyplot as plt

def linear_scheduler(t):
    return t

def sine_scheduler(t:float):
    """
    Sine scheduler that maps t from [0,1] to [0,1]
    """
    return math.sin(t * math.pi / 2)

def geometric_scheduler(t:float):
    """
    Geometric scheduler that maps t from [0,1] to [0,1]
    """
    return 1-(1-t)**2

def log_scheduler(t:float):
    """
    Logarithmic scheduler that maps t from [0,1] to [0,1]
    """
    return math.log(t+1)/math.log(2)

def poly2_scheduler(t:float):
    """
    Polynomial scheduler that maps t from [0,1] to [0,1]
    """
    return t**2

def poly05_scheduler(t:float):
    """
    Polynomial scheduler that maps t from [0,1] to [0,1]
    """
    return t**0.5

if __name__ == "__main__":
    def plot_schedulers( t_max=100):
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
