import numpy as np
import skopt
from skopt import gp_minimize
from skopt.space import Real

# Define function with uniform noise
def expensive_function(x, eps=2):
    """Noisy function where noise is non-Gaussian (e.g., Uniformly distributed)."""
    x = np.array(x)
    noise = np.random.uniform(-1, 1) * eps
    newx = x[0] + noise
    noised_value = np.sqrt(newx) if newx >= 0 else -newx
    return noised_value

def objective(x, num_samples=100):
    samples = np.stack([expensive_function(x) for _ in range(num_samples)])
    maxima = np.max(samples, axis=0)
    return maxima

for _ in range(10):
    # Optimize with Bayesian Optimization
    result = gp_minimize(objective,  # Function to minimize
                         [Real(-2.0, 2.0)],  # Search space
                         acq_func="EI",  # Expected Improvement
                         n_calls=20, n_initial_points=5, random_state=42,
                         n_jobs=1)

    print(f"Optimal x*: {result.x[0]}")
