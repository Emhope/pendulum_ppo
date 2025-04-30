import numpy as np


def sigmoid(x, value_at_1, gaussian=True):
    if gaussian:
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x*scale)**2)
    else:
        scale = np.sqrt(1-value_at_1)
        scaled_x = x*scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

def tolerance(x, bounds=(0.0, 0.0), margin=0.0,value_at_margin=0.1, gauss_sigm=True):
    lower, upper = bounds
    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, sigmoid(d, value_at_margin, gaussian=gauss_sigm))

    return float(value) if np.isscalar(x) else value
