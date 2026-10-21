import numpy as np
from src.solver import solve_temperature


def run_sensitivity(parameter_values, parameter_name):
    results = []

    for value in parameter_values:
        params = {
            "injection_temp": 40.0,
            "velocity": 1e-6,
            "conductivity": 2.5,
        }
        params[parameter_name] = value

        temperature = solve_temperature(**params)
        outlet = temperature[1:-1, -2].mean()
        results.append((value, outlet))

    return np.asarray(results)