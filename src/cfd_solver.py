
import numpy as np


def uniform_pressure_field(nx=50, ny=30, p_in=2e6, p_out=1e6):
    if nx < 2 or ny < 2:
        raise ValueError("Grid dimensions must be at least 2.")
    return np.tile(np.linspace(p_in, p_out, nx), (ny, 1))


def darcy_velocity(pressure, permeability, viscosity, dx, dy):
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 2 or min(pressure.shape) < 2:
        raise ValueError("Pressure must be a 2D array.")
    if permeability <= 0 or viscosity <= 0 or dx <= 0 or dy <= 0:
        raise ValueError("Physical properties must be positive.")

    dp_dy, dp_dx = np.gradient(pressure, dy, dx)
    return -(permeability / viscosity) * dp_dx, -(permeability / viscosity) * dp_dy
