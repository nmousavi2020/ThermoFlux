
import numpy as np

from src.cfd_solver import darcy_velocity
from src.solver import solve_temperature


def solve_coupled(
    pressure,
    permeability=1e-13,
    viscosity=1e-3,
    injection_temp=40.0,
    conductivity=2.5,
    rho_cp=2.5e6,
    length=100.0,
    height=50.0,
    initial_temp=150.0,
):
    """Calculate Darcy velocity and use it in the thermal solver."""
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 2:
        raise ValueError("Pressure must be a 2D array.")

    ny, nx = pressure.shape
    dx = length / (nx - 1)
    dy = height / (ny - 1)

    vx, vy = darcy_velocity(
        pressure, permeability, viscosity, dx, dy
    )

    temperature = solve_temperature(
        injection_temp=injection_temp,
        conductivity=conductivity,
        nx=nx,
        ny=ny,
        length=length,
        height=height,
        initial_temp=initial_temp,
        vx=vx,
        vy=vy,
        rho_cp=rho_cp,
    )

    return temperature, vx, vy
