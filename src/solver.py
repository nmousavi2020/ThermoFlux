
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_temperature(
    injection_temp=40.0,
    velocity=1e-6,
    conductivity=2.5,
    nx=50,
    ny=30,
    length=100.0,
    height=50.0,
    initial_temp=150.0,
    vx=None,
    vy=None,
    rho_cp=2.5e6,
    max_iter=30000,
    tolerance=1e-7,
):
    """Steady 2D advection-diffusion solver using upwind FDM."""

    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must be >= 3.")
    if min(length, height, conductivity, rho_cp) <= 0:
        raise ValueError("Physical parameters must be positive.")
    if not np.isfinite([injection_temp, initial_temp, velocity]).all():
        raise ValueError("Temperatures and velocity must be finite.")

    dx = length / (nx - 1)
    dy = height / (ny - 1)

    def velocity_field(value, default):
        if value is None:
            return np.full((ny, nx), default, dtype=float)
        value = np.asarray(value, dtype=float)
        if value.shape != (ny, nx) or not np.isfinite(value).all():
            raise ValueError("Invalid velocity field.")
        return value

    vx = velocity_field(vx, velocity)
    vy = velocity_field(vy, 0.0)

    # Fixed-temperature boundaries
    T = np.full((ny, nx), initial_temp, dtype=float)
    T[:, 0] = injection_temp
    T[:, -1] = initial_temp
    T[0, :] = initial_temp
    T[-1, :] = initial_temp

    # Interior face velocities
    ue = 0.5 * (vx[1:-1, 1:-1] + vx[1:-1, 2:])
    uw = 0.5 * (vx[1:-1, :-2] + vx[1:-1, 1:-1])
    vp = 0.5 * (vy[1:-1, 1:-1] + vy[2:, 1:-1])
    vm = 0.5 * (vy[:-2, 1:-1] + vy[1:-1, 1:-1])

    # Diffusion plus correctly oriented upwind advection
    ae = conductivity / dx**2 + rho_cp * np.maximum(-ue, 0) / dx
    aw = conductivity / dx**2 + rho_cp * np.maximum(uw, 0) / dx
    apj = conductivity / dy**2 + rho_cp * np.maximum(-vp, 0) / dy
    amj = conductivity / dy**2 + rho_cp * np.maximum(vm, 0) / dy

    nxi = nx - 2
    nyi = ny - 2
    size = nxi * nyi

    A = lil_matrix((size, size), dtype=float)
    b = np.zeros(size)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            row = (j - 1) * nxi + (i - 1)

            coefficients = [
                (i + 1, j, ae[j-1, i-1]),
                (i - 1, j, aw[j-1, i-1]),
                (i, j + 1, apj[j-1, i-1]),
                (i, j - 1, amj[j-1, i-1]),
            ]

            A[row, row] = sum(c for _, _, c in coefficients)

            for ni, nj, coefficient in coefficients:
                if 1 <= ni < nx - 1 and 1 <= nj < ny - 1:
                    col = (nj - 1) * nxi + (ni - 1)
                    A[row, col] -= coefficient
                else:
                    b[row] += coefficient * T[nj, ni]

    solution = spsolve(A.tocsr(), b)

    if not np.isfinite(solution).all():
        raise RuntimeError("Numerical solution contains invalid values.")

    T[1:-1, 1:-1] = solution.reshape(nyi, nxi)
    return T
