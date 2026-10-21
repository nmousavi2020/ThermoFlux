# ============================================================
# ThermoFlux: Coupled Geothermal Flow and Heat Transfer
# CFD Simulation, Surrogate Modeling, and Sensitivity Analysis
#
# Copyright (c) 2026 Naeim Mousavi
# All rights reserved.
#
# ============================================================

print()
print("╭────────────────────────────────────────────────────────────────────╮")
print("│                        ThermoFlux                                  │")
print("│             Geothermal CFD/FEM & Surrogate Modeling                │")
print("│                                                                    │")
print("│                 Copyright © 2026 Naeim Mousavi                     │")
print("│       For methodology and usage, please see the README.            │")
print("╰────────────────────────────────────────────────────────────────────╯")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    NX, NY, LENGTH, HEIGHT, INITIAL_TEMP, RHO_CP,
    PERMEABILITY, VISCOSITY, OUTLET_PRESSURE,
    N_SAMPLES, RANDOM_SEED, TEST_SIZE,
)
from src.cfd_solver import uniform_pressure_field, darcy_velocity
from src.solver import solve_temperature
from src.surrogate import train_surrogate
from src.evaluation import evaluate_model, save_metrics
from src.visualization import (
    plot_temperature, plot_pressure, plot_velocity,
    plot_validation, plot_parameter_effect, plot_sensitivity,
    plot_gpr_uncertainty, plot_geothermal_system,
)


def main():
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    dx = LENGTH / (NX - 1)
    dy = HEIGHT / (NY - 1)

    rows = []
    reference_temperature = None
    reference_pressure = None
    reference_vx = None
    reference_vy = None

    for i in range(N_SAMPLES):
        injection = rng.uniform(20.0, 80.0)
        conductivity = rng.uniform(1.5, 4.0)
        p_in = rng.uniform(1.5e6, 3.0e6)

        pressure = uniform_pressure_field(
            NX, NY, p_in, OUTLET_PRESSURE
        )

        vx, vy = darcy_velocity(
            pressure, PERMEABILITY, VISCOSITY, dx, dy
        )

        velocity_value = float(np.mean(np.hypot(vx, vy)))

        temperature = solve_temperature(
            injection_temp=injection,
            conductivity=conductivity,
            nx=NX,
            ny=NY,
            length=LENGTH,
            height=HEIGHT,
            initial_temp=INITIAL_TEMP,
            vx=vx,
            vy=vy,
            rho_cp=RHO_CP,
        )

        outlet = float(temperature[1:-1, -2].mean())

        if not np.isfinite(outlet):
            raise ValueError(f"Invalid outlet temperature at sample {i}.")

        rows.append([
            injection, velocity_value, conductivity, outlet
        ])

        if i == 0:
            reference_temperature = temperature.copy()
            reference_pressure = pressure.copy()
            reference_vx = vx.copy()
            reference_vy = vy.copy()

    data = pd.DataFrame(
        rows,
        columns=[
            "injection_temp", "velocity",
            "conductivity", "outlet_temp",
        ],
    )
    data.to_csv(out / "simulation_data.csv", index=False)

    print("\nOutlet temperature statistics:")
    print(data["outlet_temp"].describe())

    if data["outlet_temp"].nunique() < 2:
        raise ValueError("Outlet temperatures are constant.")

    X = data[
        ["injection_temp", "velocity", "conductivity"]
    ].to_numpy(dtype=float)
    y = data["outlet_temp"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = train_surrogate(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)
    save_metrics(metrics, out / "metrics.json")

    print(f"\nMean Darcy speed: {data['velocity'].mean():.3e} m/s")
    print(f"Outlet temperature range: {y.min():.4f}–{y.max():.4f} °C")
    print(f"MAE:  {metrics['MAE']:.6f} °C")
    print(f"RMSE: {metrics['RMSE']:.6f} °C")
    print(f"R²:   {metrics['R2']:.6f}")

    plot_temperature(reference_temperature, out / "temperature_field.png")
    plot_pressure(reference_pressure, out / "pressure_field.png")
    plot_velocity(reference_vx, reference_vy, out / "velocity_field.png")
    plot_validation(y_test, predictions, out / "surrogate_validation.png")
    plot_parameter_effect(data, out / "parameter_effects.png")
    plot_sensitivity(data, out / "sensitivity.png")
    plot_gpr_uncertainty(model, X_test, y_test, out / "gpr_uncertainty.png")

    # New conceptual geothermal well visualization
    plot_geothermal_system(
        reference_temperature,
        out / "geothermal_system.png",
    )

    print(f"\nAll results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
