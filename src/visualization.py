import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_temperature(T, filename):
    plt.figure(figsize=(8, 5))
    plt.imshow(T, origin="lower", extent=[0, 100, 0, 50],
               cmap="inferno", aspect="auto")
    plt.colorbar(label="Temperature (°C)")
    plt.xlabel("Distance x (m)")
    plt.ylabel("Height y (m)")
    plt.title("Geothermal Reservoir Temperature")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_pressure(P, filename):
    plt.figure(figsize=(8, 5))
    plt.imshow(P / 1e6, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Pressure (MPa)")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.title("Reservoir Pressure Field")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_velocity(vx, vy, filename):
    speed = np.hypot(vx, vy)
    ny, nx = speed.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    step = max(1, min(nx, ny) // 15)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(speed, origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(image, ax=ax, label="Darcy speed (m/s)")
    ax.quiver(x[::step, ::step], y[::step, ::step],
              vx[::step, ::step], vy[::step, ::step],
              color="cyan")
    ax.set(title="Darcy Velocity Field", xlabel="X grid", ylabel="Y grid")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_validation(y_true, y_pred, filename):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.8)
    ax.plot([low, high], [low, high], "r--", label="Ideal")
    ax.set(xlabel="Simulated outlet temperature (°C)",
           ylabel="Predicted outlet temperature (°C)",
           title="Surrogate Validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_parameter_effect(data, filename):
    parameters = [
        ("injection_temp", "Injection temperature (°C)"),
        ("velocity", "Velocity (m/s)"),
        ("conductivity", "Conductivity (W/m·K)")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (column, label) in zip(axes, parameters):
        ax.scatter(data[column], data["outlet_temp"], alpha=0.65)
        ax.set_xlabel(label)
        ax.set_ylabel("Outlet temperature (°C)")
        ax.grid(alpha=0.25)
    fig.suptitle("Parameter Effects on Outlet Temperature")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_sensitivity(data, filename):
    parameters = [
        ("injection_temp", "Injection temperature"),
        ("velocity", "Velocity"),
        ("conductivity", "Conductivity")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, label) in zip(axes, parameters):
        grouped = data.groupby(
            pd.cut(data[column], bins=8, duplicates="drop")
        )["outlet_temp"].agg(["mean", "std"])
        ax.errorbar(
            np.arange(len(grouped)), grouped["mean"],
            yerr=grouped["std"].fillna(0), marker="o"
        )
        ax.set(title=label, xlabel="Parameter bin",
               ylabel="Outlet temperature (°C)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_gpr_uncertainty(model, X_test, y_test, filename):
    prediction, std = model.predict(X_test, return_std=True)
    order = np.argsort(y_test)
    x = np.arange(len(y_test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.asarray(y_test)[order], "ko", label="Simulation")
    ax.plot(x, prediction[order], "b-", label="GPR prediction")
    ax.fill_between(
        x,
        prediction[order] - 1.96 * std[order],
        prediction[order] + 1.96 * std[order],
        alpha=0.2, label="Approx. 95% interval"
    )
    ax.set(xlabel="Test samples", ylabel="Outlet temperature (°C)",
           title="GPR Prediction and Uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle


def plot_geothermal_system(temperature, output_path):
    ny, nx = temperature.shape
    fig, ax = plt.subplots(figsize=(12, 5))

    image = ax.imshow(
        temperature,
        origin="upper",
        extent=[0, 100, 50, 0],
        cmap="inferno",
        aspect="auto",
    )

    # Reservoir boundaries
    ax.add_patch(Rectangle(
        (0, 0), 100, 50,
        fill=False, edgecolor="black", linewidth=2
    ))

    # Conceptual injection and production wells
    for x, label, color in [
        (8, "Injection well", "deepskyblue"),
        (92, "Production well", "lime"),
    ]:
        ax.plot([x, x], [-8, 50], color=color, linewidth=5)
        ax.plot([x - 2, x + 2], [50, 50],
                color=color, linewidth=7)
        ax.text(
            x, -10, label,
            ha="center", va="bottom",
            color=color, fontsize=10, fontweight="bold"
        )

    # Illustrative flow direction
    for y in [12, 25, 38]:
        ax.annotate(
            "", xy=(80, y), xytext=(20, y),
            arrowprops=dict(
                arrowstyle="->", color="white",
                lw=1.8, alpha=0.9
            )
        )

    ax.set_title("ThermoFlux | Geothermal Reservoir")
    ax.set_xlabel("Reservoir length (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(0, 100)
    ax.set_ylim(50, -14)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Temperature (°C)")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

