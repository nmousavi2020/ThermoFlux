from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import streamlit as st

st.set_page_config(
    page_title="ThermoFlux | Geothermal Simulation",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ===== GLOBAL COLORS ===== */
    :root {
        color-scheme: light;
    }

    .stApp {
        background: #F1F5F9 !important;
        color: #111827 !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 2px solid #475569 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #CBD5E1 !important;
    }

    /* ===== INPUT FIELDS ===== */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #FFFFFF !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border: 2px solid #64748B !important;
        border-radius: 6px !important;
    }

    /* Number input controls */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #64748B !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* ===== SLIDERS ===== */
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] * {
        color: #FFFFFF !important;
    }

    /* ===== ALL STANDARD TABS ===== */
    [data-baseweb="tab-list"] {
        background: #111827 !important;
        padding: 8px !important;
        gap: 6px !important;
        border-radius: 10px !important;
    }

    [data-baseweb="tab"] {
        background: #334155 !important;
        border: 1px solid #94A3B8 !important;
        border-radius: 7px !important;
        padding: 10px 14px !important;
    }

    [data-baseweb="tab"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: #FDE047 !important;
        border: 2px solid #FFFFFF !important;
    }

    [data-baseweb="tab"][aria-selected="true"] * {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    /* ===== RADIO-BASED NAVIGATION ===== */
    [data-testid="stRadio"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stRadio"] > div {
        gap: 8px !important;
    }

    [data-testid="stRadio"] label > div {
        border-color: #FDE047 !important;
    }

    /* ===== SIDEBAR PAGE NAVIGATION ===== */
    [data-testid="stSidebarNav"] {
        background: #111827 !important;
    }

    [data-testid="stSidebarNav"] a {
        background: #334155 !important;
        border: 1px solid #64748B !important;
        border-radius: 7px !important;
        margin: 4px 0 !important;
    }

    [data-testid="stSidebarNav"] a * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #FDE047 !important;
        border: 2px solid #FFFFFF !important;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] * {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    /* ===== BUTTONS ===== */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: #FDE047 !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] button[kind="primary"] * {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    /* ===== KEYBOARD FOCUS ===== */
    button:focus-visible,
    a:focus-visible,
    input:focus-visible {
        outline: 3px solid #FDE047 !important;
        outline-offset: 3px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ThermoFlux | Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "results" / "models"
MODEL_PATH = MODEL_DIR / "thermoflux_gpr.joblib"

st.set_page_config(
    page_title="ThermoFlux | Geothermal Simulation",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0B1220;
        color: #E5EDF7;
    }
    [data-testid="stSidebar"] {
        background: #101B2B;
        border-right: 1px solid #26364D;
    }
    [data-testid="stMetric"] {
        background: #111D30;
        border: 1px solid #26364D;
        border-radius: 12px;
        padding: 15px;
    }

    /* Make metric titles white */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label {
        color: #FFFFFF !important;
    }

    /* Keep metric values white too, if desired */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    [data-testid="stMetricValue"] {
        color: #43D6BE;
    }
    .hero {
        background: linear-gradient(115deg, #12323C, #14243A);
        border: 1px solid #294858;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 18px;
    }
    .hero h1 {
        color: #F0FDFA;
        margin: 0;
        font-size: 2.2rem;
    }
    .hero p {
        color: #B9CED9;
        margin-bottom: 0;
    }
    .eyebrow {
        color: #43D6BE;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Steady-state 2D FDM solver
# ============================================================

def solve_steady_state(
    length_m: float,
    height_m: float,
    nx: int,
    ny: int,
    injection_temp_c: float,
    initial_temp_c: float,
    conductivity_w_mk: float,
    permeability_m2: float,
    viscosity_pa_s: float,
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    volumetric_heat_capacity_j_m3k: float,
    max_iterations: int = 12000,
    tolerance: float = 1e-7,
) -> dict:
    """
    Educational steady-state 2D advection-diffusion solver.

    Equation:
        rho_cp * (v dot grad(T)) = div(k * grad(T))

    Pressure is prescribed as linear in x.
    Darcy velocity is derived analytically.

    Boundary conditions:
        Left: fixed injection temperature.
        Top and bottom: fixed initial/rock temperature.
        Right: zero-normal-gradient temperature.

    This is not a validated coupled geothermal reservoir solver.
    """

    values = [
        length_m, height_m, injection_temp_c, initial_temp_c,
        conductivity_w_mk, permeability_m2, viscosity_pa_s,
        inlet_pressure_pa, outlet_pressure_pa,
        volumetric_heat_capacity_j_m3k,
    ]

    if not np.all(np.isfinite(values)):
        raise ValueError("All physical parameters must be finite.")
    if length_m <= 0 or height_m <= 0:
        raise ValueError("Reservoir dimensions must be positive.")
    if nx < 5 or ny < 5:
        raise ValueError("Grid dimensions must be at least 5.")
    if conductivity_w_mk <= 0:
        raise ValueError("Thermal conductivity must be positive.")
    if permeability_m2 < 0 or viscosity_pa_s <= 0:
        raise ValueError(
            "Permeability must be nonnegative and viscosity positive."
        )
    if volumetric_heat_capacity_j_m3k <= 0:
        raise ValueError("Volumetric heat capacity must be positive.")
    if inlet_pressure_pa < outlet_pressure_pa:
        raise ValueError(
            "Inlet pressure must be greater than or equal to outlet pressure."
        )

    start = time.perf_counter()

    x = np.linspace(0.0, length_m, nx)
    y = np.linspace(0.0, height_m, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Prescribed linear pressure field.
    pressure_x = np.linspace(
        inlet_pressure_pa, outlet_pressure_pa, nx
    )
    pressure = np.broadcast_to(pressure_x, (ny, nx)).copy()

#    dp_dx = (
#        outlet_pressure_pa - inlet_pressure_pa
#    ) / length_m

#    darcy_x = -permeability_m2 / viscosity_pa_s * dp_dx
#    darcy_y = 0.0


    # Pressure gradient (Pa/m)
    dp_dx = (
        outlet_pressure_pa - inlet_pressure_pa
    ) / length_m

    # Darcy velocity (m/s)
    darcy_x = (
        -permeability_m2 / viscosity_pa_s
    ) * dp_dx

    darcy_y = 0.0

    # Verify the calculated velocity
    print("Permeability (m²):", permeability_m2)
    print("Viscosity (Pa·s):", viscosity_pa_s)
    print("Pressure gradient (Pa/m):", dp_dx)
    print("Darcy velocity (m/s):", darcy_x)



    # Diagnostic validation
    if not np.isfinite(darcy_x):
        raise ValueError("Calculated Darcy velocity is not finite.")


    # Constant properties.
    kx = conductivity_w_mk / dx**2
    ky = conductivity_w_mk / dy**2
    adv = (
        volumetric_heat_capacity_j_m3k
        * abs(darcy_x) / dx
    )

    # Initial guess and boundary conditions.
    T = np.full((ny, nx), initial_temp_c, dtype=float)
    T[:, 0] = injection_temp_c
    T[0, :] = initial_temp_c
    T[-1, :] = initial_temp_c
    T[:, 0] = injection_temp_c

    # Gauss-Seidel-style iterative update using the previous
    # complete field at each iteration (Jacobi iteration).
    # For positive Darcy velocity, use upwind values from the left.
    # For negative velocity, use upwind values from the right.
    iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        old = T.copy()

        # Zero-gradient condition at the production-side boundary.
        old[:, -1] = old[:, -2]

        left = old[1:-1, :-2]
        right = old[1:-1, 2:]
        up = old[:-2, 1:-1]
        down = old[2:, 1:-1]

        if darcy_x >= 0:
            upwind = left
        else:
            upwind = right

        denominator = 2.0 * kx + 2.0 * ky + adv

        interior = (
            kx * (left + right)
            + ky * (up + down)
            + adv * upwind
        ) / max(denominator, 1e-30)

        T[1:-1, 1:-1] = interior

        # Reapply all boundary conditions.
        T[:, 0] = injection_temp_c
        T[0, :] = initial_temp_c
        T[-1, :] = initial_temp_c
        T[:, 0] = injection_temp_c
        T[:, -1] = T[:, -2]

        iterations = iteration
        error = float(np.max(np.abs(T - old)))

        if error < tolerance:
            converged = True
            break

    # Pressure gradient and Darcy velocity fields.
    U = np.full_like(T, darcy_x)
    V = np.full_like(T, darcy_y)
    speed = np.hypot(U, V)

    # Mean temperature along the production-side boundary.
    outlet_temperature_c = float(np.mean(T[:, -1]))

    wall_time_s = time.perf_counter() - start

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "temperature": T,
        "pressure": pressure,
        "U": U,
        "V": V,
        "speed": speed,
        "darcy_x_m_s": float(darcy_x),
        "darcy_y_m_s": float(darcy_y),
        "mean_speed_m_s": float(np.mean(speed)),
        "max_speed_m_s": float(np.max(speed)),
        "outlet_temperature_c": outlet_temperature_c,
        "mean_temperature_c": float(np.mean(T)),
        "pressure_gradient_pa_m": float(dp_dx),
        "wall_time_s": wall_time_s,
        "iterations": iterations,
        "converged": converged,
        "final_residual": error,
        "dx_m": dx,
        "dy_m": dy,
    }


# ============================================================
# Plotting functions
# ============================================================


def style_figure(fig, height=440):
    fig.update_layout(
        height=height,

        # High-contrast backgrounds
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        # All plot text in black
        font=dict(
            color="#111827",
            family="Arial",
            size=13,
        ),

        # Title styling
        title_font=dict(
            color="#111827",
            size=20,
        ),

        # Layout spacing
        margin=dict(l=60, r=30, t=70, b=60),

        # Legend text in black
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                color="#111827",
                size=12,
            ),
            bgcolor="#FFFFFF",
        ),
    )

    # Axis labels and tick numbers in black
    fig.update_xaxes(
        color="#111827",
        title_font=dict(color="#111827", size=14),
        tickfont=dict(color="#111827", size=12),
        gridcolor="#D1D5DB",
        linecolor="#64748B",
        zerolinecolor="#9CA3AF",
        showline=True,
    )

    fig.update_yaxes(
        color="#111827",
        title_font=dict(color="#111827", size=14),
        tickfont=dict(color="#111827", size=12),
        gridcolor="#D1D5DB",
        linecolor="#64748B",
        zerolinecolor="#9CA3AF",
        showline=True,
    )

    return fig



def field_figure(result, field):
    if field == "Temperature":
        z = result["temperature"]
        scale = "Inferno"
        unit = "Temperature (°C)"

    elif field == "Pressure":
        z = result["pressure"] / 1e6
        scale = "Viridis"
        unit = "Pressure (MPa)"

    else:
        # Convert m/s to micrometres/s for readability
        z = result["speed"] * 1e6
        scale = "Blues"
        unit = "Darcy speed (µm/s)"

    fig = go.Figure(
        go.Heatmap(
            x=result["x"],
            y=result["y"],
            z=z,
            colorscale=scale,
            colorbar=dict(title=unit),
            hovertemplate=(
                "x: %{x:.2f} m<br>"
                "y: %{y:.2f} m<br>"
                + unit + ": %{z:.6g}<extra></extra>"
            ),
        )
    )

    fig.update_layout(title=f"{field} Field")
    fig.update_xaxes(title="Reservoir length (m)")
    fig.update_yaxes(
        title="Reservoir height (m)",
        scaleanchor="x",
        scaleratio=1,
    )

    return style_figure(fig)


def well_figure(result):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=result["x"],
        y=result["y"],
        z=result["temperature"],
        colorscale="Inferno",
        colorbar=dict(title="Temperature (°C)"),
    ))

    fig.add_trace(go.Scatter(
        x=[0, result["x"][-1]],
        y=[
            result["y"][-1] * 0.7,
            result["y"][-1] * 0.7,
        ],
        mode="markers+text",
        marker=dict(
            size=14,
            color=["#60A5FA", "#F59E0B"],
            symbol=["triangle-down", "triangle-up"],
            line=dict(color="white", width=1),
        ),
        text=["INJECTION WELL", "PRODUCTION WELL"],
        textposition="top center",
        name="Conceptual well locations",
    ))

    fig.update_layout(title="Conceptual Reservoir and Well Layout")
    fig.update_xaxes(title="Reservoir length (m)")
    fig.update_yaxes(
        title="Reservoir height (m)",
        scaleanchor="x",
        scaleratio=1,
    )
    return style_figure(fig)



def velocity_figure(result):
    # Convert m/s to micrometres/s for easier visualization
    speed_um_s = result["speed"] * 1e6

    fig = go.Figure(
        go.Heatmap(
            x=result["x"],
            y=result["y"],
            z=speed_um_s,
            colorscale="Blues",
            colorbar=dict(title="Speed (µm/s)"),
            zmin=0,
        )
    )

    xmid = result["x"][-1] * 0.5
    ymid = result["y"][-1] * 0.5
    direction = 1 if result["darcy_x_m_s"] >= 0 else -1

    fig.add_annotation(
        x=xmid + direction * result["x"][-1] * 0.15,
        y=ymid,
        ax=xmid - direction * result["x"][-1] * 0.15,
        ay=ymid,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="Darcy flow",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#111827",
        font=dict(color="#111827"),
    )

    fig.update_layout(
        title="Darcy Velocity Magnitude and Direction",
        xaxis_title="Reservoir length (m)",
        yaxis_title="Reservoir height (m)",
    )

    return style_figure(fig)



# ============================================================
# Gaussian Process Regression
# ============================================================

def make_gpr():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(3), nu=2.5)
        + WhiteKernel(noise_level=1e-4)
    )

    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=42,
        ),
    )


def generate_dataset(
    n_samples,
    seed,
    length_m,
    height_m,
    nx,
    ny,
    initial_temp_c,
    viscosity_pa_s,
    inlet_pressure_pa,
    outlet_pressure_pa,
    heat_capacity,
):
    """Generate solver-derived samples for surrogate training."""

    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n_samples):
        tin = float(rng.uniform(20.0, 110.0))
        conductivity = float(rng.uniform(0.8, 5.0))
        permeability_md = float(10 ** rng.uniform(-1.0, 2.0))

        result = solve_steady_state(
            length_m=length_m,
            height_m=height_m,
            nx=nx,
            ny=ny,
            injection_temp_c=tin,
            initial_temp_c=initial_temp_c,
            conductivity_w_mk=conductivity,
            permeability_m2=permeability_md * 9.869233e-16,
            viscosity_pa_s=viscosity_pa_s,
            inlet_pressure_pa=inlet_pressure_pa,
            outlet_pressure_pa=outlet_pressure_pa,
            volumetric_heat_capacity_j_m3k=heat_capacity,
            max_iterations=5000,
            tolerance=1e-5,
        )

        records.append({
            "injection_temp_c": tin,
            "darcy_speed_m_s": abs(result["darcy_x_m_s"]),
            "conductivity_w_mk": conductivity,
            "outlet_temp_c": result["outlet_temperature_c"],
        })

    return pd.DataFrame(records)


def train_and_evaluate(data, test_fraction=0.25, seed=42):
    features = [
        "injection_temp_c",
        "darcy_speed_m_s",
        "conductivity_w_mk",
    ]

    X = data[features].to_numpy()
    y = data["outlet_temp_c"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
    )

    model = make_gpr()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions, uncertainty = model.predict(
        X_test,
        return_std=True,
    )
    prediction_time = time.perf_counter() - start

    metrics = {
        "MAE (°C)": mean_absolute_error(y_test, predictions),
        "RMSE (°C)": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R²": r2_score(y_test, predictions),
    }

    return {
        "model": model,
        "metrics": metrics,
        "actual": y_test,
        "predicted": predictions,
        "uncertainty": uncertainty,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "features": features,
    }


def load_saved_model():
    if not MODEL_PATH.exists():
        return None
    try:
        # Only load locally generated, trusted model files.
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def surrogate_plot(evaluation):
    actual = evaluation["actual"]
    predicted = evaluation["predicted"]
    std = evaluation["uncertainty"]

    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[low, high],
        y=[low, high],
        mode="lines",
        name="Ideal prediction",
        line=dict(color="#98A9BF", dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode="markers",
        name="Independent test samples",
        error_y=dict(
            type="data",
            array=1.96 * std,
            visible=True,
        ),
        marker=dict(
            color="#43D6BE",
            size=9,
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            "Actual: %{x:.2f} °C<br>"
            "Predicted: %{y:.2f} °C<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Actual vs Predicted Outlet Temperature",
        xaxis_title="Simulated outlet temperature (°C)",
        yaxis_title="GPR prediction (°C)",
    )
    return style_figure(fig)


# ============================================================
# SIMULATION RESULTS
# ============================================================


# ============================================================
# SIDEBAR — SIMULATION PARAMETERS
# ============================================================

st.sidebar.title("ThermoFlux")
st.sidebar.caption("Geothermal Reservoir Simulation")

st.sidebar.subheader("Reservoir Geometry")

length_m = st.sidebar.number_input(
    "Reservoir length (m)",
    min_value=10.0,
    max_value=5000.0,
    value=500.0,
    step=50.0,
)

height_m = st.sidebar.number_input(
    "Reservoir height (m)",
    min_value=10.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
)

nx = st.sidebar.slider(
    "Grid points along length",
    min_value=10,
    max_value=100,
    value=40,
)

ny = st.sidebar.slider(
    "Grid points along height",
    min_value=10,
    max_value=60,
    value=25,
)

st.sidebar.subheader("Thermal Properties")

injection_temp_c = st.sidebar.number_input(
    "Injection temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=40.0,
    step=5.0,
)

initial_temp_c = st.sidebar.number_input(
    "Initial reservoir temperature (°C)",
    min_value=0.0,
    max_value=350.0,
    value=150.0,
    step=5.0,
)

conductivity_w_mk = st.sidebar.number_input(
    "Thermal conductivity (W/m·K)",
    min_value=0.1,
    max_value=20.0,
    value=2.5,
    step=0.1,
)

heat_capacity = st.sidebar.number_input(
    "Volumetric heat capacity (J/m³·K)",
    min_value=100000.0,
    max_value=10000000.0,
    value=2500000.0,
    step=100000.0,
)

st.sidebar.subheader("Flow Properties")

permeability_md = st.sidebar.number_input(
    "Permeability (mD)",
    min_value=0.0,
    max_value=10000.0,
    value=10.0,
    step=1.0,
)

viscosity_pa_s = st.sidebar.number_input(
    "Dynamic viscosity (Pa·s)",
    min_value=0.00001,
    max_value=10.0,
    value=0.001,
    format="%.5f",
)

inlet_pressure_mpa = st.sidebar.number_input(
    "Inlet pressure (MPa)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
)

outlet_pressure_mpa = st.sidebar.number_input(
    "Outlet pressure (MPa)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
)

max_iterations = st.sidebar.number_input(
    "Maximum solver iterations",
    min_value=100,
    max_value=20000,
    value=5000,
    step=500,
)

run_button = st.sidebar.button(
    "Run Simulation",
    type="primary",
    use_container_width=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None

if "simulation_parameters" not in st.session_state:
    st.session_state.simulation_parameters = None

if "gpr_evaluation" not in st.session_state:
    st.session_state.gpr_evaluation = None

if "gpr_dataset" not in st.session_state:
    st.session_state.gpr_dataset = None

if "sensitivity_df" not in st.session_state:
    st.session_state.sensitivity_df = None

# ============================================================
# RUN SIMULATION
# ============================================================

if run_button:
    if inlet_pressure_mpa < outlet_pressure_mpa:
        st.sidebar.error(
            "Inlet pressure must be greater than or equal "
            "to outlet pressure."
        )
    else:
        try:
            with st.spinner("Running steady-state simulation..."):
                new_result = solve_steady_state(
                    length_m=length_m,
                    height_m=height_m,
                    nx=int(nx),
                    ny=int(ny),
                    injection_temp_c=injection_temp_c,
                    initial_temp_c=initial_temp_c,
                    conductivity_w_mk=conductivity_w_mk,
                    permeability_m2=permeability_md * 9.869233e-16,
                    viscosity_pa_s=viscosity_pa_s,
                    inlet_pressure_pa=inlet_pressure_mpa * 1e6,
                    outlet_pressure_pa=outlet_pressure_mpa * 1e6,
                    volumetric_heat_capacity_j_m3k=heat_capacity,
                    max_iterations=int(max_iterations),
                )

            st.session_state.simulation_result = new_result

            st.session_state.simulation_parameters = {
                "length_m": length_m,
                "height_m": height_m,
                "nx": int(nx),
                "ny": int(ny),
                "injection_temp_c": injection_temp_c,
                "initial_temp_c": initial_temp_c,
                "conductivity_w_mk": conductivity_w_mk,
                "permeability_md": permeability_md,
                "viscosity_pa_s": viscosity_pa_s,
                "inlet_pressure_mpa": inlet_pressure_mpa,
                "outlet_pressure_mpa": outlet_pressure_mpa,
                "heat_capacity": heat_capacity,
                "max_iterations": int(max_iterations),
            }

            st.success(
                f"Simulation completed in "
                f"{new_result['wall_time_s']:.4f} seconds."
            )

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

result = st.session_state.simulation_result
params = st.session_state.simulation_parameters

if result is None:
    st.info("Configure the sidebar and run a simulation to continue.")
    st.stop()

p = params

st.markdown("## Simulation Overview")

cols = st.columns(4)

cols[0].metric(
    "Mean Outlet Temperature",
    f"{result['outlet_temperature_c']:.2f} °C",
)
cols[1].metric(
    "Mean Reservoir Temperature",
    f"{result['mean_temperature_c']:.2f} °C",
)
cols[2].metric(
    "Mean Darcy Speed",
    f"{result['mean_speed_m_s']:.3e} m/s",
)
cols[3].metric(
    "Solver Runtime",
    f"{result['wall_time_s']:.4f} s",
)

cols = st.columns(4)

cols[0].metric("Grid Resolution", f"{p['nx']} × {p['ny']}")
cols[1].metric(
    "Pressure Drop",
    f"{p['inlet_pressure_mpa'] - p['outlet_pressure_mpa']:.2f} MPa",
)
cols[2].metric("Iterations", f"{result['iterations']:,}")
cols[3].metric(
    "Final Residual",
    f"{result['final_residual']:.2e}",
)

tabs = st.tabs([
    "Reservoir Fields",
    "Well Layout",
#    "Animation",
    "GPR Surrogate",
    "Sensitivity",
    "Performance",
    "Assumptions & Export",
])

# ============================================================
# TAB 1 — RESERVOIR FIELDS
# ============================================================

with tabs[0]:
    st.subheader("Temperature, Pressure and Velocity")

    selected_field = st.selectbox(
        "Select physical field",
        ["Temperature", "Pressure", "Darcy speed"],
    )

    st.plotly_chart(
        field_figure(result, selected_field),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Temperature Statistics")
        st.write(f"Minimum: {np.min(result['temperature']):.2f} °C")
        st.write(f"Maximum: {np.max(result['temperature']):.2f} °C")
        st.write(
            f"Outlet mean: {result['outlet_temperature_c']:.2f} °C"
        )

    with right:
        st.markdown("#### Darcy Flow Statistics")
        st.write(
            f"Horizontal velocity: {result['darcy_x_m_s']:.4e} m/s"
        )
        st.write(
            f"Maximum speed: {result['max_speed_m_s']:.4e} m/s"
        )
        st.write(
            f"Pressure gradient: {result['pressure_gradient_pa_m']:.4f} Pa/m"
        )

    st.plotly_chart(
        velocity_figure(result),
        width="stretch",
    )

    st.metric(
        "Horizontal Darcy Velocity",
        f"{result['darcy_x_m_s']:.6e} m/s",
    )
# ============================================================
# TAB 2 — CONCEPTUAL WELL LAYOUT
# ============================================================

with tabs[1]:
    st.subheader("Geothermal Reservoir Layout")

    st.plotly_chart(
        well_figure(result),
        use_container_width=True,
    )

    st.caption(
        "Injection and production wells are schematic. "
        "Three-dimensional well flow and local pressure losses "
        "are not resolved."
    )



# ============================================================
# TAB 4 — GPR SURROGATE MODEL
# ============================================================

with tabs[2]:
    st.subheader("Gaussian Process Regression")

    st.write(
        "Predict mean outlet temperature from injection "
        "temperature, Darcy speed, and thermal conductivity."
    )

    st.warning(
        "Training data are generated by the numerical solver. "
        "Metrics indicate agreement with that solver, not "
        "accuracy against real geothermal field measurements."
    )

    with st.expander("Training Configuration", expanded=True):
        sample_count = st.slider(
            "Number of generated samples",
            min_value=30,
            max_value=200,
            value=60,
            step=10,
        )

        test_fraction = st.slider(
            "Independent test fraction",
            min_value=0.15,
            max_value=0.40,
            value=0.25,
            step=0.05,
        )

        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=100000,
            value=42,
        )

    if st.button("Train GPR Model", type="primary"):
        try:
            with st.spinner("Generating data and training GPR..."):
                dataset = generate_dataset(
                    n_samples=int(sample_count),
                    seed=int(random_seed),
                    length_m=p["length_m"],
                    height_m=p["height_m"],
                    nx=min(p["nx"], 32),
                    ny=min(p["ny"], 20),
                    initial_temp_c=p["initial_temp_c"],
                    viscosity_pa_s=p["viscosity_pa_s"],
                    inlet_pressure_pa=p["inlet_pressure_mpa"] * 1e6,
                    outlet_pressure_pa=p["outlet_pressure_mpa"] * 1e6,
                    heat_capacity=p["heat_capacity"],
                )

                evaluation = train_and_evaluate(
                    dataset,
                    test_fraction=float(test_fraction),
                    seed=int(random_seed),
                )

                MODEL_DIR.mkdir(parents=True, exist_ok=True)

                joblib.dump(
                    {
                        "model": evaluation["model"],
                        "features": evaluation["features"],
                        "metrics": evaluation["metrics"],
                        "seed": int(random_seed),
                        "n_samples": len(dataset),
                    },
                    MODEL_PATH,
                )

                st.session_state.gpr_evaluation = evaluation
                st.session_state.gpr_dataset = dataset

            st.success("GPR model trained and saved.")

        except Exception as exc:
            st.error(f"GPR training failed: {exc}")

    evaluation = st.session_state.get("gpr_evaluation")

    if evaluation is not None:
        st.markdown("#### Independent Test Metrics")

        cols = st.columns(3)

        for col, (name, value) in zip(
            cols, evaluation["metrics"].items()
        ):
            col.metric(name, f"{value:.4f}")

        st.plotly_chart(
            surrogate_plot(evaluation),
            use_container_width=True,
        )

        test_df = pd.DataFrame({
            "Actual outlet temperature (°C)": evaluation["actual"],
            "Predicted outlet temperature (°C)": evaluation["predicted"],
            "Predictive standard deviation (°C)": (
                evaluation["uncertainty"]
            ),
        })

        st.dataframe(
            test_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

        cols = st.columns(3)

        cols[0].metric(
            "Training Time",
            f"{evaluation['training_time']:.4f} s",
        )

        cols[1].metric(
            "Batch Inference Time",
            f"{evaluation['prediction_time']:.6f} s",
        )

        cols[2].metric(
            "Independent Test Samples",
            str(evaluation["n_test"]),
        )

        st.download_button(
            "Download Test Predictions",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="thermoflux_gpr_test_results.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Training Dataset",
            data=st.session_state.gpr_dataset.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="thermoflux_training_data.csv",
            mime="text/csv",
        )

    else:
        saved = load_saved_model()

        if saved is not None:
            features = np.array([[
                p["injection_temp_c"],
                abs(result["darcy_x_m_s"]),
                p["conductivity_w_mk"],
            ]])

            try:
                start = time.perf_counter()

                prediction, uncertainty = saved["model"].predict(
                    features,
                    return_std=True,
                )

                inference_time = time.perf_counter() - start

                cols = st.columns(3)

                cols[0].metric(
                    "Predicted Outlet Temperature",
                    f"{prediction[0]:.2f} °C",
                )

                cols[1].metric(
                    "Predictive Standard Deviation",
                    f"{uncertainty[0]:.2f} °C",
                )

                cols[2].metric(
                    "Inference Time",
                    f"{inference_time:.6f} s",
                )

                st.caption(
                    "Prediction from the saved model. Verify that "
                    "the current inputs are within its training domain."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        else:
            st.info(
                "No saved model found. Train GPR to view "
                "evaluation metrics and predictive uncertainty."
            )


# ============================================================
# TAB 5 — SENSITIVITY ANALYSIS
# ============================================================

with tabs[3]:
    st.subheader("Parameter-Effect Analysis")

    st.write(
        "Vary one parameter at a time and rerun the "
        "steady-state numerical solver."
    )

    parameter = st.selectbox(
        "Parameter to vary",
        [
            "Injection temperature",
            "Thermal conductivity",
            "Permeability",
        ],
    )

    if parameter == "Injection temperature":
        values = np.linspace(20.0, 110.0, 8)
        x_label = "Injection Temperature (°C)"
    elif parameter == "Thermal conductivity":
        values = np.linspace(0.8, 5.0, 8)
        x_label = "Thermal Conductivity (W/m·K)"
    else:
        values = np.geomspace(0.1, 100.0, 8)
        x_label = "Permeability (mD)"

    if st.button("Run Sensitivity Analysis"):
        records = []
        progress = st.progress(0)

        for i, value in enumerate(values):
            tin = p["injection_temp_c"]
            conductivity = p["conductivity_w_mk"]
            permeability = p["permeability_md"]

            if parameter == "Injection temperature":
                tin = float(value)
            elif parameter == "Thermal conductivity":
                conductivity = float(value)
            else:
                permeability = float(value)

            run = solve_steady_state(
                length_m=p["length_m"],
                height_m=p["height_m"],
                nx=min(p["nx"], 50),
                ny=min(p["ny"], 30),
                injection_temp_c=tin,
                initial_temp_c=p["initial_temp_c"],
                conductivity_w_mk=conductivity,
                permeability_m2=permeability * 9.869233e-16,
                viscosity_pa_s=p["viscosity_pa_s"],
                inlet_pressure_pa=p["inlet_pressure_mpa"] * 1e6,
                outlet_pressure_pa=p["outlet_pressure_mpa"] * 1e6,
                volumetric_heat_capacity_j_m3k=p["heat_capacity"],
                max_iterations=min(p["max_iterations"], 5000),
            )

            records.append({
                "Parameter value": float(value),
                "Mean outlet temperature (°C)": (
                    run["outlet_temperature_c"]
                ),
                "Mean Darcy speed (m/s)": run["mean_speed_m_s"],
                "Solver time (s)": run["wall_time_s"],
                "Converged": run["converged"],
            })

            progress.progress((i + 1) / len(values))

        st.session_state.sensitivity_df = pd.DataFrame(records)

    sensitivity_df = st.session_state.get("sensitivity_df")

    if sensitivity_df is not None:
        fig = go.Figure(
            go.Scatter(
                x=sensitivity_df["Parameter value"],
                y=sensitivity_df["Mean outlet temperature (°C)"],
                mode="lines+markers",
                name="Mean outlet temperature",
                line=dict(color="#43D6BE", width=3),
            )
        )

        fig.update_layout(
            title="Parameter Effect on Outlet Temperature",
            xaxis_title=x_label,
            yaxis_title="Mean Outlet Temperature (°C)",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sensitivity_df.round(6),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Sensitivity Results",
            data=sensitivity_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="thermoflux_sensitivity.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 6 — PERFORMANCE COMPARISON
# ============================================================

with tabs[4]:
    st.subheader("Computational Performance")

    performance_rows = [{
        "Method": "Steady-state FDM",
        "Time (s)": result["wall_time_s"],
        "Details": (
            f"{p['nx']} × {p['ny']} grid; "
            f"{result['iterations']} iterations"
        ),
    }]

    evaluation = st.session_state.get("gpr_evaluation")

    if evaluation is not None:
        performance_rows.extend([
            {
                "Method": "GPR Training",
                "Time (s)": evaluation["training_time"],
                "Details": (
                    f"{evaluation['n_train']} training samples"
                ),
            },
            {
                "Method": "GPR Batch Inference",
                "Time (s)": evaluation["prediction_time"],
                "Details": (
                    f"{evaluation['n_test']} test samples"
                ),
            },
        ])

    performance_df = pd.DataFrame(performance_rows)

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True,
    )

    colors = ["#F59E0B", "#43D6BE", "#60A5FA"]

    fig = go.Figure(
        go.Bar(
            x=performance_df["Method"],
            y=performance_df["Time (s)"],
            text=[
                f"{value:.5g} s"
                for value in performance_df["Time (s)"]
            ],
            textposition="outside",
            marker_color=colors[:len(performance_df)],
        )
    )

    fig.update_layout(
        title="Execution Time Comparison",
        xaxis_title="Method",
        yaxis_title="Wall Time (s)",
    )

    st.plotly_chart(fig, use_container_width=True)

    if evaluation is not None and evaluation["prediction_time"] > 0:
        ratio = (
            result["wall_time_s"]
            / evaluation["prediction_time"]
        )

        st.metric(
            "Apparent Batch Speed Ratio",
            f"{ratio:.2f}×",
        )

        st.caption(
            "This compares one numerical solve with a complete "
            "GPR test batch, not equivalent per-sample workloads. "
            "Training costs and predictive accuracy also matter."
        )

    st.download_button(
        "Download Performance Report",
        data=performance_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="thermoflux_performance.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 7 — ASSUMPTIONS AND EXPORTS
# ============================================================

with tabs[5]:
    st.subheader("Model Assumptions and Limitations")

    st.markdown("""
    ### Thermal model

    - Two-dimensional steady-state finite-difference solver.
    - Constant thermal conductivity and volumetric heat capacity.
    - Fixed injection temperature at the left boundary.
    - Fixed rock temperature at the top and bottom boundaries.
    - Zero-normal-gradient temperature at the production boundary.
    - Upwind approximation for advection.
    - No transient thermal-breakthrough calculation.

    ### Flow model

    - Linear pressure distribution between inlet and outlet.
    - Darcy velocity derived from the prescribed pressure gradient.
    - Constant permeability and viscosity.
    - No full pressure PDE, CFD solver, or FEM implementation.
    - No resolved three-dimensional well flow.

    ### Surrogate model

    - Inputs: injection temperature, Darcy speed, and conductivity.
    - Target: mean outlet temperature.
    - Metrics use an independent held-out test set.
    - Training samples are generated by the numerical solver.
    - GPR uncertainty does not represent total physical uncertainty.

    ### Validation

    - Numerical convergence does not establish physical validity.
    - Boundary conditions influence the temperature field.
    - Mesh-convergence and analytical benchmark tests are necessary.
    - The application has not been validated against real geothermal
      field data and is not intended for engineering design.
    """)

    st.markdown("### Numerical Diagnostics")

    cols = st.columns(3)

    cols[0].metric(
        "Converged",
        "Yes" if result["converged"] else "No",
    )

    cols[1].metric(
        "Final Residual",
        f"{result['final_residual']:.3e}",
    )

    cols[2].metric(
        "Grid Spacing",
        f"{result['dx_m']:.2f} × {result['dy_m']:.2f} m",
    )

    st.markdown("### Simulation Parameters")

    config_df = pd.DataFrame([
        {"Parameter": key, "Value": value}
        for key, value in p.items()
    ])

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Export Numerical Fields")

    temperature_df = pd.DataFrame(
        result["temperature"],
        index=result["y"],
        columns=result["x"],
    )
    temperature_df.index.name = "y_m"
    temperature_df.columns.name = "x_m"

    pressure_df = pd.DataFrame(
        result["pressure"],
        index=result["y"],
        columns=result["x"],
    )
    pressure_df.index.name = "y_m"
    pressure_df.columns.name = "x_m"

    speed_df = pd.DataFrame(
        result["speed"],
        index=result["y"],
        columns=result["x"],
    )
    speed_df.index.name = "y_m"
    speed_df.columns.name = "x_m"

    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "Temperature Field CSV",
        data=temperature_df.to_csv().encode("utf-8"),
        file_name="thermoflux_temperature_c.csv",
        mime="text/csv",
        use_container_width=True,
    )

    col2.download_button(
        "Pressure Field CSV",
        data=pressure_df.to_csv().encode("utf-8"),
        file_name="thermoflux_pressure_pa.csv",
        mime="text/csv",
        use_container_width=True,
    )

    col3.download_button(
        "Darcy Speed Field CSV",
        data=speed_df.to_csv().encode("utf-8"),
        file_name="thermoflux_darcy_speed_m_s.csv",
        mime="text/csv",
        use_container_width=True,
    )

    summary = pd.DataFrame([{
        **p,
        "outlet_temperature_c": result["outlet_temperature_c"],
        "mean_temperature_c": result["mean_temperature_c"],
        "mean_darcy_speed_m_s": result["mean_speed_m_s"],
        "solver_time_s": result["wall_time_s"],
        "iterations": result["iterations"],
        "converged": result["converged"],
        "final_residual": result["final_residual"],
    }])

    st.download_button(
        "Download Run Summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="thermoflux_run_summary.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ThermoFlux | Geothermal Reservoir Simulation & Surrogate Modeling "
    "| Educational Research Prototype | Copyright © 2026 Naeim Mousavi"
)

