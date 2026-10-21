
# Reservoir geometry
NX = 50
NY = 30
LENGTH = 100.0  # m
HEIGHT = 50.0   # m

# Thermal properties
INITIAL_TEMP = 150.0       # °C
INJECTION_TEMP = 40.0      # °C
THERMAL_CONDUCTIVITY = 2.5 # W/(m·K)
RHO_CP = 2.5e6             # J/(m³·K)

# Darcy flow properties
PERMEABILITY = 1e-13  # m²
VISCOSITY = 1e-3      # Pa·s
INLET_PRESSURE = 2e6  # Pa
OUTLET_PRESSURE = 1e6 # Pa

# Dataset generation
N_SAMPLES = 100
RANDOM_SEED = 42
TEST_SIZE = 0.2
