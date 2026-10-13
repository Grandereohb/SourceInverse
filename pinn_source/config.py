import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# Data Paths
# =========================
SITE_PATH = 'C:/Document/phd/SourceInverse/SourceInverse/data/jjj/sites.xlsx'
CONC_PATH = 'C:/Document/phd/SourceInverse/SourceInverse/data/jjj/concentration.xlsx'
WIND_PATH = 'C:/Document/phd/SourceInverse/SourceInverse/data/jjj/wind.xlsx'
# =========================
# Model / Device
# =========================
DEVICE = "auto"  # "auto", "cpu", or "cuda"
FIELD_MODE = "recurrent_pde"

# Single algorithm selector; both modes use the shared models.pinn.PINN backend.
# Set to "ia_vpdi" to switch both direct and batch runs to variable projection.
# PINN_INVERSION_MODE overrides this value; joint_adam remains the default.
INVERSION_MODE = "joint_adam"  # "joint_adam" or "ia_vpdi"

# =========================
# IA-VPDI (Experimental)
# =========================
IA_VP_Q_BASIS_COUNT = 0  # 0 selects 3-5 basis functions from event length
IA_VP_Q_BASIS_TYPE = "hat"
IA_VP_RESPONSE_BATCH_SIZE = 8  # 1 retains the serial reference; no Q/search change
IA_VP_JACOBIAN_BATCH_SIZE = 16  # diagnostic VJP rows; 1 retains serial gradients
IA_VP_TIMING_ENABLED = True
IA_VP_Q_LAYOUT_STRATEGY = "uniform"  # opt-in: local_templates (clock-only hat layouts)
IA_VP_Q_UPPER_BOUND = float("inf")  # opt-in model-unit amplitude bound
IA_VP_Q_KNOTS = None  # optional normalized times; fixed during refinement
IA_VP_SEARCH_STEP_M = 500.0
IA_VP_TOP_K = 3
IA_VP_START_STRATEGY = "score"  # opt-in "spatial_diverse"; baseline unchanged
IA_VP_START_MIN_DISTANCE_M = 1000.0  # relaxed if too few distinct neighborhoods
IA_VP_DIFFUSION_SCALE_SEEDS = (0.5, 1.0, 2.0)
IA_VP_DIFFUSION_RATIO_SEEDS = (1.0, 3.0)
IA_VP_OUTER_STEPS = 200
IA_VP_OUTER_LR = 1.0e-2
IA_VP_OUTER_PATIENCE = 30
IA_VP_OUTER_MIN_DELTA = 1.0e-8
IA_VP_MAX_GRAD_NORM = 10.0
IA_VP_ACTIVE_Q_TOL = 1.0e-8
IA_VP_INFORMATION_RCOND = 1.0e-10
IA_VP_INFORMATION_CONDITION_MAX = 1.0e8
IA_VP_LOCALIZATION_RADIUS_LIMIT_M = (
    None  # diagnostic only; None uses search-domain diagonal
)
IA_VP_BOUNDARY_MARGIN_M = 250.0
IA_VP_REPORT_SCORE_TOLERANCES = (
    0.01,
    0.05,
    0.10,
)  # sampled MSE sets, not confidence levels
IA_VP_SENSOR_RECOMMENDATION_ENABLED = False
IA_VP_SENSOR_GRID_STEP_M = 500.0
IA_VP_SENSOR_TOP_K = 5
IA_VP_SENSOR_MIN_DISTANCE_M = 250.0

# Optional wind-aligned diffusion tensor. Disabled by default so the current
# isotropic solver remains unchanged.
ENABLE_ANISOTROPIC_DIFFUSION = False
D_PARALLEL_MIN = 1.0  # m^2/s
D_PERP_MIN = 1.0  # m^2/s

# =========================
# Output
# =========================
OUTPUT_DIR = os.environ.get(
    "PINN_OUTPUT_DIR",
    str(PROJECT_ROOT / "result"),
)
TARGET_POLLUTANT = '苯'
MAKE_PLOTS = True

# =========================
# Wind
# =========================
WIND_DIR_IS_FROM = True
ENABLE_WIND_VECTOR_SMOOTHING = True
WIND_SMOOTH_WINDOW = 5
WIND_SMOOTH_LOW_SPEED_MPS = 0.8
# Fraction of the measured hourly mean wind used by the effective transport model.
WIND_SCALE = 0.25

# =========================
# Training
# =========================
EPOCHS = 2500
LR = 1e-3
LOSS_W_DATA = 1.0
MAX_GRAD_NORM = 10.0
DEBUG_EVERY = 1000

EARLY_STOP_START = 700
EARLY_STOP_PATIENCE = 200
EARLY_STOP_MIN_DELTA = 1.0

# =========================
# Domain / Source
# =========================
DOMAIN_PAD_M = 500.0
# Positive values allow source candidates outside the station envelope.
SOURCE_POSITION_PAD_M = 500.0
SOURCE_INIT_MODE = "max_station_upwind"  # "center" or "max_station_upwind"
SOURCE_INIT_UPWIND_DISTANCE_M = 1000.0
SOURCE_INIT_PEAK_VALUE = (
    "raw"  # "raw" uses observed high-value peak, "fit" uses residual peak
)
SIGMA_SRC = 0.05
SIGMA_SRC_M = None  # optional Gaussian standard deviation in meters
RECURRENT_SOURCE_MODE = "gaussian"  # legacy; opt-in "cell_gaussian" or "point"
RECURRENT_DIFFUSION_SOLVER = (
    "gaussian"  # opt-in "tensor_semigroup"; "implicit" is scalar only
)
RECURRENT_TRANSPORT_SOLVER = "grid"  # opt-in full-space "gaussian_parcels"
RECURRENT_PARCEL_SUBDIVISIONS = 64  # release quadrature, independent of spatial grid
D_MIN_PHYS = 1.0  # m^2/s

# =========================
# Recurrent PDE Plume
# =========================
RECURRENT_GRID_NX = 36
RECURRENT_GRID_NY = 36
RECURRENT_SUBSTEPS = 1  # minimum when adaptive stepping is enabled
RECURRENT_ADAPTIVE_SUBSTEPS = True
RECURRENT_MAX_ADVECTION_CELLS = 6.0
RECURRENT_MAX_SUBSTEPS = 3
RECURRENT_SOURCE_SCALE = 1.0
RECURRENT_DECAY = 0.50  # 1/hour
RECURRENT_INITIAL_RELEASE_FRACTION = 1.0
RECURRENT_SOURCE_DECAY_INTEGRATOR = "legacy"  # opt-in: "exponential_linear"

# =========================
# Source Strength Q(t)
# =========================
Q_MODE = "smooth_time"  # "neural", "smooth_time", or "piecewise"
Q_SEGMENT_LENGTH = 6
Q_SMOOTH_WEIGHT = 0.0
Q_L2_WEIGHT = 0.0001
Q_MIN = 0.05
Q_MAX = 100.0

# These remain for PINN model compatibility. They are frozen in recurrent_pde mode.
PLUME_MAX = None
PLUME_RAW_SHIFT = 4.0

# =========================
# Data Fitting / Event Window
# =========================
DATA_NORMALIZE = True
TRAIN_ON_RESIDUAL = True
BASELINE_MODE = "median"  # "median", "q25", or "q40"

ENABLE_EVENT_WINDOW_CROP = True
EVENT_WINDOW_MIN_MAX = 1.0
EVENT_WINDOW_MIN_RELIEF = 0.15
EVENT_WINDOW_PAD_STEPS = 2

DATA_SCALE_PERCENTILE = 95.0
DATA_HIGH_WEIGHT = 2.0
DATA_HIGH_PERCENTILE = 95.0
DATA_HIGH_POWER = 1.0

DATA_TIME_PEAK_WEIGHT = 4.0
DATA_TIME_PEAK_RATIO = 0.6
DATA_TIME_PEAK_POWER = 1.0
DATA_TIME_PEAK_MIN_RELIEF = 0.15

EVENT_TIME_WEIGHT = 3.0
EVENT_PEAK_WEIGHT = 3.0
EVENT_PEAK_RATIO = 0.6

# Residual Focal Loss is kept for optional rollback, but raw worst-residual loss
# is now the primary way to prevent severe misses from being averaged away.
RESIDUAL_FOCAL_WEIGHT = 0.0
RESIDUAL_FOCAL_POWER = 1.0
RESIDUAL_FOCAL_SCALE = 1.0
RESIDUAL_FOCAL_MAX_WEIGHT = 20.0

RAW_RESIDUAL_BASE_WEIGHT = 1.0
RAW_RESIDUAL_WORST_WEIGHT = 0.5

# Debug-only multi-station peak counters.
MULTI_HIGH_RATIO = 0.5
MULTI_HIGH_MIN_RELIEF = 0.15

# =========================
# Diagnostics / Ablation
# =========================
SINGLE_STATION_DOMINANCE_RATIO = 0.8
SINGLE_STATION_PEAK_MISS_RATIO = 0.2
SINGLE_STATION_PEAK_TIME_TOL_H = 6.0

ENABLE_STATION_ABLATION = False
ABLATION_TARGET_STATION = "上石化边界卫三路站"
ABLATION_NEIGHBOR_RADIUS_M = 2500.0
ABLATION_MAX_NEIGHBORS = 4

# =========================
# Source Confidence / Plots
# =========================
USE_SOURCE_LANDSCAPE_CONFIDENCE = True
SOURCE_LANDSCAPE_MODE = "local"  # "local" or "source_domain"
SOURCE_LANDSCAPE_RADIUS_M = 450.0
SOURCE_LANDSCAPE_STEP_M = 150.0
SOURCE_LANDSCAPE_TEMPERATURE = 0.1
SOURCE_LANDSCAPE_LEVELS = [0.5, 0.8, 0.95]

DIFFUSION_N_FRAMES = 24
DIFFUSION_NX = 140
DIFFUSION_NY = 140
DIFFUSION_FPS = 2
ADD_BASELINE_TO_VIZ = True
