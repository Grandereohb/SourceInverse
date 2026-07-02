# =========================
# Data Paths
# =========================
SITE_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\sites.xlsx"
CONC_PATH = (
    r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\concentration.xlsx"
)
WIND_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\wind.xlsx"

# =========================
# Model / Device
# =========================
MODEL_NAME = "pinn"
DEVICE = "auto"  # "auto", "cpu", or "cuda"
FIELD_MODE = "recurrent_pde"

# =========================
# Output
# =========================
OUTPUT_DIR = r"C:\Document\phd\SourceInverse\SourceInverse\result"
TARGET_POLLUTANT = ""
MAKE_PLOTS = True

# =========================
# Wind
# =========================
WIND_DIR_IS_FROM = True
ENABLE_WIND_VECTOR_SMOOTHING = True
WIND_SMOOTH_WINDOW = 5
WIND_SMOOTH_LOW_SPEED_MPS = 0.8
WIND_SCALE = 10.0

# =========================
# Training
# =========================
EPOCHS = 5000
LR = 1e-3
LOSS_W_DATA = 1.0
MAX_GRAD_NORM = 10.0
DEBUG_EVERY = 500

EARLY_STOP_START = 1800
EARLY_STOP_PATIENCE = 500
EARLY_STOP_MIN_DELTA = 1e-4

# =========================
# Domain / Source
# =========================
DOMAIN_PAD_M = 500.0
SOURCE_POSITION_PAD_M = -300.0
SIGMA_SRC = 0.05
D_MIN_PHYS = 500.0

# =========================
# Recurrent PDE Plume
# =========================
RECURRENT_GRID_NX = 56
RECURRENT_GRID_NY = 56
RECURRENT_SUBSTEPS = 2
RECURRENT_SOURCE_SCALE = 1.0
RECURRENT_DECAY = 0.15
RECURRENT_INITIAL_RELEASE_FRACTION = 1.0

# =========================
# Source Strength Q(t)
# =========================
Q_MODE = "smooth_time"  # "neural", "smooth_time", or "piecewise"
Q_SEGMENT_LENGTH = 6
Q_SMOOTH_WEIGHT = 0.03
Q_L2_WEIGHT = 0.001
Q_MIN = 0.2
Q_MAX = 5.0

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

# Debug-only multi-station peak counters.
MULTI_HIGH_RATIO = 0.5
MULTI_HIGH_MIN_RELIEF = 0.15

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
DIFFUSION_NX = 80
DIFFUSION_NY = 80
ADD_BASELINE_TO_VIZ = True
