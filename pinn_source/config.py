# =========================
# Data Paths
# =========================
SITE_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh\sites.xlsx"
CONC_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh\concentration.xlsx"
WIND_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh\wind.xlsx"

# =========================
# Model Selection
# =========================
# MODEL_NAME: model key used by model registry.
MODEL_NAME = "pinn"

# =========================
# Wind Direction Convention
# =========================
# WIND_DIR_IS_FROM:
# True  -> meteorological "from" convention (0=N means wind comes from north)
# False -> "to" convention (0=N means wind blows toward north)
WIND_DIR_IS_FROM = True

# =========================
# Training Basics
# =========================
# EPOCHS: number of optimization iterations.
EPOCHS = 5000

# LR: learning rate for model optimizer (Adam).
LR = 1e-3

# N_COLLOCATION: number of PDE collocation points per cycle.
N_COLLOCATION = 5000

# DOMAIN_PAD_M: extra padding (meters) added around station bounding box.
DOMAIN_PAD_M = 500.0

# =========================
# Core Loss Weights (Base)
# =========================
# Base multipliers before optional adaptive weighting.
LOSS_W_DATA = 1.0
LOSS_W_PDE = 1.0
LOSS_W_PENALTY = 1.0

# =========================
# Source / Physics
# =========================
# SIGMA_SRC: source Gaussian width in normalized coordinates.
SIGMA_SRC = 0.05

# D_MIN_PHYS: lower bound of physical diffusion coefficient before normalization.
D_MIN_PHYS = 0.01

# D_PERP_RATIO: fixed ratio D_perp / D_parallel for anisotropic diffusion.
D_PERP_RATIO = 0.2

# WIND_SCALE: multiplier for normalized wind velocity to tune advection strength.
WIND_SCALE = 10.0

# =========================
# Residual Weighting / Collocation Sampling
# =========================
# RESIDUAL_R: source-near radius for PDE residual weighting (normalized coords).
RESIDUAL_R = 0.05

# RESIDUAL_W_SCALE: extra residual weight near source; 0 disables source-local boost.
RESIDUAL_W_SCALE = 0.2

# COLLOC_SOURCE_RATIO: fraction of collocation points sampled near estimated source.
COLLOC_SOURCE_RATIO = 0.2

# COLLOC_SOURCE_R: spread (normalized) for source-focused collocation sampling.
COLLOC_SOURCE_R = 0.1


# =========================
# Source-ID Extra Loss Weights
# =========================
# LOSS_W_RADIAL: weight for outward radial monotonicity constraint.
LOSS_W_RADIAL = 0.0

# LOSS_W_WIND: weight for upwind concentration suppression.
LOSS_W_WIND = 0.0

# LOSS_W_BOUNDARY: weight for source boundary repulsion penalty.
LOSS_W_BOUNDARY = 0.0

# LOSS_W_AXIS: weight for plume-axis wind-alignment constraint.
LOSS_W_AXIS = 5.0

# LOSS_W_CROSSWIND: weight for suppressing excessive crosswind spreading.
LOSS_W_CROSSWIND = 1.0

# LOSS_W_PLUME: weight for enforcing monotonic decay along downwind plume direction.
LOSS_W_PLUME = 0.0

# LOSS_W_SOURCE_LOCAL: weight for keeping source-neighborhood concentration above far field.
LOSS_W_SOURCE_LOCAL = 1.0

# LOSS_W_TIME_SMOOTH: weight for suppressing unrealistic temporal jumps at fixed locations.
LOSS_W_TIME_SMOOTH = 0.5

# AXIS_UPDATE_INTERVAL: compute axis loss once every N epochs and reuse cached value in between.
AXIS_UPDATE_INTERVAL = 5

# =========================
# Adaptive Loss Weighting
# =========================
# USE_ADAPTIVE_LOSS: whether to learn data/pde/penalty balancing weights.
USE_ADAPTIVE_LOSS = False

# ADAPTIVE_LOSS_LR: optimizer learning rate for adaptive loss weights.
ADAPTIVE_LOSS_LR = 1e-2

# ADAPTIVE_INIT_LOG_VARS: initial log-variance values [data, pde, penalty].
ADAPTIVE_INIT_LOG_VARS = [0.0, 0.0, 0.0]

# ADAPTIVE_WARMUP_EPOCHS: fixed-weight warmup epochs before adaptive updates.
ADAPTIVE_WARMUP_EPOCHS = 1000

# ADAPTIVE_MIN_PRECISIONS: lower bound of adaptive precisions [data, pde, penalty].
ADAPTIVE_MIN_PRECISIONS = [0.3, 1.0, 0.0]

# ADAPTIVE_MAX_PRECISIONS: upper bound of adaptive precisions [data, pde, penalty].
ADAPTIVE_MAX_PRECISIONS = [10.0, 10.0, 10.0]


# =========================
# Data Fitting Stabilization
# =========================
# DATA_NORMALIZE: enable robust scaling for concentration target to improve optimization stability.
DATA_NORMALIZE = True

# DATA_SCALE_PERCENTILE: robust scale based on percentile(|c_obs|), used when DATA_NORMALIZE=True.
DATA_SCALE_PERCENTILE = 85.0

# DATA_WARMUP_EPOCHS: train with data-dominant objective in early epochs.
DATA_WARMUP_EPOCHS = 1500

# DATA_WARMUP_PDE_FACTOR: multiplier for PDE term during warmup (0 means data-only warmup).
DATA_WARMUP_PDE_FACTOR = 0.0

# PDE_RAMP_EPOCHS: epochs to smoothly increase PDE contribution from warmup factor to full weight.
PDE_RAMP_EPOCHS = 3000

# MAX_GRAD_NORM: gradient clipping threshold for training stability (None or <=0 disables).
MAX_GRAD_NORM = 10.0

# =========================
# Legacy (currently not used in pipeline)
# =========================
# Kept for compatibility; current pipeline computes dynamic L/T from data span.
SCALE_XY = 1000.0
SCALE_T = 1.0
