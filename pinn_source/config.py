# =========================
# Data Paths
# =========================
SITE_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\sites.xlsx"
CONC_PATH = (
    r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\concentration.xlsx"
)
WIND_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\shsh_js\wind.xlsx"

# =========================
# Model Selection
# =========================
# MODEL_NAME: model key used by model registry.
MODEL_NAME = "pinn"

# DEVICE:
# "auto" -> use CUDA when available, otherwise CPU
# "cpu"  -> force CPU and avoid CUDA probing
# "cuda" -> force CUDA
DEVICE = "auto"

# FIELD_MODE:
# "default"       -> bg(t) + gate * (source_bias + plume_strength) * Q(t)
# "no_gate"       -> bg(t) + plume_strength * Q(t)
# "no_background" -> gate * (source_bias + plume_strength) * Q(t)
# "minimal"       -> plume_strength * Q(t)
# "analytic_plume" -> physically constrained downwind plume kernel * Q(t)
FIELD_MODE = "analytic_plume"

# PDE_SOURCE_MODE:
# "gaussian" -> keep Gaussian source term in PDE residual
# "none"     -> disable PDE source term for ablation
PDE_SOURCE_MODE = "none"

# =========================
# Output
# =========================
# OUTPUT_DIR: parent directory for timestamped run result folders.
OUTPUT_DIR = r"C:\Document\phd\SourceInverse\SourceInverse\result"

# TARGET_POLLUTANT: appended to timestamped result folder names.
TARGET_POLLUTANT = "间-二甲苯+对-二甲苯"

# MAKE_PLOTS: show/save the original diagnostic plots for a single run.
MAKE_PLOTS = True

# =========================
# Wind Direction Convention
# =========================
# WIND_DIR_IS_FROM:
# True  -> meteorological "from" convention (0=N means wind comes from north)
# False -> "to" convention (0=N means wind blows toward north)
WIND_DIR_IS_FROM = True

# Smooth wind vectors before training/visualization. This reduces frame-to-frame
# plume jitter when low-speed wind directions jump sharply.
ENABLE_WIND_VECTOR_SMOOTHING = True
WIND_SMOOTH_WINDOW = 5
WIND_SMOOTH_LOW_SPEED_MPS = 0.8

# =========================
# Training Basics
# =========================
# EPOCHS: number of optimization iterations.
# Daily runs use a balanced default; increase to 5000 for final high-precision runs.
EPOCHS = 3500

# LR: learning rate for model optimizer (Adam).
LR = 1e-3

# N_COLLOCATION: number of PDE collocation points per cycle.
# PDE autograd dominates runtime, so this is the main speed/precision knob.
N_COLLOCATION = 1500

# DOMAIN_PAD_M: extra padding (meters) added around station bounding box.
DOMAIN_PAD_M = 500.0

# SOURCE_POSITION_PAD_M: source candidates are constrained to the station
# bounding box plus this padding. This is a projected parameter bound, not an
# added loss term.
SOURCE_POSITION_PAD_M = -300.0

# =========================
# Core Loss Weights (Base)
# =========================
# Base multipliers before optional adaptive weighting.
LOSS_W_DATA = 1.0
LOSS_W_PDE = 1.0

# =========================
# Source / Physics
# =========================
# SIGMA_SRC: source Gaussian width in normalized coordinates.
SIGMA_SRC = 0.05

# Gate-shape controls used by source_gate().
GATE_CORE_SCALE = 1.0
GATE_CROSS_SCALE = 2.5
GATE_CROSS_MIN = 0.05
GATE_STEEPNESS_SCALE = 1.5
GATE_STEEPNESS_MIN = 0.04
GATE_DECAY_SCALE = 6.0
GATE_DECAY_MIN = 0.18
GATE_FLOOR = 0.0
GATE_DOWNWIND_BROADEN = 1.0

# Analytic plume transport memory used when FIELD_MODE="analytic_plume".
# Ages are in normalized event-window time units; 0.25 means roughly one quarter
# of the cropped training window.
ANALYTIC_PLUME_LAG_STEPS = 3
ANALYTIC_PLUME_MAX_AGE = 0.45
ANALYTIC_PLUME_MIN_AGE = 0.05
ANALYTIC_PLUME_AGE_DECAY = 0.18
ANALYTIC_PLUME_ALONG_SPREAD = 0.04
ANALYTIC_PLUME_CROSS_SPREAD = 0.15
ANALYTIC_PLUME_TRANSPORT_SCALE = 12.0
ANALYTIC_PLUME_SOURCE_CORE_WEIGHT = 0.0

# D_MIN_PHYS: lower bound of physical diffusion coefficient before normalization.
D_MIN_PHYS = 500.0

# D_PERP_RATIO: fixed ratio D_perp / D_parallel for anisotropic diffusion.
D_PERP_RATIO = 0.2

# WIND_SCALE: multiplier for normalized wind velocity to tune advection strength.
WIND_SCALE = 10.0

# =========================
# Source Strength Q(t)
# =========================
# Q_MODE:
# "neural"      -> continuous Q(t) = exp(logQ + q_net(t))
# "smooth_time" -> continuous piecewise-linear logQ(t) with one control point per timestamp
# "piecewise"   -> event-window time steps are grouped and each segment has one logQ_k
Q_MODE = "smooth_time"

# Q_SEGMENT_LENGTH: number of unique event-window timestamps in one Q segment.
Q_SEGMENT_LENGTH = 6

# Smoothness and amplitude regularization for time-varying logQ.
# For neural Q(t), these act on q_net over the observed time grid.
Q_SMOOTH_WEIGHT = 0.03
Q_L2_WEIGHT = 0.001

# Optional clamp bounds for Q(t). Set to None to disable a bound.
Q_MIN = 0.2
Q_MAX = 5.0

# PLUME_MAX: upper bound for the learned plume shape factor before multiplying
# by gate and Q(t). This prevents the model from hiding unphysical plume spikes
# behind very small Q values.
PLUME_MAX = None

# PLUME_RAW_SHIFT: larger values make the bounded plume start closer to zero
# while preserving the upper bound PLUME_MAX.
PLUME_RAW_SHIFT = 4.0

# =========================
# Residual Weighting / Collocation Sampling
# =========================
# RESIDUAL_R: source-near radius for PDE residual weighting (normalized coords).
RESIDUAL_R = 0.05

# RESIDUAL_W_SCALE: extra residual weight near source; 0 disables source-local boost.
RESIDUAL_W_SCALE = 0.2

# COLLOC_SOURCE_RATIO: fraction of collocation points sampled near estimated source.
COLLOC_SOURCE_RATIO = 0.3

# COLLOC_PLUME_RATIO: fraction of collocation points sampled along the downwind plume axis.
COLLOC_PLUME_RATIO = 0.4

# COLLOC_SOURCE_R: spread (normalized) for source-focused collocation sampling.
COLLOC_SOURCE_R = 0.1

# COLLOC_PLUME_LENGTH: downwind sampling extent in normalized coordinates.
COLLOC_PLUME_LENGTH = 1.0


# =========================
# Source-ID Extra Loss Weights
# =========================
# LOSS_W_AXIS: weight for plume-axis wind-alignment constraint.
LOSS_W_AXIS = 1.0
ENABLE_LOSS_AXIS = True
AXIS_MIN_RELIEF = 0.15
AXIS_HIGH_RATIO = 0.6
AXIS_ALONG_MARGIN = 0.03
AXIS_CROSS_BASE = 0.05
AXIS_CROSS_SLOPE = 0.35

# LOSS_W_SOURCE_LOCAL: weight for keeping source-neighborhood concentration above far field.
LOSS_W_SOURCE_LOCAL = 1.0
ENABLE_LOSS_SOURCE_LOCAL = True

# SOURCE_LOCAL_MARGIN: required concentration margin between source neighborhood and far field.
SOURCE_LOCAL_MARGIN = 0.2

# SOURCE_LOCAL_RING_R: normalized radius of the annulus used to compare source-center vs nearby field.
SOURCE_LOCAL_RING_R = 0.12

# AXIS_UPDATE_INTERVAL: compute axis loss once every N epochs and reuse cached value in between.
AXIS_UPDATE_INTERVAL = 5

# AUX_LOSS_UPDATE_INTERVAL: compute top_station / multi_high / high_downwind / source_local
# once every N epochs and reuse cached values in between.
AUX_LOSS_UPDATE_INTERVAL = 3

# =========================
# Adaptive Loss Weighting
# =========================
# USE_ADAPTIVE_LOSS: whether to learn data/pde balancing weights.
USE_ADAPTIVE_LOSS = False

# ADAPTIVE_LOSS_LR: optimizer learning rate for adaptive loss weights.
ADAPTIVE_LOSS_LR = 1e-2

# ADAPTIVE_INIT_LOG_VARS: initial log-variance values [data, pde].
ADAPTIVE_INIT_LOG_VARS = [0.0, 0.0]

# ADAPTIVE_WARMUP_EPOCHS: fixed-weight warmup epochs before adaptive updates.
ADAPTIVE_WARMUP_EPOCHS = 1000

# ADAPTIVE_MIN_PRECISIONS: lower bound of adaptive precisions [data, pde].
ADAPTIVE_MIN_PRECISIONS = [0.3, 1.0]

# ADAPTIVE_MAX_PRECISIONS: upper bound of adaptive precisions [data, pde].
ADAPTIVE_MAX_PRECISIONS = [10.0, 10.0]


# =========================
# Data Fitting Stabilization
# =========================
# DATA_NORMALIZE: enable robust scaling for concentration target to improve optimization stability.
DATA_NORMALIZE = True

# TRAIN_ON_RESIDUAL: fit plume anomaly after subtracting a robust per-timestamp background baseline.
TRAIN_ON_RESIDUAL = True

# BASELINE_MODE: robust baseline estimator used when TRAIN_ON_RESIDUAL=True.
# Supported: "median", "q25", "q40"
BASELINE_MODE = "median"

# ENABLE_EVENT_WINDOW_CROP: keep only the main anomaly window (with small padding) for training.
ENABLE_EVENT_WINDOW_CROP = True

# EVENT_WINDOW_MIN_MAX: minimum residual max at a timestamp to mark it as part of the anomaly event.
EVENT_WINDOW_MIN_MAX = 1.0

# EVENT_WINDOW_MIN_RELIEF: minimum relative contrast needed to regard a timestamp as anomalous.
EVENT_WINDOW_MIN_RELIEF = 0.15

# EVENT_WINDOW_PAD_STEPS: number of timestamps kept before the first and after the last anomalous timestamp.
EVENT_WINDOW_PAD_STEPS = 2

# DATA_SCALE_PERCENTILE: robust scale based on percentile(|c_obs|), used when DATA_NORMALIZE=True.
DATA_SCALE_PERCENTILE = 95.0

# DATA_HIGH_WEIGHT: extra weight multiplier for anomalously high observation residuals.
DATA_HIGH_WEIGHT = 2.0

# DATA_HIGH_PERCENTILE: observations above this residual percentile receive extra fitting weight.
DATA_HIGH_PERCENTILE = 95.0

# DATA_HIGH_POWER: nonlinearity of anomaly weighting; >1 emphasizes extreme peaks more strongly.
DATA_HIGH_POWER = 1.0

# DATA_TIME_PEAK_WEIGHT: extra per-timestamp weight for stations that are locally high at a given time.
DATA_TIME_PEAK_WEIGHT = 4.0

# DATA_TIME_PEAK_RATIO: stations above this fraction of the timestamp max residual receive time-local boost.
DATA_TIME_PEAK_RATIO = 0.6

# DATA_TIME_PEAK_POWER: nonlinearity for the within-timestamp anomaly boost.
DATA_TIME_PEAK_POWER = 1.0

# DATA_TIME_PEAK_MIN_RELIEF: skip per-timestamp boosting when the timestamp has weak anomaly contrast.
DATA_TIME_PEAK_MIN_RELIEF = 0.15

# EVENT_TIME_WEIGHT: extra weight for timestamps identified as anomaly periods.
EVENT_TIME_WEIGHT = 3.0

# EVENT_PEAK_WEIGHT: extra weight for samples that are locally high inside anomaly timestamps.
EVENT_PEAK_WEIGHT = 3.0

# EVENT_PEAK_RATIO: inside anomaly timestamps, stations above this fraction of the timestamp max get extra boost.
EVENT_PEAK_RATIO = 0.6

# DATA_WARMUP_EPOCHS: train with data-dominant objective in early epochs.
DATA_WARMUP_EPOCHS = 300

# DATA_WARMUP_PDE_FACTOR: multiplier for PDE term during warmup (0 means data-only warmup).
DATA_WARMUP_PDE_FACTOR = 0.2

# PDE_RAMP_EPOCHS: epochs to smoothly increase PDE contribution from warmup factor to full weight.
PDE_RAMP_EPOCHS = 1000

# STAGE1_EPOCHS: first training stage focuses on fitting high-value observations before full physics is restored.
STAGE1_EPOCHS = 0

# STAGE1_PDE_FACTOR: PDE multiplier used during stage 1.
STAGE1_PDE_FACTOR = 1.0

# STAGE1_DATA_MULT: additional multiplier on the data term during stage 1.
STAGE1_DATA_MULT = 1.0

# STAGE1_TOP_STATION_MULT: additional multiplier on top-station ranking loss during stage 1.
STAGE1_TOP_STATION_MULT = 1.0

# STAGE1_MULTI_HIGH_MULT: additional multiplier on multi-high-station fitting loss during stage 1.
STAGE1_MULTI_HIGH_MULT = 1.0

# STAGE1_HIGH_DOWNWIND_MULT: stage-1 multiplier for downwind consistency loss.
STAGE1_HIGH_DOWNWIND_MULT = 1.0

# STAGE1_SOURCE_LOCAL_MULT: stage-1 multiplier for source-local dominance loss.
STAGE1_SOURCE_LOCAL_MULT = 0.0

# MAX_GRAD_NORM: gradient clipping threshold for training stability (None or <=0 disables).
MAX_GRAD_NORM = 10.0

# DEBUG_EVERY: print field/PDE component diagnostics every N epochs.
DEBUG_EVERY = 500

# VISUALIZE_GATE_ONLY: when True, animation shows source_gate instead of concentration.
VISUALIZE_GATE_ONLY = False

# ADD_BASELINE_TO_VIZ: when training on residual plume, add observed baseline back in animation.
ADD_BASELINE_TO_VIZ = True

# LOSS_W_TOP_STATION: enforce that the highest observed station at each timestamp remains the highest predicted one.
LOSS_W_TOP_STATION = 0.0

# LOSS_W_MULTI_HIGH: enforce simultaneous fitting of multiple high-valued stations at the same timestamp.
LOSS_W_MULTI_HIGH = 0.0

# MULTI_HIGH_RATIO: stations above this fraction of the timestamp maximum residual are treated as joint high-value points.
MULTI_HIGH_RATIO = 0.5

# MULTI_HIGH_MIN_RELIEF: skip the multi-high constraint if the timestamp has weak anomaly contrast.
MULTI_HIGH_MIN_RELIEF = 0.15

# MULTI_HIGH_MARGIN: high-value stations should exceed non-high stations by at least this normalized margin.
MULTI_HIGH_MARGIN = 0.05

# LOSS_W_HIGH_DOWNWIND: require clearly anomalous observed stations to lie downwind of the source.
LOSS_W_HIGH_DOWNWIND = 2.0

# HIGH_DOWNWIND_RATIO: within each timestamp, stations above this fraction of the max residual are treated as anomalous peaks.
HIGH_DOWNWIND_RATIO = 0.6

# HIGH_DOWNWIND_MIN_RELIEF: skip downwind constraint if the timestamp has no clear anomaly contrast.
HIGH_DOWNWIND_MIN_RELIEF = 0.15

# HIGH_DOWNWIND_MARGIN: minimum normalized downwind projection expected for anomalous stations.
HIGH_DOWNWIND_MARGIN = 0.03

# Low wind protection: downwind/axis constraints are weaker when raw wind speed is unreliable.
LOW_WIND_SPEED_THRESHOLD = 0.5
DOWNWIND_LOSS_LOW_WIND_FACTOR = 0.2
AXIS_LOSS_LOW_WIND_FACTOR = 0.2
CORRIDOR_LOSS_LOW_WIND_FACTOR = 0.2

# =========================
# Fast Source Confidence Landscape
# =========================
# USE_SOURCE_LANDSCAPE_CONFIDENCE:
# After one normal training run, scan fixed source locations and convert the
# loss landscape into a fast pseudo-probability source region.
USE_SOURCE_LANDSCAPE_CONFIDENCE = True
SOURCE_LANDSCAPE_MODE = "local"  # "local" or "source_domain"
SOURCE_LANDSCAPE_RADIUS_M = 450.0
SOURCE_LANDSCAPE_STEP_M = 150.0
SOURCE_LANDSCAPE_TEMPERATURE = 0.1
SOURCE_LANDSCAPE_LEVELS = [0.5, 0.8, 0.95]
SOURCE_LANDSCAPE_INCLUDE_GEOMETRY = False

# Diffusion GIF resolution. These affect post-training generation time only.
DIFFUSION_N_FRAMES = 24
DIFFUSION_NX = 80
DIFFUSION_NY = 80

# EARLY_STOP_START: earliest epoch where convergence-based early stopping can trigger.
EARLY_STOP_START = 1800

# EARLY_STOP_PATIENCE: number of epochs with no meaningful improvement before stopping.
EARLY_STOP_PATIENCE = 500

# EARLY_STOP_MIN_DELTA: minimum raw_loss improvement counted as real progress.
EARLY_STOP_MIN_DELTA = 1e-4

# =========================
# Legacy (currently not used in pipeline)
# =========================
# Kept for compatibility; current pipeline computes dynamic L/T from data span.
SCALE_XY = 1000.0
SCALE_T = 1.0
