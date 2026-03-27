# Configuration
SITE_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\sites.xlsx"
CONC_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\concentration.xlsx"
WIND_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\wind.xlsx"

# If your wind direction is "from" (meteorological), keep True.
# If your dir is "to" (where wind is blowing toward), set False.
WIND_DIR_IS_FROM = True

# Training hyperparams
EPOCHS = 5000
LR = 1e-3
N_COLLOCATION = 5000
# Model selection
MODEL_NAME = "pinn"

