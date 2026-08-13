import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Data lives inside the repo by default; override on Sherlock so the
# large .fits files sit on SCRATCH instead of HOME.
DATA_ROOT = Path(os.environ.get("HMC_DUST_DATA", ROOT))

DATA = DATA_ROOT / "Data_And_Samplers"
GAIA = DATA_ROOT / "Gaia XP"