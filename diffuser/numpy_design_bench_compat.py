"""design_bench approximate_oracle uses np.loads; NumPy 2.0 removed it. Import before design_bench."""
import pickle

import numpy as np

if not hasattr(np, "loads"):
    np.loads = pickle.loads  # type: ignore[attr-defined]
