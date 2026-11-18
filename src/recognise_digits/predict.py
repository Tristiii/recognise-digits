from utils import get_model

import numpy as np
from joblib import load

def predict(data: np.ndarray) -> int:
    model = load(get_model())
    return model.predict(data)[0]