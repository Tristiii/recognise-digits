from utils import get_model

import numpy as np
from joblib import load

def predict(data: np.ndarray) -> int:
    model = load(get_model())
    print(data)
    return model.predict(data)[0]