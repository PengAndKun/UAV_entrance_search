from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def serializable_or2_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): jsonable(value)
        for key, value in prediction.items()
        if key not in {"clearance_warning_mask", "obstacle_warning_mask", "must_stop_mask"}
    }
