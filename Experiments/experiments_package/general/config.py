"""
General options and constants that are used in and are important for
several places.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

USED_STORE_ID = 4051653300272


class ProductIds(Enum):
    BENS_LUNCHTIME = "4260705920294"


@dataclass
class DatasetOptions:
    window_width: int
    label_width: int
    shift: int
    data_origin: str
    drop_columns: List[str]
    label_columns: Optional[List[str]] = None


def normalize(data, norm_values_to_use=None):
    norm_values_used = norm_values_to_use
    if norm_values_to_use is None:
        std = data.std()
        std = std.where(lambda x: np.invert(np.isclose(x, 0)), other=0.3)
        norm_values_used = {
            "std": std,
            "mean": data.mean(),
        }

    return (data - norm_values_used["mean"]) / (
        norm_values_used["std"]
    ), norm_values_used


def denormalize(data, used_normalization, labels=None):
    if labels is None:
        labels = slice(None)
    return (
            data * used_normalization["std"][labels]
            + used_normalization["mean"][labels]
    )


def denormalize_list(data_list, used_normalization, label):
    data_list = np.array(data_list)
    return (
            data_list * used_normalization["std"][label]
            + used_normalization["mean"][label]
    )
