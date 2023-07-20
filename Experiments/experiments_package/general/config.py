"""
General options and constants that are used in and are important for
several places.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

USED_STORE_ID = 4051653300272

ALL_USED_PRODUCT_IDS = [
    '4260705920294',
    "4260705920003",
    "4260705920010",
    "4260705920027",
    "4260705920034",
    "4260705920041",
    "4260705920058",
    "4260705920065",
    "4260705920072",
    "4260705920089",
    "4260705920096",
    "4260705920102",
    "4260705920119",
    "4260705920126",
    "4260705920133",
    "4260705920140",
    "4260705920157",
    "4260705920164",
    "4260705920171",
    "4260705920188",
    "4260705920195",
    "4260705920201",
    "4260705920218",
    "4260705920225",
    "4260705920232",
    "4260705920249",
    "4260705920256",
    "4260705920263",
    "4260705920270",
    "4260705920287",
    "4260705920300",
    "4260705920317",
    "4260705920324",
    "4260705920331",
    "4260705920355",
    "4260705920362",
    "4260705920393",
    "4260705920409",
    "4260705920416",
    "4260705920423",
    "4260705920430",
    "4260705920461",
    "4260705920478",
    "4260705920492",
    "4260705920508",
    "4260705920515",
    "4260705920522",
    "4260705920539",
    "4260705920546",
    "4260705920553",
    "4260705920560",
    "4260705920577",
    "4260705920584",
    "4260705920591",
    "4260705920607",
    "4260705920638",
]


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
