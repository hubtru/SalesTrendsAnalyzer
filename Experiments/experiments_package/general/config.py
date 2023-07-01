"""
General options and constants that are used in and are important for
several places.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

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
