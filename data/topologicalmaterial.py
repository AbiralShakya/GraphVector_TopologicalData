#take for instance this material https://topologicalquantumchemistry.org/#/detail/670212
#capture all info with custom class

import numpy as np
import pandas as pd 
from dataclasses import dataclass
from typing import Dict, List, Optional, Table
import json
import torch
import torch.nn as nn 
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class TopologicalMaterial: 
    compound: str
    icsd_id: str
    symmetry_group: str
    symmetry_group_symbol: str

    lattice_params: Dict[str, float]
    volume: float
    valence_electrons: int

    topological_type: str
    topological_subtype: str

    z2_invariants: Dict[str, int] # (topological indices)
    higher_order_invariants: Dict[str, int] # beyond z2

    band_gaps: Dict[str, float]
    smallest_gap: float
    soc_enabled: bool
