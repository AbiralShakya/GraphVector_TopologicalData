#take for instance this material https://topologicalquantumchemistry.org/#/detail/670212
#capture all info with custom class

import numpy as np
import pandas as pd 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Table, Any
import json
import torch
import torch.nn as nn 
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass 
class BandRepresentationDetail: 
    #irreducible representations at k point for band group
    k_point: str
    irrep_label: Optional[str] = None 
    degeneracy: Optional[int] = None
    energy: Optional[int] = None

@dataclass
class BandGroupRepresentation:
    #row from Band representations and their degeneracies table in tqc
    num_bands: Optional[int] = None 
    subclass: Optional[str] = None
    representations_at_k_points: List[BandRepresentationDetail] = field(default_factory=list)
    representations_at_k_points_by_level_per:  List[BandRepresentationDetail] = field(default_factory=list)
    representations_at_k_points_up_to_a_band:  List[BandRepresentationDetail] = field(default_factory=list)

@dataclass
class SymmetryTransition:
    #row from the Transitions upon symmetry lowering table in tqc
    target_symmetry_group_number: Optional[int] = None
    target_symmetry_group_symbol: Optional[str] = None
    topological_indices: Dict[str, Any] = field(default_factory=dict)
    band_representations_info: Optional[str] = None

@dataclass
class TopologicalMaterial: 
    compound: str
    icsd_id: str
    symmetry_group: str
    symmetry_group_symbol: str

    lattice_params: Dict[str, float] = field(default_factory= dict)
    volume: float
    valence_electrons: int

    topological_type: str
    topological_subtype: str

    z2_invariants: Dict[str, int] = field(default_factory= dict) # (topological indices)
    higher_order_invariants: Dict[str, int] = field(default_factory= dict) # beyond z2

    band_gaps: Dict[str, float] = field(default_factory= dict)
    smallest_gap: float
    soc_enabled: bool

    band_group_representations: List[BandGroupRepresentation] = field(default_factory=list)
    fragile_bands: Optional[int] = None
    ebr_class: Optional[int] = None # elementary Band Representation class

    symmetry_transitions: List[SymmetryTransition] = field(default_factory=list)
