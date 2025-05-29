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
import re 

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


class TopologicalDataProcessor:
    # process and vectorize tqc data
    def __init__(self):
        # TODO: add onto this a lot haha.. 
        self.invariant_names = [
            'Z2w_1', 'Z2w_2', 'Z2w_3', 'Z2', 'Z4', 'Z4pi', 'Z8'
        ]
        self.gap_points = ['Γ', 'M', 'Z', 'R', 'X', 'A'] # common k-points
        self.space_group_encoder = LabelEncoder()
        self.topology_encoder = LabelEncoder()
        
    def parse_material_data(self, data_text: str) -> TopologicalMaterial:
        # parse raw tqc entries per input crystal
            
        formula = self._extract_formula(data_text)
        space_group_info = self._extract_space_group(data_text)
        topo_status = self._extract_topological_status(data_text)
        invariants = self._extract_topological_invariants(data_text)
        crystal_data = self._extract_crystallographic_data(data_text)
        electronic_data = self._extract_electronic_data(data_text)
        
        band_group_reps_data, fragile_bands, ebr_class = self._extract_band_representation_data(data_text)
        symmetry_transitions_data = self._extract_symmetry_transitions_data(data_text)
        
        return TopologicalMaterial(
            formula=formula,
            icsd_id=crystal_data.get('icsd_id'),
            space_group=space_group_info.get('number'),
            space_group_symbol=space_group_info.get('symbol'),
            lattice_params=crystal_data.get('lattice', {}),
            volume=crystal_data.get('volume'),
            valence_electrons=crystal_data.get('valence_electrons'),
            topological_type=topo_status.get('type'),
            topological_subtype=topo_status.get('subtype'),
            z2_invariants=invariants.get('z2', {}),
            higher_order_invariants=invariants.get('higher_order', {}),
            band_gaps=electronic_data.get('gaps', {}),
            smallest_gap=electronic_data.get('smallest_gap'),
            soc_enabled=electronic_data.get('soc'),
            band_group_representations=band_group_reps_data,
            fragile_bands=fragile_bands,                    
            ebr_class=ebr_class,                            
            symmetry_transitions=symmetry_transitions_data   
        )

    def _extract_formula(self, text: str) -> str:
        lines = text.split('\n')
        for line in lines:
            cleaned_line = line.strip().strip('*').strip()
            if cleaned_line and not cleaned_line.lower().startswith(("symmetry group:", "topological status:", "icsd:")):
                # Basic check to avoid grabbing headers if formula is missing or first line is a header
                if re.match(r'^[A-Za-z0-9\s]+$', cleaned_line) and not re.search(r'\d+\s*\(', cleaned_line):
                     return cleaned_line
        return "UnknownFormula"


    def _extract_space_group(self, text: str) -> Dict:
        # Regex to find "Symmetry Group: 123 (P4/mmm)" or "(\*+123\*+) (\*+P4/mmm\*+)" like patterns
        sg_match = re.search(r'(?:Symmetry Group:|\s)\s*(\d+)\s*\(([^)]+)\)', text, re.IGNORECASE)
        if not sg_match: # Try alternative common format from image
            sg_match = re.search(r'Symmetry Group\s+(\d+)\s*\(\*+([^*]+)\*+\)', text, re.IGNORECASE)
        if not sg_match: # For cases like "1 (P1)" directly without "Symmetry Group:"
            sg_match = re.search(r'^\s*(\d+)\s+\(([^)]+)\)', text, re.MULTILINE)


        if sg_match:
            return {
                'number': int(sg_match.group(1)),
                'symbol': sg_match.group(2).strip().replace('*', '')
            }
        return {} # Return empty if not found

    def _extract_topological_status(self, text: str) -> Dict:
        status_match = re.search(r'Topological Status(?:[^:]*):\s*([^(]+)\s*\(([^)]+)\)', text, re.IGNORECASE)
        if status_match:
            return {
                'type': status_match.group(1).strip(),
                'subtype': status_match.group(2).strip()
            }
        return {}

    def _extract_topological_invariants(self, text: str) -> Dict:
        z2_invariants = {}
        higher_order = {}
        
        # More robustly find the line/section with topological indices
        indices_text = text
        indices_match = re.search(r'Topological indices:(.*?)($|\n[A-Z])', text, re.DOTALL | re.IGNORECASE)
        if indices_match:
            indices_text = indices_match.group(1)

        # Using a general pattern for key = value pairs
        # Matches "Z2w,1 = 1", "Z4π = 3", "Z2 = 1" etc.
        all_indices = re.findall(r'([A-Za-z0-9π,]+)\s*=\s*(\d+)', indices_text)
        
        z2_keys = {'Z2w,1', 'Z2w,2', 'Z2w,3', 'Z2'}
        ho_keys = {'Z4', 'Z4π', 'Z8'} # Z4pi needs to be normalized if Z4π is used

        for key, value in all_indices:
            normalized_key = key.replace('Z4π', 'Z4pi') # Normalize if needed
            if normalized_key in z2_keys:
                z2_invariants[normalized_key] = int(value)
            elif normalized_key in ho_keys:
                higher_order[normalized_key] = int(value)
            # else: could store unknown/other indices if necessary
            
        return {'z2': z2_invariants, 'higher_order': higher_order}

    def _extract_crystallographic_data(self, text: str) -> Dict:
        data = {}
        
        icsd_match = re.search(r'ICSD:\s*(\d+)', text, re.IGNORECASE)
        data['icsd_id'] = icsd_match.group(1) if icsd_match else None
        
        lattice = {}
        # Adjusted regex to be more flexible with spacing and labels
        patterns = [
            (r'Cell Length A\s*([\d.]+)', 'a'), (r'a\s*=\s*([\d.]+)', 'a'),
            (r'Cell Length B\s*([\d.]+)', 'b'), (r'b\s*=\s*([\d.]+)', 'b'),
            (r'Cell Length C\s*([\d.]+)', 'c'), (r'c\s*=\s*([\d.]+)', 'c'),
            (r'Cell Angle alpha\s*([\d.]+)', 'alpha'), (r'Cell Angle α\s*([\d.]+)', 'alpha'), (r'alpha\s*=\s*([\d.]+)', 'alpha'),
            (r'Cell Angle beta\s*([\d.]+)', 'beta'), (r'Cell Angle β\s*([\d.]+)', 'beta'), (r'beta\s*=\s*([\d.]+)', 'beta'),
            (r'Cell Angle gamma\s*([\d.]+)', 'gamma'), (r'Cell Angle γ\s*([\d.]+)', 'gamma'), (r'gamma\s*=\s*([\d.]+)', 'gamma'),
        ]
        for pattern, key in patterns:
            if key not in lattice: # Prioritize first match for each key
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    lattice[key] = float(match.group(1))
        data['lattice'] = lattice
        
        vol_match = re.search(r'Cell Volume\s*([\d.]+)', text, re.IGNORECASE)
        data['volume'] = float(vol_match.group(1)) if vol_match else None
        
        val_match = re.search(r'Nr of valence electrons:\s*(\d+)', text, re.IGNORECASE)
        data['valence_electrons'] = int(val_match.group(1)) if val_match else None
        
        return data

    def _extract_electronic_data(self, text: str) -> Dict:
        gaps = {}
        # Simpler extraction for gaps, assuming they appear as "K-POINT: VALUE" or similar
        # This part is highly dependent on actual text format for gaps.
        # The provided image shows a plot, not raw text for gaps at k-points.
        # If you have text like "Gap at Gamma: 0.1 eV", adapt regex here.
        # For now, this remains a placeholder or relies on the old simpler logic
        gap_section_match = re.search(r'Gaps at high-symmetry points.*?:(.*?)(?:Smallest Computed Gap|Nr Fragile Bands|EBR equivalence class|---)', text, re.DOTALL | re.IGNORECASE)
        if gap_section_match:
            gap_text = gap_section_match.group(1)
            # Example: expecting "Gamma 0.1 M 0.2 Z 0.05" or "Gamma:0.1, M:0.2"
            # This regex is very basic and might need significant improvement
            found_gaps = re.findall(r'([ΓΜΖRXAΥΣ])\s*[:\s-]?\s*([\d.-]+)', gap_text)
            for point, value in found_gaps:
                gaps[point] = float(value)

        smallest_match = re.search(r'Smallest Computed Gap\s*([\d.-]+)', text, re.IGNORECASE)
        smallest_gap = float(smallest_match.group(1)) if smallest_match else None
        
        soc_enabled = None
        soc_match = re.search(r'With SOC:\s*(Yes|No)', text, re.IGNORECASE)
        if soc_match:
            soc_enabled = soc_match.group(1).lower() == 'yes'
            
        return {'gaps': gaps, 'smallest_gap': smallest_gap, 'soc': soc_enabled}

    def _extract_band_representation_data(self, text: str) -> Tuple[List[BandGroupRepresentation], Optional[int], Optional[int]]:
        band_groups: List[BandGroupRepresentation] = []
        fragile_bands_overall: Optional[int] = None
        ebr_class_overall: Optional[int] = None

        # Overall fragile bands and EBR class (if listed once at the end)
        fragile_match_overall = re.search(r'Nr Fragile Bands\s*(\d+)', text, re.IGNORECASE)
        if fragile_match_overall:
            fragile_bands_overall = int(fragile_match_overall.group(1))

        ebr_match_overall = re.search(r'EBR equivalence class\s*(\d+)', text, re.IGNORECASE)
        if ebr_match_overall:
            ebr_class_overall = int(ebr_match_overall.group(1))

        # This is a complex table. We'll try to find the section and parse row by row.
        # The header is assumed to be something like "Nr. Bands Subclass Γ M Z R X A"
        # The pattern looks for the start of the table.
        # We need to find the start of the table "Band representations and their degeneracies"
        # or a line that looks like a header: Nr. Bands Subclass Γ M Z R X A
        header_pattern = r"(?:Band representations and their degeneracies|Nr\. Bands\s+Subclass\s+[ΓΓSYM\s]+[MMASYM\s]+[ZZASYM\s]+[RRASYM\s]+[XXASYM\s]+[AAASYM\s]+)"
        
        # Look for the start of the section related to band representations
        # It might be preceded by "Band representations and their degeneracies"
        # Or directly the table header.
        
        # Let's assume the table data starts after a header line like the one in the image
        # (Nr. Bands Subclass Γ M Z R X A) and ends before "Transitions upon symmetry lowering"
        # or another major section.

        table_section_match = re.search(r"(Nr\. Bands\s+Subclass\s+Γ\s+M\s+Z\s+R\s+X\s+A.*?)(?:\n\n|Transitions upon symmetry lowering|Minimal subgroups are highlighted)", text, re.DOTALL | re.IGNORECASE)

        if table_section_match:
            table_text = table_section_match.group(1)
            lines = table_text.strip().split('\n')
            
            # First line is header, find k-point names from it
            header_line = lines[0].strip()
            k_point_header_names = re.findall(r'([ΓΜΖRXAΥΣ]\S*)', header_line)
            # Default if not found in header
            if not k_point_header_names: k_point_header_names = ['Γ', 'M', 'Z', 'R', 'X', 'A']


            for line in lines[1:]: # Skip header
                line = line.strip()
                if not line: continue

                parts = re.split(r'\s+', line, maxsplit=len(k_point_header_names) + 1) # num_bands, subclass, then k-points
                
                if len(parts) < 2: continue # Need at least num_bands and subclass

                current_group = BandGroupRepresentation()
                try:
                    current_group.num_bands = int(parts[0])
                except ValueError:
                    # Might be a continuation line or malformed, skip
                    continue 
                current_group.subclass = parts[1]
                
                rep_data_parts = parts[2:]

                for i, k_point_name in enumerate(k_point_header_names):
                    if i < len(rep_data_parts):
                        rep_str = rep_data_parts[i] # e.g., "7(1)" or "7"
                        irrep_match = re.match(r'(\S+)\s*\((\d+)\)', rep_str) # Label(Degeneracy)
                        if irrep_match:
                            current_group.representations_at_k_points.append(
                                BandRepresentationDetail(
                                    k_point=k_point_name,
                                    irrep_label=irrep_match.group(1),
                                    degeneracy=int(irrep_match.group(2))
                                )
                            )
                        elif rep_str.strip() and rep_str.strip() != '-': # Single label, assume degeneracy 1 or unknown
                             current_group.representations_at_k_points.append(
                                BandRepresentationDetail(
                                    k_point=k_point_name,
                                    irrep_label=rep_str.strip()
                                    # degeneracy will be None (default)
                                )
                            )
                if current_group.num_bands is not None : # only add if num_bands was parsed
                    band_groups.append(current_group)
        
        return band_groups, fragile_bands_overall, ebr_class_overall


    def _extract_symmetry_transitions_data(self, text: str) -> List[SymmetryTransition]:
        transitions: List[SymmetryTransition] = []
        
        # Find the section for symmetry transitions
        section_match = re.search(r'Transitions upon symmetry lowering(.*?)(?:EBR equivalence class|Minimal subgroups are highlighted|$)', text, re.DOTALL | re.IGNORECASE)
        if not section_match:
            return transitions
            
        content = section_match.group(1).strip()
        lines = content.split('\n')

        current_transition: Optional[SymmetryTransition] = None

        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("symmetry group") or line.lower().startswith("---"): # Skip headers or empty lines
                continue

            # Try to match a new symmetry group line e.g., "1 (P1)" or "83 (P4/m)"
            sg_match = re.match(r'(\d+)\s*\(([^)]+)\)', line)
            if sg_match: # This line defines a new target symmetry group
                if current_transition: # Save previous one if exists
                    transitions.append(current_transition)
                
                current_transition = SymmetryTransition(
                    target_symmetry_group_number=int(sg_match.group(1)),
                    target_symmetry_group_symbol=sg_match.group(2).strip()
                )
                # Check if the rest of the line contains topological indices for this new SG
                remaining_line_for_sg = line[sg_match.end():].strip()
                if remaining_line_for_sg:
                    indices_on_sg_line = re.findall(r'([A-Za-z0-9π,]+)\s*=\s*(\d+)', remaining_line_for_sg)
                    for key, value in indices_on_sg_line:
                        if current_transition: # mypy check
                             current_transition.topological_indices[key.replace('Z4π', 'Z4pi')] = int(value)
            
            # If not a new SG line, it might be topological indices for the current_transition
            elif current_transition:
                indices_on_line = re.findall(r'([A-Za-z0-9π,]+)\s*=\s*(\d+)', line)
                if indices_on_line:
                    for key, value in indices_on_line:
                        current_transition.topological_indices[key.replace('Z4π', 'Z4pi')] = int(value)
                elif not current_transition.band_representations_info: # If no indices, could be band rep info (simplified)
                    current_transition.band_representations_info = line # Store as raw string for now

        if current_transition: # Add the last parsed transition
            transitions.append(current_transition)
            
        return transitions


class TopologicalVectorizer:
    """Convert topological materials to ML-ready vectors"""
    
    def __init__(self):
        self.feature_dim = 128
        self.scalers = {}
        
    def create_invariant_vector(self, material: TopologicalMaterial) -> np.ndarray:
        """Create vector from topological invariants"""
        
        # Z2 invariants (4 components)
        z2_vector = np.zeros(4)
        z2_names = ['Z2w_1', 'Z2w_2', 'Z2w_3', 'Z2']
        for i, name in enumerate(z2_names):
            if name in material.z2_invariants:
                z2_vector[i] = material.z2_invariants[name]
        
        # Higher-order invariants (3 components)
        ho_vector = np.zeros(3)
        ho_names = ['Z4', 'Z4pi', 'Z8']
        for i, name in enumerate(ho_names):
            if name in material.higher_order_invariants:
                ho_vector[i] = material.higher_order_invariants[name]
        
        # Combine invariants
        invariant_vector = np.concatenate([z2_vector, ho_vector])
        
        return invariant_vector
    
    def create_structure_vector(self, material: TopologicalMaterial) -> np.ndarray:
        """Create vector from crystal structure"""
        
        # Lattice parameters (6 components)
        lattice_vector = np.zeros(6)
        lattice_keys = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for i, key in enumerate(lattice_keys):
            if key in material.lattice_params:
                lattice_vector[i] = material.lattice_params[key]
        
        # Volume and electrons
        volume_vector = np.array([
            material.volume,
            material.valence_electrons,
            material.space_group
        ])
        
        return np.concatenate([lattice_vector, volume_vector])
    
    def create_electronic_vector(self, material: TopologicalMaterial) -> np.ndarray:
        """Create vector from electronic properties"""
        
        # Band gaps at high-symmetry points
        gap_vector = np.zeros(6)
        gap_points = ['Γ', 'M', 'Z', 'R', 'X', 'A']
        for i, point in enumerate(gap_points):
            if point in material.band_gaps:
                gap_vector[i] = material.band_gaps[point]
        
        # Additional electronic features
        electronic_features = np.array([
            material.smallest_gap,
            1.0 if material.soc_enabled else 0.0,
            material.fragile_bands,
            material.ebr_class
        ])
        
        return np.concatenate([gap_vector, electronic_features])
    
    def create_topology_vector(self, material: TopologicalMaterial) -> np.ndarray:
        # comprehensive topological vector
        #         
        invariant_vec = self.create_invariant_vector(material)
        structure_vec = self.create_structure_vector(material)
        electronic_vec = self.create_electronic_vector(material)
        
        # Pad to fixed size
        full_vector = np.zeros(self.feature_dim)
        
        # Assign sections
        inv_end = min(len(invariant_vec), 20)
        struct_end = inv_end + min(len(structure_vec), 20)
        elec_end = struct_end + min(len(electronic_vec), 20)
        
        full_vector[:inv_end] = invariant_vec[:inv_end]
        full_vector[inv_end:struct_end] = structure_vec[:min(len(structure_vec), 20)]
        full_vector[struct_end:elec_end] = electronic_vec[:min(len(electronic_vec), 20)]
        
        return full_vector

class TopologicalClassifier(nn.Module):
    # place in nn for topological classification (just replace with my prev multi task classifier from other repo)
    
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.topology_classifier = nn.Linear(128, num_classes)
        self.gap_regressor = nn.Linear(128, 1)
        self.invariant_regressor = nn.Linear(128, 7)  # 7 topological invariants
        
    def forward(self, x):
        features = self.feature_encoder(x)
        
        return {
            'topology_class': self.topology_classifier(features),
            'band_gap': self.gap_regressor(features),
            'invariants': self.invariant_regressor(features)
        }

#  TODO: fix this a lot
def process_h2sc1_example():
    # proces H2Sc1 crystal as an example ... need to build more robust
    
    # sample data
    sample_data = """***H2 Sc1***
Symmetry Group: 123 (P4/mmm)
Topological Status (Type): TI (NLC)
Topological indices: Z2w,1 = 1, Z2w,2 = 1, Z2w,3 = 1, Z4 = 1, Z2 = 1, Z4π = 3, Z8 = 5
ICSD: 670212
Cell Length A 2.633 Cell Length B 2.633 Cell Length C 8.324 Cell Angle α 90 Cell Angle β 90 Cell Angle γ 90 Cell Volume 57.7
Nr of valence electrons: 20.
Smallest Computed Gap 0.0349
With SOC: Yes
Nr Fragile Bands 0
EBR equivalence class 40

Band representations and their degeneracies
Nr. Bands Subclass Γ   M   Z   R   X   A
4         NLC      7(1) 9(1) 6(1) 7(1) 7(1) 9(1)
                             5(2) 5(2) 6(1) 9(1)
2         NLC      7(1) 6(1) 9(1) 6(1) 6(1) 8(1)

Transitions upon symmetry lowering
Symmetry Group      Topological Indices                             Band representations
1 (P1)
2 (P-1)             Z2w,1 = 1, Z2w,2 = 1, Z2w,3 = 1, Z4 = 1
3 (P2)
5 (C2)
6 (Pm)
8 (Cm)
10 (P2/m)           Z2w,1 = 1, Z2w,2 = 1, Z2w,3 = 1, Z4 = 1
83 (P4/m)           Z2w,1 = 1, Z2w,2 = 1, Z2w,3 = 1, Z4 = 1, Z2 = 1, Z4π = 3, Z8 = 5
123 (P4/mmm)        Z2w,1 = 1, Z2w,2 = 1, Z2w,3 = 1, Z4 = 1, Z2 = 1, Z4π = 3, Z8 = 5

Minimal subgroups are highlighted
"""
    
    processor = TopologicalDataProcessor()
    
    material = processor.parse_material_data(sample_data)
    
    print(f"--- Parsed Material: {material.formula} ---")
    print(f"Space Group: {material.space_group} ({material.space_group_symbol})")
    print(f"Topological Type: {material.topological_type} ({material.topological_subtype})")
    print(f"Z2 Invariants: {material.z2_invariants}")
    print(f"Higher-Order Invariants: {material.higher_order_invariants}")
    print(f"Smallest Gap: {material.smallest_gap}, SOC: {material.soc_enabled}")
    print(f"Fragile Bands: {material.fragile_bands}, EBR Class: {material.ebr_class}")

    print("\n--- Band Group Representations ---")
    for bg_rep in material.band_group_representations:
        print(f"  Num Bands: {bg_rep.num_bands}, Subclass: {bg_rep.subclass}")
        for rep_detail in bg_rep.representations_at_k_points:
            print(f"    K-Point: {rep_detail.k_point}, Irrep: {rep_detail.irrep_label}, Degeneracy: {rep_detail.degeneracy}")

    print("\n--- Symmetry Transitions ---")
    for trans in material.symmetry_transitions:
        print(f"  To SG: {trans.target_symmetry_group_number} ({trans.target_symmetry_group_symbol})")
        print(f"    Indices: {trans.topological_indices}")
        if trans.band_representations_info:
            print(f"    Band Rep Info: {trans.band_representations_info}")
            
    # proceed to vectorization with an updated TopologicalVectorizer
    # vectorizer = TopologicalVectorizer() 
    # topology_vector = vectorizer.create_topology_vector(material) # TODO: this method needs update
    # print(f"\nTopology vector shape (example, needs updated vectorizer): {topology_vector.shape}")
