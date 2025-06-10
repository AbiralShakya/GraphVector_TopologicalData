# all of this is for non magnetic
# rather than saving k space graph for all, just link it to given space group's k space graph
# homology point cloud for atom build, vectorize it
# atom graph from poscar build
# use materials project instead of jarvis since we can query 2d materials in mat project and mat project has all the relevant materials


import torch
import pandas as pd
import h5py
import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict
import pickle
from build_dataset_test import TopologicalMaterialAnalyzer 
import time
import tqdm

from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# Assuming TopologicalMaterialAnalyzer and its dependencies are available
# (e.g., from build_dataset_test.py or defined within this file)
# If TopologicalMaterialAnalyzer uses JARVIS data, you'll need to adapt it
# to accept Structure objects from pymatgen, or to fetch data itself.

# --- Keep your existing classes and methods (MaterialRecord, MultiModalMaterialDatabase, POSCARReader) ---
# (I'll omit them here for brevity, assuming they are in your full script)

@dataclass
class MaterialRecord:
    """Structured material record with all 4 data modalities"""
    jid: str
    formula: str
    normalized_formula: str
    space_group: str
    space_group_number: int
    
    lattice_matrix: np.ndarray  # 3x3
    atomic_positions: np.ndarray  # Nx3
    atomic_numbers: List[int]
    elements: List[str]
    
    crystal_graph: Dict  # Real-space graph
    kspace_graph: Dict   # K-space graph
    
    asph_features: np.ndarray
    
    topological_class: str  # 'TI', 'SM', 'NI', etc.
    topological_binary: float  # 0 or 1
    band_gap: Optional[float]
    formation_energy: Optional[float]
    
    symmetry_operations: Dict
    local_database_props: Dict
    processing_timestamp: str

class MultiModalMaterialDatabase:
    """
    Enhanced database structure for multi-modal materials data
    Supports multiple storage formats and efficient querying
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create organized directory structure"""
        directories = [
            'raw_data',           # Original JARVIS + local data
            'structures',         # POSCAR files and structure data
            'graphs',            # Graph representations
            'point_clouds',      # Topological/ASPH features
            'metadata',          # JSON metadata and indices
            'datasets',          # Processed datasets for ML
            'analysis'           # Analysis results and visualizations
        ]
        
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_material_record(self, record: MaterialRecord, 
                           format_type: str = 'hybrid') -> None:
        """
        Save material record in specified format
        
        Formats:
        - 'hybrid': Combination of HDF5 + JSON + individual files
        - 'hdf5': Everything in HDF5 (efficient, but less human-readable)
        - 'individual': Separate files for each component (most flexible)
        """
        
        if format_type == 'hybrid':
            self._save_hybrid_format(record)
        elif format_type == 'hdf5':
            self._save_hdf5_format(record)
        elif format_type == 'individual':
            self._save_individual_format(record)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _save_hybrid_format(self, record: MaterialRecord):
        """Hybrid format: HDF5 for arrays, JSON for metadata, separate graph files"""
        
        # 1. Structure data in HDF5 (efficient for arrays)
        structure_path = self.base_path / 'structures' / f"{record.jid}.h5" 
        with h5py.File(structure_path, 'w') as f:
            f.create_dataset('lattice_matrix', data=record.lattice_matrix)
            f.create_dataset('atomic_positions', data=record.atomic_positions)
            f.create_dataset('atomic_numbers', data=record.atomic_numbers)
            
            # Store strings as fixed-length or variable-length
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('elements', data=np.array(record.elements, dtype=object), dtype=dt)
        
        # 2. Point cloud data as separate numpy file
        point_cloud_path = self.base_path / 'point_clouds' / f"{record.jid}_asph.npy"
        np.save(point_cloud_path, record.asph_features)

        # 3. Graph data as PyTorch files (preserves tensor structure)
        graph_dir = self.base_path / 'graphs' / record.jid
        graph_dir.mkdir(exist_ok=True)
        
        torch.save(record.crystal_graph, graph_dir / 'crystal_graph.pt')
        torch.save(record.kspace_graph, graph_dir / 'kspace_graph.pt')

        # 4. Create POSCAR file
        poscar_path = self.base_path / 'structures' / f"{record.jid}_POSCAR"
        poscar_content = self._create_poscar_string(record)
        with open(poscar_path, 'w') as f:
            f.write(poscar_content)
        
        # 5. Metadata as JSON (human-readable)
        metadata = {
            'jid': record.jid,
            'formula': record.formula,
            'normalized_formula': record.normalized_formula,
            'space_group': record.space_group,
            'space_group_number': record.space_group_number,
            'topological_class': record.topological_class,
            'topological_binary': record.topological_binary,
            'band_gap': record.band_gap,
            'formation_energy': record.formation_energy,
            'processing_timestamp': record.processing_timestamp,
            'local_database_props': record.local_database_props,
            'num_atoms': len(record.elements),
            'file_locations': {
                'structure_hdf5': str(structure_path.relative_to(self.base_path)),
                'poscar': str(poscar_path.relative_to(self.base_path)),
                'point_cloud': str(point_cloud_path.relative_to(self.base_path)),
                'crystal_graph': str((graph_dir / 'crystal_graph.pt').relative_to(self.base_path)),
                'kspace_graph': str((graph_dir / 'kspace_graph.pt').relative_to(self.base_path))
            }
        }
        
        metadata_path = self.base_path / 'metadata' / f"{record.jid}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_hdf5_format(self, record: MaterialRecord):
        """Everything in HDF5 (efficient, but less human-readable)"""
        # This method is not fully implemented in your provided code
        # You'd need to serialize all components (including graphs) into HDF5 datasets
        pass

    def _save_individual_format(self, record: MaterialRecord):
        """Individual files for maximum flexibility and interoperability"""
        
        jid_dir = self.base_path / 'individual' / record.jid
        jid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. POSCAR file (standard crystal structure format)
        poscar_content = self._create_poscar_string(record)
        with open(jid_dir / 'POSCAR', 'w') as f:
            f.write(poscar_content)
        
        # 2. Crystal graph as GraphML (interoperable)
        self._save_graph_as_graphml(record.crystal_graph, jid_dir / 'crystal_graph.graphml')
        
        # 3. K-space graph as GraphML
        self._save_graph_as_graphml(record.kspace_graph, jid_dir / 'kspace_graph.graphml')
        
        # 4. Point cloud data as CSV
        asph_df = pd.DataFrame(record.asph_features.reshape(1, -1))
        asph_df.to_csv(jid_dir / 'asph_features.csv', index=False)
        
        # 5. Complete metadata as JSON
        metadata = asdict(record)
        # Convert numpy arrays to lists for JSON serialization
        metadata['lattice_matrix'] = record.lattice_matrix.tolist()
        metadata['atomic_positions'] = record.atomic_positions.tolist()
        metadata['asph_features'] = record.asph_features.tolist()
        
        with open(jid_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_master_index(self) -> pd.DataFrame:
        """Create a master index/catalog of all materials"""
        
        records = []
        metadata_dir = self.base_path / 'metadata'
        
        for json_file in metadata_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                records.append(metadata)
        
        df = pd.DataFrame(records)
        
        # Save as both CSV (human-readable) and parquet (efficient)
        df.to_csv(self.base_path / 'master_index.csv', index=False)
        df.to_parquet(self.base_path / 'master_index.parquet', index=False)
        
        return df
    
    def create_ml_datasets(self, train_split: float = 0.8, 
                          val_split: float = 0.1) -> Dict[str, str]:
        """Create train/val/test splits for ML"""
        
        df = self.create_master_index()
        
        # Stratified split by topological class
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, train_size=train_split, 
            stratify=df['topological_binary'],
            random_state=42
        )
        
        # Second split: val vs test
        val_size = val_split / (1 - train_split)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size,
            stratify=temp_df['topological_binary'],
            random_state=42
        )
        
        # Save splits
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        split_paths = {}
        
        for split_name, split_df in splits.items():
            split_path = self.base_path / 'datasets' / f'{split_name}_split.csv'
            split_df.to_csv(split_path, index=False)
            split_paths[split_name] = str(split_path)
            
            print(f"{split_name.capitalize()} set: {len(split_df)} materials")
            print(f"  Topological: {(split_df['topological_binary'] == 1).sum()}")
            print(f"  Non-topological: {(split_df['topological_binary'] == 0).sum()}")
        
        return split_paths
    
    def _create_poscar_string(self, record: MaterialRecord) -> str:
        """Create POSCAR format string from material record"""
        lines = []
        lines.append(f"{record.formula} # {record.jid}")
        lines.append("1.0")  # Scaling factor
        
        # Lattice vectors
        for row in record.lattice_matrix:
            lines.append(f"  {row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}")
        
        # Element symbols and counts
        unique_elements = []
        element_counts = []
        for element in record.elements:
            if element not in unique_elements:
                unique_elements.append(element)
                element_counts.append(1)
            else:
                idx = unique_elements.index(element)
                element_counts[idx] += 1
        
        lines.append("  ".join(unique_elements))
        lines.append("  ".join(map(str, element_counts)))
        lines.append("Direct")  # Fractional coordinates
        
        # Convert cartesian to fractional coordinates
        lattice_inv = np.linalg.inv(record.lattice_matrix)
        fractional_positions = record.atomic_positions @ lattice_inv
        
        # Atomic positions
        for pos in fractional_positions:
            lines.append(f"  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        
        return "\n".join(lines)
    
    def _save_graph_as_graphml(self, graph_data: Dict, filepath: Path):
        """Save graph in GraphML format for interoperability"""
        # This is a simplified version - you'd need to implement
        # conversion from your graph format to NetworkX/GraphML
        pass

class POSCARReader:
    """Utility class to read and parse POSCAR files"""
    
    @staticmethod
    def read_poscar(poscar_path: str) -> Dict:
        """
        Read POSCAR file and extract structure information
        Returns dict with lattice_matrix, atomic_positions, elements, etc.
        """
        try:
            with open(poscar_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            
            # Parse POSCAR format
            comment = lines[0]
            scale_factor = float(lines[1])
            
            # Lattice vectors (lines 2-4)
            lattice_matrix = np.zeros((3, 3))
            for i in range(3):
                vector = [float(x) for x in lines[2 + i].split()]
                lattice_matrix[i] = vector
            
            # Scale the lattice
            lattice_matrix *= scale_factor
            
            # Element names (line 5)
            element_line = lines[5].split()
            elements = element_line
            
            # Element counts (line 6)
            count_line = [int(x) for x in lines[6].split()]
            
            # Create full element list
            full_elements = []
            atomic_numbers = []
            for elem, count in zip(elements, count_line):
                full_elements.extend([elem] * count)
                # Convert element symbol to atomic number
                atomic_numbers.extend([POSCARReader.element_to_atomic_number(elem)] * count)
            
            # Coordinate mode (line 7)
            coord_mode = lines[7].lower()
            direct_coords = coord_mode.startswith('d')
            
            # Atomic positions (starting from line 8)
            total_atoms = sum(count_line)
            positions = []
            for i in range(8, 8 + total_atoms):
                pos = [float(x) for x in lines[i].split()[:3]]  # Take only x,y,z
                positions.append(pos)
            
            positions = np.array(positions)
            
            # Convert direct to cartesian if needed
            if direct_coords:
                # Convert fractional coordinates to cartesian
                positions = positions @ lattice_matrix
            
            return {
                'comment': comment,
                'lattice_matrix': lattice_matrix,
                'atomic_positions': positions,
                'elements': full_elements,
                'atomic_numbers': atomic_numbers,
                'total_atoms': total_atoms,
                'formula': POSCARReader.create_formula(elements, count_line)
            }
            
        except Exception as e:
            print(f"Error reading POSCAR file {poscar_path}: {str(e)}")
            return None
    
    @staticmethod
    def element_to_atomic_number(element: str) -> int:
        """Convert element symbol to atomic number"""
        element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92
        }
        return element_to_z.get(element, 0)
    
    @staticmethod
    def create_formula(elements: List[str], counts: List[int]) -> str:
        """Create chemical formula from elements and counts"""
        formula_parts = []
        for elem, count in zip(elements, counts):
            if count == 1:
                formula_parts.append(elem)
            else:
                formula_parts.append(f"{elem}{count}")
        return "".join(formula_parts)

class EnhancedTopologicalMaterialAnalyzer(TopologicalMaterialAnalyzer):
    """
    Enhanced analyzer that uses the new database structure and can integrate
    with Materials Project data.
    
    Note: The 'TopologicalMaterialAnalyzer' class needs to be adapted or
    defined elsewhere to handle pymatgen Structure objects if it currently
    relies heavily on JARVIS-specific data structures.
    """
    def __init__(self, csv_path: str, sqlite_db_path: str, output_database_path: str, mp_api_key: str):
        super().__init__(csv_path, sqlite_db_path)
        self.database = MultiModalMaterialDatabase(output_database_path)
        self.mp_rest = MPRester(mp_api_key)
        self.structure_matcher = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=5) # Tolerance for matching structures

    def get_mp_material_data(self, formula: str, space_group_number: int) -> Optional[Dict]:
        """
        Query Materials Project for material data matching formula and space group.
        Returns the first matching entry or None.
        """
        print(f"Searching Materials Project for Formula: {formula}, Space Group: {space_group_number}")
        try:
            # Query Materials Project by formula
            # Note: Using 'fields' parameter instead of 'properties'
            docs = self.mp_rest.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure", "band_gap",
                    "formation_energy_per_atom", "symmetry"]
            )
            
            if not docs:
                print(f"No entries found in MP for formula: {formula}")
                return None
            
            # Filter by space group number
            target_sg_num = space_group_number
            for doc in docs:
                # Debug: Print what we got
                print(f"  Found entry: {doc.material_id} (Formula: {doc.formula_pretty})")
                
                # Try multiple ways to access space group info
                sg_number = None
                
                # Method 1: Direct symmetry attribute
                if hasattr(doc, 'symmetry') and doc.symmetry is not None:
                    if hasattr(doc.symmetry, 'number'):
                        sg_number = doc.symmetry.number
                    elif hasattr(doc.symmetry, 'space_group_number'):
                        sg_number = doc.symmetry.space_group_number
                
                # Method 2: Check if it's a dict-like object
                if sg_number is None and hasattr(doc, 'symmetry') and doc.symmetry is not None:
                    if isinstance(doc.symmetry, dict):
                        sg_number = doc.symmetry.get('number') or doc.symmetry.get('space_group_number')
                
                # Method 3: Try accessing structure's space group directly
                if sg_number is None and hasattr(doc, 'structure') and doc.structure is not None:
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(doc.structure)
                        sg_number = sga.get_space_group_number()
                    except:
                        pass
                
                print(f"    Space group found: {sg_number}")
                
                if sg_number == target_sg_num:
                    print(f" ✓ Found MP entry: {doc.material_id} (Formula: {doc.formula_pretty}, SG: {sg_number})")
                    # Convert doc to dictionary format for consistency
                    return {
                        'material_id': doc.material_id,
                        'formula_pretty': doc.formula_pretty,
                        'structure': doc.structure,
                        'band_gap': getattr(doc, 'band_gap', None),
                        'formation_energy_per_atom': getattr(doc, 'formation_energy_per_atom', None),
                        'symmetry': {'number': sg_number} if sg_number else None
                    }
            
            print(f"No exact match in MP for Formula: {formula} AND Space Group: {space_group_number}")
            return None
            
        except Exception as e:
            print(f"Error querying Materials Project for {formula} (SG: {space_group_number}): {e}")
            return None

    def generate_and_save_material_record(self, mp_material_data: Dict, save_id: str, 
                                          poscar_file_path: Optional[str] = None, 
                                          format_type: str = 'hybrid') -> Optional[MaterialRecord]:
        """
        Generate complete material record from Materials Project data and save to database.
        
        Args:
            mp_material_data (Dict): The dictionary containing material data from MP.
            save_id (str): A unique ID for saving (e.g., 'MP-mp-xxxxx').
            poscar_file_path (Optional[str]): Path to a local POSCAR file if available.
            format_type (str): Format to save the record ('hybrid', 'hdf5', 'individual').
        """
        
        if not mp_material_data:
            print(f"Cannot generate record for {save_id}: No MP data provided.")
            return None

        try:
            # Extract basic info
            material_id = mp_material_data.get('material_id', save_id)
            formula = mp_material_data.get('formula_pretty', 'Unknown')
            normalized_formula = mp_material_data.get('formula_reduced_abc', formula) # MP has reduced formula
            
            # Extract structure (pymatgen Structure object)
            structure: Structure = mp_material_data.get('structure')
            if not structure:
                print(f"No structure found for MP material {material_id}. Skipping.")
                return None
            
            # Get lattice and atomic info from pymatgen Structure
            lattice_matrix = structure.lattice.matrix
            atomic_positions = np.array(structure.cart_coords) # Cartesian coordinates
            elements = [str(site.specie.symbol) for site in structure]
            atomic_numbers = [site.specie.Z for site in structure]

            # Get space group info from symmetry
            symmetry_data = mp_material_data.get('symmetry', {})
            space_group = symmetry_data.get('symbol', 'Unknown')
            space_group_number = symmetry_data.get('number', 0)
            
            # Extract properties
            band_gap = mp_material_data.get('band_gap')
            formation_energy = mp_material_data.get('formation_energy_per_atom') # MP has per atom
            
            # Topological properties (these typically aren't directly in MP data)
            # You'll need to compute or infer these. For now, using placeholders.
            # The 'TopologicalMaterialAnalyzer' (if it has topological logic)
            # needs to be adapted to process the Structure object.
            
            # Placeholder for topological data and graph data
            # This is where your `TopologicalMaterialAnalyzer` would typically compute these.
            # You need to integrate the logic from `build_dataset_test.py` here.
            # For demonstration, I'll use dummy values.
            
            # HACK: Create a dummy data_block for now. This needs to be replaced
            # by actual calculations from your TopologicalMaterialAnalyzer
            # based on the pymatgen Structure object.
            # The `generate_data_block_with_sg_check` method from your `TopologicalMaterialAnalyzer`
            # needs to be refactored to accept a pymatgen Structure object.
            dummy_crystal_graph = {'nodes': len(elements), 'edges': []}
            dummy_kspace_graph = {'nodes': len(elements), 'edges': []}
            dummy_asph_features = np.random.rand(128) # Placeholder point cloud features
            dummy_topological_class = "Unknown"
            dummy_topological_binary = 0.0 # Default to non-topological if not determined
            dummy_symmetry_operations = symmetry_data # Using MP symmetry data
            dummy_local_database_props = {}
            
            # If your original TopologicalMaterialAnalyzer can work with a pymatgen Structure,
            # you would do something like:
            # material_block = self.generate_data_block_from_structure(structure)
            # Then extract 'crystal_graph', 'kspace_graph', 'asph_features', 'target_label' etc.
            # from material_block.
            
            # For the purpose of this replacement, we'll assume the original
            # `generate_data_block_with_sg_check` can somehow be made to work
            # with MP data, or you have a separate topological analysis module.
            # The most direct path is to convert the MP structure into the format
            # expected by your `TopologicalMaterialAnalyzer`.
            
            # Here's where you'd call your `TopologicalMaterialAnalyzer`
            # This part needs to be adapted based on how your `TopologicalMaterialAnalyzer`
            # processes structural data and generates features/graphs.
            # For this example, I'll call a hypothetical method that takes a Pymatgen Structure.
            
            # Example of how you might integrate with your analyzer:
            # Assuming analyzer has a method `process_structure_for_topological_data`
            # which returns a dict similar to `data_block`
            
            # This part requires you to modify your `TopologicalMaterialAnalyzer`
            # or `build_dataset_test.py` to handle `pymatgen.core.Structure` objects.
            # For now, I'm using placeholder graphs and features.
            
            # To actually integrate:
            # You would likely need to adapt `TopologicalMaterialAnalyzer` to accept
            # a `pymatgen.core.Structure` object instead of a JID, and then
            # extract lattice, positions, etc., and perform graph/feature generation.
            
            # For now, we use placeholders. YOU MUST REPLACE THESE WITH ACTUAL LOGIC.
            crystal_graph_data = dummy_crystal_graph # Replace with actual graph generation
            kspace_graph_data = dummy_kspace_graph # Replace with actual graph generation
            asph_features_data = dummy_asph_features # Replace with actual point cloud generation
            topological_class_data = dummy_topological_class
            topological_binary_data = dummy_topological_binary

            record = MaterialRecord(
                jid=material_id, # Using MP material_id as JID
                formula=formula,
                normalized_formula=normalized_formula,
                space_group=space_group,
                space_group_number=space_group_number,
                
                lattice_matrix=lattice_matrix,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                elements=elements,
                
                crystal_graph=crystal_graph_data,
                kspace_graph=kspace_graph_data,
                
                asph_features=asph_features_data, 
                
                topological_class=topological_class_data,
                topological_binary=topological_binary_data,
                band_gap=band_gap,
                formation_energy=formation_energy,
                
                symmetry_operations=dummy_symmetry_operations,
                local_database_props=dummy_local_database_props,
                processing_timestamp=pd.Timestamp.now().isoformat()
            )
            
            self.database.save_material_record(record, format_type)
            return record

        except Exception as e:
            print(f"Error creating material record for MP entry {save_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function with Materials Project API integration."""
    # --- Configuration ---
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    sqlite_db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"
    output_database_dir = "./multimodal_materials_db_mp"
    
    # You can set this as an environment variable or store it securely.
    mp_api_key = "8O0LxKct7DKVrG2KqE9WhieXWpmsAZuu"
    print (mp_api_key)
    os.makedirs(output_database_dir, exist_ok=True)
    max_materials = 10 

    # --- Load Local CSV Data ---
    print(f"Reading data from local CSV: {csv_path}...")
    materials_df = pd.read_csv(csv_path)

    # --- Initialize Analyzer (now with MP API key) ---
    print("Initializing Enhanced Topological Material Analyzer...")
    analyzer = EnhancedTopologicalMaterialAnalyzer(
        csv_path=csv_path, 
        sqlite_db_path=sqlite_db_path, # This is still used by your original analyzer logic
        output_database_path=output_database_dir,
        mp_api_key=mp_api_key
    )
    
    # --- Main Processing Loop ---
    print(f"\nStarting data generation for up to {max_materials} materials from your CSV (using Materials Project)...")
    
    successful_generations = 0 
    SPACE_GROUP_COLUMN_NAME = 'Space Group' # Make sure this matches your CSV column name

    # Use a set to keep track of processed ICSD IDs to avoid redundant processing
    processed_icsd_ids = set()

    for index, row in tqdm.tqdm(materials_df.head(max_materials).iterrows(), total=max_materials, desc="Processing Materials"):
        formula = row['Formula']
        icsd_id = row['ICSD_ID']
        
        # Skip if already processed (e.g., from a previous run or duplicate in CSV)
        if icsd_id in processed_icsd_ids:
            continue
        
        if pd.isna(row[SPACE_GROUP_COLUMN_NAME]):
            print(f"✗ Skipping formula '{formula}' (ICSD: {icsd_id}): Missing Space Group in CSV.")
            continue
        
        try:
            space_group_string = str(row[SPACE_GROUP_COLUMN_NAME])
            space_group_number_str = space_group_string.split(' ')[0]
            space_group_from_csv = int(space_group_number_str)
        except (ValueError, IndexError):
            print(f"✗ Skipping formula '{formula}' (ICSD: {icsd_id}): Could not parse space group number from '{row[SPACE_GROUP_COLUMN_NAME]}'.")
            continue

        unique_save_id = f"ICSD-{icsd_id}"
        # Check if the material has already been successfully processed and saved
        # You'll need to adjust how you check for existence if you're saving in a new way
        # For 'hybrid' format, check for the metadata JSON file.
        existing_metadata_path = Path(output_database_dir) / 'metadata' / f"{unique_save_id}.json"
        if existing_metadata_path.exists():
            print(f"✓ Skipping {unique_save_id}: Already processed.")
            successful_generations += 1 # Count already processed as successful
            processed_icsd_ids.add(icsd_id)
            continue
        
        # --- NEW: Query Materials Project instead of JARVIS ---
        mp_data = analyzer.get_mp_material_data(formula, space_group_from_csv)
        
        if not mp_data:
            print(f"✗ No Materials Project entry found for Formula: {formula} AND Space Group: {space_group_from_csv}. Skipping.")
            continue
        
        # Now, use the retrieved MP data to generate and save the material record
        # Note: Your `generate_and_save_material_record` needs to be adapted to
        # take the MP data directly, as it currently expects `data_block` which
        # was JARVIS-specific. I've updated its signature above.
        record = analyzer.generate_and_save_material_record(
            mp_material_data=mp_data,
            save_id=unique_save_id,
            # poscar_file_path=analyzer.find_poscar_file(unique_save_id, formula) # Optional: if you still want to use local POSCARs
        )
        
        if record:
            successful_generations += 1
            processed_icsd_ids.add(icsd_id)
            print(f"✓ Successfully processed and saved {unique_save_id} (MP ID: {record.jid})")
            
    # --- Final Report ---
    print(f"\n" + "="*60)
    print(f"DATASET GENERATION COMPLETE")
    print(f"="*60)
    print(f"Total materials in CSV: {len(materials_df)}")
    print(f"Successfully processed and saved: {successful_generations}")
    print(f"Database location: {output_database_dir}")
    print(f"="*60)
    
    # Create master index and ML splits after processing
    # Note: `create_master_index` and `create_ml_datasets` belong to MultiModalMaterialDatabase
    # so we call them via `analyzer.database`.
    if successful_generations > 0:
        print("\nCreating master index and ML datasets...")
        master_df = analyzer.database.create_master_index()
        if not master_df.empty:
            analyzer.database.create_ml_datasets()
        else:
            print("No materials were successfully processed to create a master index or ML datasets.")
    else:
        print("No materials were successfully processed. Skipping master index and ML dataset creation.")

    # Print directory structure
    inspect_database(output_database_dir)

def inspect_database(db_path: str):
    """Helper function to inspect what's in the database"""
    db_path_obj = Path(db_path)
    
    print(f"\nDatabase inspection: {db_path}")
    print("="*50)
    
    if not db_path_obj.exists():
        print("Database directory does not exist!")
        return
    
    # Check each subdirectory
    subdirs = ['structures', 'graphs', 'point_clouds', 'metadata', 'datasets']
    for subdir in subdirs:
        subdir_path = db_path_obj / subdir
        if subdir_path.exists():
            files = list(subdir_path.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"{subdir}: {len(files)} files")
            for f in files[:3]:  # Show first 3 files
                print(f"  - {f.name}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more")
        else:
            print(f"{subdir}: directory not found")
    
    # Check master index
    master_csv = db_path_obj / 'master_index.csv'
    if master_csv.exists():
        df = pd.read_csv(master_csv)
        print(f"Master index: {len(df)} materials")
        if len(df) > 0:
            print(f"  Columns: {list(df.columns)}")
            if 'topological_binary' in df.columns:
                print(f"  Topological materials: {(df['topological_binary'] == 1).sum()}")
    else:
        print("Master index: not found")

if __name__ == "__main__":
    main()