# todo implement point cloud 
# read poscar data from local file
# figure out 

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

def get_jarvis_jids(max_jids: int = 1000) -> List[str]:
    """Get a list of JIDs from JARVIS database."""
    print(f"Fetching JID list from JARVIS (max: {max_jids})...")
    
    # You can get JIDs in several ways:
    # 1. From a specific range
    jid_list = []
    for i in range(1, max_jids + 1):
        jid_list.append(f"JVASP-{i}")
    
    print(f"Generated {len(jid_list)} JIDs")
    return jid_list


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
            f.create_dataset('asph_features', data=record.asph_features)
            
            # Store strings as fixed-length or variable-length
            f.create_dataset('elements', data=[s.encode('utf-8') for s in record.elements])
        
        # 2. Graph data as PyTorch files (preserves tensor structure)
        graph_dir = self.base_path / 'graphs' / record.jid
        graph_dir.mkdir(exist_ok=True)
        
        torch.save(record.crystal_graph, graph_dir / 'crystal_graph.pt')
        torch.save(record.kspace_graph, graph_dir / 'kspace_graph.pt')
        
        # 3. Metadata as JSON (human-readable)
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
            'file_locations': {
                'structure': str(structure_path.relative_to(self.base_path)),
                'crystal_graph': str((graph_dir / 'crystal_graph.pt').relative_to(self.base_path)),
                'kspace_graph': str((graph_dir / 'kspace_graph.pt').relative_to(self.base_path))
            }
        }
        
        metadata_path = self.base_path / 'metadata' / f"{record.jid}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
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
        
        # Atomic positions
        for pos in record.atomic_positions:
            lines.append(f"  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        
        return "\n".join(lines)
    
    def _save_graph_as_graphml(self, graph_data: Dict, filepath: Path):
        """Save graph in GraphML format for interoperability"""
        # This is a simplified version - you'd need to implement
        # conversion from your graph format to NetworkX/GraphML
        pass
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
import re

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


class MultiModalMaterialDatabase:
    """Enhanced database structure for multi-modal materials data"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create organized directory structure"""
        directories = [
            'raw_data',           # Original JARVIS + local data
            'structures',         # POSCAR files and structure data
            'graphs',            # Graph representations
            'point_clouds',      # Topological/ASPH features (separate files)
            'metadata',          # JSON metadata and indices
            'datasets',          # Processed datasets for ML
            'analysis'           # Analysis results and visualizations
        ]
        
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_material_record(self, record: 'MaterialRecord', 
                           format_type: str = 'hybrid') -> None:
        """Save material record in specified format"""
        
        if format_type == 'hybrid':
            self._save_hybrid_format(record)
        elif format_type == 'hdf5':
            self._save_hdf5_format(record)
        elif format_type == 'individual':
            self._save_individual_format(record)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _save_hybrid_format(self, record: 'MaterialRecord'):
        """Hybrid format: HDF5 for arrays, JSON for metadata, separate files for graphs and point clouds"""
        
        # 1. Structure data in HDF5 (efficient for arrays)
        structure_path = self.base_path / 'structures' / f"{record.jid}.h5"
        with h5py.File(structure_path, 'w') as f:
            f.create_dataset('lattice_matrix', data=record.lattice_matrix)
            f.create_dataset('atomic_positions', data=record.atomic_positions)
            f.create_dataset('atomic_numbers', data=record.atomic_numbers)
            
            # Store strings as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('elements', data=record.elements, dtype=dt)
        
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
    
    def _create_poscar_string(self, record: 'MaterialRecord') -> str:
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
    
    def create_master_index(self) -> pd.DataFrame:
        """Create a master index/catalog of all materials"""
        
        records = []
        metadata_dir = self.base_path / 'metadata'
        
        for json_file in metadata_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                records.append(metadata)
        
        if not records:
            print("No metadata files found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Save as both CSV (human-readable) and parquet (efficient)
        df.to_csv(self.base_path / 'master_index.csv', index=False)
        df.to_parquet(self.base_path / 'master_index.parquet', index=False)
        
        return df


class EnhancedTopologicalMaterialAnalyzer(TopologicalMaterialAnalyzer):
    """Enhanced analyzer that uses the new database structure and reads POSCAR files"""
    
    def __init__(self, csv_path: str, database_path: str):
        super().__init__(csv_path)
        self.database = MultiModalMaterialDatabase(database_path)
        
    def find_poscar_file(self, jid: str, formula: str) -> Optional[str]:
        """Find POSCAR file for given JID or formula"""
        if not self.poscar_dir or not self.poscar_dir.exists():
            return None
        
        # Search patterns for POSCAR files
        search_patterns = [
            f"{jid}*POSCAR*",
            f"*{jid}*POSCAR*",
            f"*{formula}*POSCAR*",
            f"POSCAR*{jid}*",
            f"POSCAR*{formula}*"
        ]
        
        for pattern in search_patterns:
            matches = list(self.poscar_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        # Also search in subdirectories
        for pattern in search_patterns:
            matches = list(self.poscar_dir.rglob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def generate_and_save_material_record(self, identifier: str, save_id: str, poscar_file_path : Optional[str] = None, 
                                        format_type: str = 'hybrid') -> Optional['MaterialRecord']:
        """Generate complete material record and save to database"""
        
        data_block = self.generate_data_block_with_sg_check(identifier)
        if not data_block:
            # The identifier might have been a JID, so let's use the save_id for the message
            print(f"Could not generate data block for {save_id} using identifier '{identifier}'")
            return None
        
        try:
            poscar_data = None
            if poscar_file_path and Path(poscar_file_path).exists():
                poscar_data = POSCARReader.read_poscar(poscar_file_path)
                print(f"✓ Using provided POSCAR file for {save_id}: {poscar_file_path}")
            
            # Extract structure data (prefer POSCAR if available)
            if poscar_data:
                lattice_matrix = poscar_data['lattice_matrix']
                atomic_positions = poscar_data['atomic_positions']
                atomic_numbers = poscar_data['atomic_numbers']
                elements = poscar_data['elements']
                formula = poscar_data['formula']
            else:
                # Fallback to crystal graph data
                crystal_graph = data_block['crystal_graph']
                
                if hasattr(crystal_graph, 'pos') and crystal_graph.pos is not None:
                    atomic_positions = crystal_graph.pos.numpy()
                else:
                    print(f"Warning: No atomic positions found for {save_id}")
                    return None
                
                if hasattr(crystal_graph, 'x') and crystal_graph.x is not None:
                    # Assuming first column contains normalized atomic numbers
                    atomic_numbers = [int(z * 100) for z in crystal_graph.x[:, 0].numpy()]
                else:
                    print(f"Warning: No atomic features found for {save_id}")
                    return None
                
                # Get lattice matrix from JARVIS properties or create default
                jarvis_props = data_block.get('jarvis_props', {})
                if 'lattice_abc' in jarvis_props and 'lattice_angles' in jarvis_props:
                    lattice_matrix = self._abc_angles_to_matrix(
                        jarvis_props['lattice_abc'], 
                        jarvis_props['lattice_angles']
                    )
                else:
                    lattice_matrix = np.eye(3) * 10.0  # Default 10 Angstrom cell
                
                elements = data_block.get('elements', [])
                formula = data_block.get('formula', 'Unknown')
            
            # Get JARVIS properties
            jarvis_props = data_block.get('jarvis_props', {})
            
            # Create the record
            record = MaterialRecord(
                jid= save_id,
                formula=formula,
                normalized_formula=data_block.get('normalized_formula', formula),
                space_group=jarvis_props.get('space_group', 'Unknown'),
                space_group_number=jarvis_props.get('space_group_number', 0),
                
                # Structure data
                lattice_matrix=lattice_matrix,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                elements=elements,
                
                # Graph data
                crystal_graph=data_block['crystal_graph'],
                kspace_graph=data_block['kspace_graph'],
                
                # Point cloud
                asph_features=data_block['asph_features'].numpy(),
                
                # Labels
                topological_class=data_block.get('topological_class', 'Unknown'),
                topological_binary=float(data_block['target_label'].item()),
                band_gap=jarvis_props.get('band_gap'),
                formation_energy=jarvis_props.get('formation_energy'),
                
                # Metadata
                symmetry_operations=data_block.get('symmetry_ops', {}),
                local_database_props=data_block.get('local_topo_data', {}),
                processing_timestamp=pd.Timestamp.now().isoformat()
            )
            
            # Save to database
            self.database.save_material_record(record, format_type)
            return record
            
        except Exception as e:
            print(f"Error creating material record for {save_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _abc_angles_to_matrix(self, abc: List[float], angles: List[float]) -> np.ndarray:
        """Convert lattice parameters (a,b,c,alpha,beta,gamma) to matrix"""
        a, b, c = abc
        alpha, beta, gamma = np.radians(angles)  # Convert to radians
        
        # Standard conversion from lattice parameters to matrix
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        # Calculate lattice vectors
        ax = a
        ay = 0
        az = 0
        
        bx = b * cos_gamma
        by = b * sin_gamma
        bz = 0
        
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * np.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)
        
        return np.array([
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz]
        ])


def main():
    """Main execution function with proper database integration."""
    # --- Configuration ---
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    output_dir = "./graph_vector_dataset"  # Keep this for backward compatibility
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/src/db_all"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    max_materials = 10  # Start with a smaller number for testing

    print(f"Reading data from {csv_path}...")
    try:
        materials_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at path: {csv_path}")
        return
    
    poscarlinks = materials_df['POSCAR_link']

    
    # --- Initialize Enhanced Analyzer ---
    print("Initializing Enhanced Topological Material Analyzer...")
    analyzer = EnhancedTopologicalMaterialAnalyzer(
        csv_path=csv_path, 
        database_path=db_path,
    )
    
    if not analyzer.formula_lookup:
        print("ERROR: No materials loaded from local database. Please check the CSV file.")
        return
    
    # --- Main Processing Loop ---
    print(f"\nStarting data generation for up to {max_materials} materials from your CSV...")
    print(f"Looking for matches in local database with {len(analyzer.formula_lookup)} materials...")
    print(f"Database will be saved to: {db_path}")
    # if analyzer.poscar_dir:
    #     print(f"POSCAR files directory: {analyzer.poscar_dir}")
    
    successful_generations = 0
    successful_db_saves = 0
    
    for index, row in tqdm.tqdm(materials_df.head(max_materials).iterrows(), total=max_materials, desc="Processing Materials"):
        
        # 1. Get data directly from the current row
        formula = row['Formula']
        icsd_id = row['ICSD_ID']
        poscar_path = row['POSCAR_link']
        
        # 2. Create a unique ID for saving files (e.g., "ICSD-3")
        unique_save_id = f"ICSD-{icsd_id}"

        # 3. Check if this material has already been processed
        metadata_path = Path(db_path) / 'metadata' / f"{unique_save_id}.json"
        if metadata_path.exists():
            print(f"Skipping {unique_save_id} - already exists in database")
            continue
        
        try:
            # 4. Call the modified method
            record = analyzer.generate_and_save_material_record(
                identifier=formula,         # Use the formula to find data
                save_id=unique_save_id,     # Use the ICSD ID for saving
                poscar_file_path=poscar_path
            )
            
            if record:
                print(f"✓ Successfully processed and saved {unique_save_id}")
                
        except Exception as e:
            print(f"✗ Error processing {unique_save_id} ({formula}): {str(e)}")
            continue
        
        # Print progress every 5 materials
        if (index + 1) % 5 == 0:
            print(f"\nProgress: {index+1}/{max_materials} processed")
            print(f"  Database saves: {successful_db_saves}")
            print(f"  Matches found: {analyzer.matched_count}")
        
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    # --- Create Master Index and ML Datasets ---
    if successful_db_saves > 0:
        print("\n" + "="*60)
        print("CREATING MASTER INDEX AND ML DATASETS...")
        print("="*60)
        
        try:
            # Create master index
            master_df = analyzer.database.create_master_index()
            print(f"✓ Master index created with {len(master_df)} materials")
            
            # Create ML datasets
            if len(master_df) > 5:  # Only create splits if we have enough data
                split_paths = analyzer.database.create_ml_datasets()
                print("✓ ML dataset splits created:")
                for split_name, path in split_paths.items():
                    print(f"  {split_name}: {path}")
            else:
                print("⚠ Not enough materials for train/val/test splits")
                
        except Exception as e:
            print(f"✗ Error creating master index/datasets: {str(e)}")
    
    # --- Final Report ---
    print(f"\n" + "="*60)
    print(f"DATASET GENERATION COMPLETE")
    print(f"="*60)
    print(f"Total JIDs processed: {analyzer.processed_count}")
    print(f"Matches found in local DB: {analyzer.matched_count}")
    print(f"Successful data blocks: {successful_generations}")
    print(f"Successful database saves: {successful_db_saves}")
    print(f"Success rate: {successful_generations/analyzer.processed_count*100:.1f}% (processing)")
    print(f"Database save rate: {successful_db_saves/analyzer.processed_count*100:.1f}% (database)")
    print(f"Database location: {db_path}")
    print(f"="*60)
    
    # Print directory structure
    inspect_database(db_path)


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

# Recommended database structure
"""
materials_database/
├── raw_data/                    # Original data sources
│   ├── jarvis_cache/
│   └── local_database.csv
├── structures/                  # Crystal structures
│   ├── JVASP-1.h5              # Lattice, positions, elements
│   └── JVASP-2.h5
├── graphs/                      # Graph representations
│   ├── JVASP-1/
│   │   ├── crystal_graph.pt    # Real-space graph
│   │   └── kspace_graph.pt     # K-space graph
│   └── JVASP-2/
├── point_clouds/               # Topological features
│   ├── JVASP-1_asph.npy
│   └── JVASP-2_asph.npy
├── metadata/                   # Searchable metadata
│   ├── JVASP-1.json
│   └── JVASP-2.json
├── datasets/                   # ML-ready datasets
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── master_index.csv           # Complete catalog
├── master_index.parquet       # Efficient version
└── analysis/                  # Analysis results
    ├── statistics.json
    └── visualizations/
"""