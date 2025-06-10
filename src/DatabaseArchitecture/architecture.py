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

class EnhancedTopologicalMaterialAnalyzer(TopologicalMaterialAnalyzer):
    """Enhanced analyzer that uses the new database structure"""
    
    def __init__(self, csv_path: str, database_path: str):
        super().__init__(csv_path)
        self.database = MultiModalMaterialDatabase(database_path)
    
    def generate_and_save_material_record(self, jid: str, 
                                    format_type: str = 'hybrid') -> Optional[MaterialRecord]:
        """Generate complete material record and save to database"""
    
        data_block = self.generate_data_block_with_sg_check(jid)
        if not data_block:
            return None
        
        try:
            # Extract crystal graph data safely
            crystal_graph = data_block['crystal_graph']
            
            # Get atomic positions from crystal graph
            if hasattr(crystal_graph, 'pos') and crystal_graph.pos is not None:
                atomic_positions = crystal_graph.pos.numpy()
            else:
                print(f"Warning: No atomic positions found for {jid}")
                return None
            
            # Get atomic numbers from crystal graph features
            if hasattr(crystal_graph, 'x') and crystal_graph.x is not None:
                # Assuming first column contains normalized atomic numbers
                atomic_numbers = [int(z * 100) for z in crystal_graph.x[:, 0].numpy()]
            else:
                print(f"Warning: No atomic features found for {jid}")
                return None
            
            # Get lattice matrix from JARVIS properties
            jarvis_props = data_block.get('jarvis_props', {})
            
            # Try to get lattice from JARVIS data
            if 'lattice_abc' in jarvis_props and 'lattice_angles' in jarvis_props:
                # Convert from abc + angles to matrix (you'll need to implement this)
                lattice_matrix = self._abc_angles_to_matrix(
                    jarvis_props['lattice_abc'], 
                    jarvis_props['lattice_angles']
                )
            else:
                # Fallback: create identity matrix scaled by average position
                lattice_matrix = np.eye(3) * 10.0  # Default 10 Angstrom cell
            
            # Get elements list
            elements = data_block.get('elements', [])
            if not elements and 'formula' in data_block:
                # Try to extract elements from formula
                elements = self._extract_elements_from_formula(data_block['formula'])
            
            # Create the record
            record = MaterialRecord(
                jid=data_block['jid'],
                formula=data_block.get('formula', 'Unknown'),
                normalized_formula=data_block.get('normalized_formula', 'Unknown'),
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
            print(f"Error creating material record for {jid}: {str(e)}")
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

    def _extract_elements_from_formula(self, formula: str) -> List[str]:
        """Extract element symbols from chemical formula"""
        import re
        
        # Find all element symbols (capital letter followed by optional lowercase)
        elements = re.findall(r'[A-Z][a-z]?', formula)
        return elements
    
def main():
    """Main execution function with proper database integration."""
    # --- Configuration ---
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    output_dir = "./graph_vector_dataset"  # Keep this for backward compatibility
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/src/db_all"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    max_materials = 10  # Start with a smaller number for testing
    
    # --- Initialize Enhanced Analyzer ---
    print("Initializing Enhanced Topological Material Analyzer...")
    analyzer = EnhancedTopologicalMaterialAnalyzer(csv_path=csv_path, database_path=db_path)
    
    if not analyzer.formula_lookup:
        print("ERROR: No materials loaded from local database. Please check the CSV file.")
        return
    
    # --- Get JIDs to process ---
    jids_to_process = get_jarvis_jids(max_jids=max_materials)
    
    # --- Main Processing Loop ---
    print(f"\nStarting data generation for up to {len(jids_to_process)} materials...")
    print(f"Looking for matches in local database with {len(analyzer.formula_lookup)} materials...")
    print(f"Database will be saved to: {db_path}")
    
    successful_generations = 0
    successful_db_saves = 0
    
    for i, jid in enumerate(tqdm.tqdm(jids_to_process, desc="Processing JIDs")):
        # Check if already processed in database
        metadata_path = Path(db_path) / 'metadata' / f"{jid}.json"
        if metadata_path.exists():
            print(f"Skipping {jid} - already exists in database")
            successful_generations += 1
            successful_db_saves += 1
            continue
        
        # Generate and save material record using the enhanced analyzer
        try:
            record = analyzer.generate_and_save_material_record(jid, format_type='hybrid')
            
            if record:
                successful_generations += 1
                successful_db_saves += 1
                
                # Also save old format for backward compatibility (optional)
                old_output_path = os.path.join(output_dir, f"{jid}.pt")
                if not os.path.exists(old_output_path):
                    # Generate the old data block format
                    material_block = analyzer.generate_data_block_with_sg_check(jid)
                    if material_block:
                        torch.save(material_block, old_output_path)
                        
                        # Save metadata as JSON for easy inspection
                        metadata_path = os.path.join(output_dir, f"{jid}_metadata.json")
                        metadata = {
                            'jid': material_block['jid'],
                            'formula': material_block['formula'],
                            'normalized_formula': material_block['normalized_formula'],
                            'target_label': material_block['target_label'].item(),
                            'num_atoms': material_block['crystal_graph'].x.shape[0],
                            'num_edges': material_block['crystal_graph'].edge_index.shape[1],
                            'jarvis_props': material_block['jarvis_props']
                        }
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                
                print(f"✓ Successfully processed and saved {jid} to database")
            else:
                print(f"✗ Failed to process {jid}")
                
        except Exception as e:
            print(f"✗ Error processing {jid}: {str(e)}")
            continue
        
        # Print progress every 5 materials
        if (i + 1) % 5 == 0:
            print(f"\nProgress: {i+1}/{len(jids_to_process)} processed")
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
            split_paths = analyzer.database.create_ml_datasets()
            print("✓ ML dataset splits created:")
            for split_name, path in split_paths.items():
                print(f"  {split_name}: {path}")
                
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
    print(f"Legacy dataset location: {output_dir}")
    print(f"="*60)
    
    # Print directory structure
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        print(f"\nDatabase structure:")
        for item in sorted(db_path_obj.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(db_path_obj)
                print(f"  {rel_path}")


def inspect_database(db_path: str):
    """Helper function to inspect what's in the database"""
    db_path_obj = Path(db_path)
    
    print(f"Database inspection: {db_path}")
    print("="*50)
    
    if not db_path_obj.exists():
        print("Database directory does not exist!")
        return
    
    # Check each subdirectory
    subdirs = ['structures', 'graphs', 'metadata', 'datasets']
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
    master_parquet = db_path_obj / 'master_index.parquet'
    
    if master_csv.exists():
        df = pd.read_csv(master_csv)
        print(f"Master index: {len(df)} materials")
        if len(df) > 0:
            print(f"  Columns: {list(df.columns)}")
            print(f"  Topological materials: {(df['topological_binary'] == 1).sum()}")
    else:
        print("Master index: not found")


# Add this at the end of main() if you want to inspect right after
def main_with_inspection():
    """Main function with database inspection"""
    main()
    
    # Inspect the database after processing
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/src/db_all"
    inspect_database(db_path)

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