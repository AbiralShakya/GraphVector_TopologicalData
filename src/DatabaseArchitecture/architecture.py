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
from dataset_builder_backend import TopologicalMaterialAnalyzer 
import time
import tqdm

from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from kspace_graph_generation import KSpacePhysicsGraphBuilder
@dataclass
class MaterialRecord:
    """Enhanced material record with vectorized features and expanded MP data"""
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
    kspace_graph_path: str

    asph_features: np.ndarray  
    vectorized_features: Dict 
    
    topological_class: str  # 'TI', 'SM', 'NI'
    topological_binary: float  # 0 or 1
    
    band_gap: Optional[float]
    formation_energy: Optional[float]
    energy_above_hull: Optional[float]
    density: Optional[float]
    volume: Optional[float]
    nsites: Optional[int]
    total_magnetization: Optional[float]
    magnetic_type: Optional[str]
    theoretical: Optional[bool]
    stability: Optional[Dict]
    electronic_structure: Optional[Dict]
    mechanical_properties: Optional[Dict]
    
    symmetry_operations: Dict
    local_database_props: Dict
    processing_timestamp: str

    kspace_physics_features: Optional[Dict[str, List]] # Dict of lists (numpy array converted)
    kspace_connectivity_matrix: Optional[List[List[int]]] # List of lists (numpy array converted)


class PointCloudVectorizer:
    """Task 1: Vectorize point cloud features for compatibility with non-GNN models"""
    
    @staticmethod
    def vectorize_point_cloud(point_cloud: np.ndarray, methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Convert point cloud to multiple vectorized representations
        
        Args:
            point_cloud: Original point cloud features (N x D)
            methods: List of vectorization methods to apply
        
        Returns:
            Dictionary of vectorized features
        """
        if methods is None:
            methods = ['statistical', 'histogram', 'moments', 'pca', 'aggregated']
        
        vectorized = {}
        
        if 'statistical' in methods:
            vectorized['statistical'] = PointCloudVectorizer._statistical_features(point_cloud)
        
        if 'histogram' in methods:
            vectorized['histogram'] = PointCloudVectorizer._histogram_features(point_cloud)
        
        if 'moments' in methods:
            vectorized['moments'] = PointCloudVectorizer._moment_features(point_cloud)
        
        if 'pca' in methods:
            vectorized['pca'] = PointCloudVectorizer._pca_features(point_cloud)
        
        if 'aggregated' in methods:
            vectorized['aggregated'] = PointCloudVectorizer._aggregated_features(point_cloud)
        
        if 'bag_of_words' in methods:
            vectorized['bag_of_words'] = PointCloudVectorizer._bag_of_words_features(point_cloud)
        
        return vectorized
    

    @staticmethod
    def _statistical_features(point_cloud: np.ndarray) -> np.ndarray:
        """Extract statistical features: mean, std, min, max, median, skew, kurtosis"""
        from scipy import stats
        
        features = []
        # Per-dimension statistics
        features.extend(np.mean(point_cloud, axis=0))  # Mean per dimension
        features.extend(np.std(point_cloud, axis=0))   # Std per dimension
        features.extend(np.min(point_cloud, axis=0))   # Min per dimension
        features.extend(np.max(point_cloud, axis=0))   # Max per dimension
        features.extend(np.median(point_cloud, axis=0)) # Median per dimension
        
        # Global statistics
        features.append(np.mean(point_cloud))  # Overall mean
        features.append(np.std(point_cloud))   # Overall std
        features.append(np.min(point_cloud))   # Overall min
        features.append(np.max(point_cloud))   # Overall max
        
        # Higher-order moments per dimension
        for dim in range(point_cloud.shape[1]):
            features.append(stats.skew(point_cloud[:, dim]))     # Skewness
            features.append(stats.kurtosis(point_cloud[:, dim])) # Kurtosis
        
        return np.array(features)


    @staticmethod
    def _histogram_features(point_cloud: np.ndarray, bins: int = 10) -> np.ndarray:
        """Convert to histogram representation"""
        features = []
        
        # Per-dimension histograms
        for dim in range(point_cloud.shape[1]):
            hist, _ = np.histogram(point_cloud[:, dim], bins=bins, density=True)
            features.extend(hist)
        
        # Overall histogram (flattened point cloud)
        hist, _ = np.histogram(point_cloud.flatten(), bins=bins, density=True)
        features.extend(hist)
        
        return np.array(features)
    
    @staticmethod
    def _moment_features(point_cloud: np.ndarray, max_moment: int = 4) -> np.ndarray:
        """Extract statistical moments up to specified order"""
        from scipy import stats
        
        features = []
        
        for moment in range(1, max_moment + 1):
            # Per-dimension moments
            for dim in range(point_cloud.shape[1]):
                features.append(stats.moment(point_cloud[:, dim], moment=moment))
        
        return np.array(features)
    
    @staticmethod
    def _pca_features(point_cloud: np.ndarray, n_components: int = 10) -> np.ndarray:
        """Extract PCA-based features"""
        from sklearn.decomposition import PCA
        
        # Ensure we don't request more components than available
        n_components = min(n_components, min(point_cloud.shape) - 1)
        
        if n_components <= 0:
            return np.array([])
        
        pca = PCA(n_components=n_components)
        
        try:
            pca_transformed = pca.fit_transform(point_cloud)
            
            features = []
            # Principal component statistics
            features.extend(np.mean(pca_transformed, axis=0))
            features.extend(np.std(pca_transformed, axis=0))
            
            # Explained variance ratios
            features.extend(pca.explained_variance_ratio_)
            
            return np.array(features)
        except:
            # Return zeros if PCA fails
            return np.zeros(n_components * 3)
    
    @staticmethod
    def _aggregated_features(point_cloud: np.ndarray) -> np.ndarray:
        """Aggregated features using different pooling operations"""
        features = []
        
        # Different pooling operations
        features.extend(np.mean(point_cloud, axis=0))    # Mean pooling
        features.extend(np.max(point_cloud, axis=0))     # Max pooling
        features.extend(np.min(point_cloud, axis=0))     # Min pooling
        features.extend(np.median(point_cloud, axis=0))  # Median pooling
        
        # L2 norm per dimension
        features.extend(np.linalg.norm(point_cloud, axis=0))
        
        # Range per dimension
        features.extend(np.ptp(point_cloud, axis=0))
        
        return np.array(features)
    
    @staticmethod
    def _bag_of_words_features(point_cloud: np.ndarray, n_clusters: int = 50) -> np.ndarray:
        """Bag of words representation using k-means clustering"""
        from sklearn.cluster import KMeans
        
        try:
            # Fit k-means to create "visual words"
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(point_cloud)
            
            # Create histogram of cluster assignments
            hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters-1))
            
            # Normalize to get frequencies
            hist = hist.astype(float) / len(labels)
            
            return hist
        except:
            # Return zeros if clustering fails
            return np.zeros(n_clusters)

class SpaceGroupManager:
    """Task 3: Manage shared space group k-space graphs to reduce redundancy"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.space_group_graphs_dir = self.base_path / 'space_group_graphs'
        self.space_group_graphs_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded space group graphs
        self._cached_graphs = {}
    
    def get_space_group_graph_path(self, space_group_number: int) -> Path:
        """Get the file path for a space group k-space graph"""
        return self.space_group_graphs_dir / f'sg_{space_group_number}_kspace.pt'
    
    def save_space_group_graph(self, space_group_number: int, kspace_graph: Dict) -> str:
        """
        Save k-space graph for a space group (only if it doesn't exist)
        Returns reference string for the graph
        """
        graph_path = self.get_space_group_graph_path(space_group_number)
        
        if not graph_path.exists():
            torch.save(kspace_graph, graph_path)
            print(f"Saved k-space graph for space group {space_group_number}")
        
        # Return relative path as reference
        return str(graph_path.relative_to(self.base_path))
    
    def load_space_group_graph(self, space_group_number: int) -> Optional[Dict]:
        """Load k-space graph for a space group with caching"""
        
        # Check cache first
        if space_group_number in self._cached_graphs:
            return self._cached_graphs[space_group_number]
        
        graph_path = self.get_space_group_graph_path(space_group_number)
        
        if graph_path.exists():
            try:
                graph = torch.load(graph_path)
                self._cached_graphs[space_group_number] = graph
                return graph
            except Exception as e:
                print(f"Error loading space group graph {space_group_number}: {e}")
                return None
        else:
            print(f"Space group graph {space_group_number} not found")
            return None
    
    def get_graph_reference(self, space_group_number: int) -> str:
        """Get reference string for space group graph"""
        graph_path = self.get_space_group_graph_path(space_group_number)
        return str(graph_path.relative_to(self.base_path))
    
    def create_space_group_index(self) -> pd.DataFrame:
        """Create an index of all available space group graphs"""
        space_group_files = list(self.space_group_graphs_dir.glob('sg_*_kspace.pt'))
        
        records = []
        for file in space_group_files:
            # Extract space group number from filename
            sg_num = int(file.stem.split('_')[1])
            
            records.append({
                'space_group_number': sg_num,
                'file_path': str(file.relative_to(self.base_path)),
                'file_size_mb': file.stat().st_size / (1024 * 1024),
                'exists': True
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('space_group_number')
        
        # Save index
        index_path = self.space_group_graphs_dir / 'space_group_index.csv'
        df.to_csv(index_path, index=False)
        
        return df
    

class MultiModalMaterialDatabase:
    """Enhanced database with vectorized features and space group optimization"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.setup_directory_structure()
        self.vectorizer = PointCloudVectorizer()
        self.space_group_manager = SpaceGroupManager(base_path)

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
        
        df.to_csv(self.base_path / 'master_index.csv', index=False)
        dict_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, dict)).any()]
        for col in dict_cols:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

        df.to_parquet(self.base_path / 'master_index.parquet', index=False)
        return df
    
    def setup_directory_structure(self):
        """Create organized directory structure with new additions"""
        directories = [
            'raw_data',           # Original data
            'structures',         # POSCAR files and structure data
            'graphs',            # Material-specific crystal graphs
            'space_group_graphs', # Shared k-space graphs by space group
            'point_clouds',      # Original topological/ASPH features
            'vectorized_features', # Vectorized representations
            'metadata',          # JSON metadata and indices
            'datasets',          # Processed datasets for ML
            'analysis'           # Analysis results and visualizations
        ]
        
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_material_record(self, record: MaterialRecord, 
                           format_type: str = 'hybrid') -> None:
        """Enhanced save with vectorized features and space group optimization"""
        
        if format_type == 'hybrid':
            self._save_hybrid_format_enhanced(record)
        elif format_type == 'hdf5':
            self._save_hdf5_format(record)
        elif format_type == 'individual':
            self._save_individual_format_enhanced(record)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _save_hybrid_format_enhanced(self, record: MaterialRecord):
        """Enhanced hybrid format with vectorized features"""
        
        # 1. Structure data in HDF5
        structure_path = self.base_path / 'structures' / f"{record.jid}.h5" 
        with h5py.File(structure_path, 'w') as f:
            f.create_dataset('lattice_matrix', data=record.lattice_matrix)
            f.create_dataset('atomic_positions', data=record.atomic_positions)
            f.create_dataset('atomic_numbers', data=record.atomic_numbers)
            
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('elements', data=np.array(record.elements, dtype=object), dtype=dt)
        
        # 2. Original point cloud data
        point_cloud_path = self.base_path / 'point_clouds' / f"{record.jid}_asph.npy"
        np.save(point_cloud_path, record.asph_features)

        # Task 1: Save vectorized features
        vectorized_dir = self.base_path / 'vectorized_features' / record.jid
        vectorized_dir.mkdir(exist_ok=True)
        
        for method_name, features in record.vectorized_features.items():
            feature_path = vectorized_dir / f"{method_name}.npy"
            np.save(feature_path, features)

        # 3. Crystal graph (material-specific)
        graph_dir = self.base_path / 'graphs' / record.jid
        graph_dir.mkdir(exist_ok=True)
        #torch.save(record.crystal_graph, graph_dir / 'crystal_graph.pt')
        with open(graph_dir / 'crystal_graph.pkl', 'wb') as f:
            pickle.dump(record.crystal_graph, f)
        
        # K-space graph (now directly embedded as a dict of numpy arrays)
        # Create a separate folder for kspace graphs specific to material if you want
        # or just save it alongside the crystal graph if it's always paired.
        # For this setup, it's best to save it next to the crystal graph for that material.
        # with open(graph_dir / 'kspace_graph.pkl', 'wb') as f:
        #     pickle.dump(record.kspace_graph, f)
        
        # Task 3: K-space graph is now just a reference, not saved per material
        # The actual k-space graph is managed by SpaceGroupManager

        # 4. Create POSCAR file
        poscar_path = self.base_path / 'structures' / f"{record.jid}_POSCAR"
        poscar_content = self._create_poscar_string(record)
        with open(poscar_path, 'w') as f:
            f.write(poscar_content)
        
        # 5. Enhanced metadata with new MP properties
        metadata = {
            'jid': record.jid,
            'formula': record.formula,
            'normalized_formula': record.normalized_formula,
            'space_group': record.space_group,
            'space_group_number': record.space_group_number,
            'topological_class': record.topological_class,
            'topological_binary': record.topological_binary,
            
            # Task 2: Expanded MP properties
            'band_gap': record.band_gap,
            'formation_energy': record.formation_energy,
            'energy_above_hull': record.energy_above_hull,
            'density': record.density,
            'volume': record.volume,
            'nsites': record.nsites,
            'total_magnetization': record.total_magnetization,
            'magnetic_type': record.magnetic_type,
            'theoretical': record.theoretical,
            'stability': record.stability,
            'electronic_structure': record.electronic_structure,
            'mechanical_properties': record.mechanical_properties,
            
            'processing_timestamp': record.processing_timestamp,
            'local_database_props': record.local_database_props,
            'num_atoms': len(record.elements),
            
            # File locations
            'file_locations': {
                'structure_hdf5': str(structure_path.relative_to(self.base_path)),
                'poscar': str(poscar_path.relative_to(self.base_path)),
                'point_cloud': str(point_cloud_path.relative_to(self.base_path)),
                'crystal_graph': str((graph_dir / 'crystal_graph.pkl').relative_to(self.base_path)),
                # 'kspace_graph_reference': record.kspace_graph_reference, # REMOVED
                #'kspace_graph': str((graph_dir / 'kspace_graph.pkl').relative_to(self.base_path)), 
                "kspace_graph_shared" : record.kspace_graph_path,
                'vectorized_features_dir': str(vectorized_dir.relative_to(self.base_path))
            },
            
            # Task 1: Vectorized feature info
            'vectorized_methods': list(record.vectorized_features.keys()),
            'vectorized_feature_sizes': {
                method: features.shape[0] if hasattr(features, 'shape') else len(features)
                for method, features in record.vectorized_features.items()
            }
        }
        
        metadata_path = self.base_path / 'metadata' / f"{record.jid}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_hdf5_format(self, record: MaterialRecord):
        """Save in HDF5 format (currently placeholder, copy from _save_hybrid_format_enhanced logic if needed)"""
        # This function was not fully implemented in the provided code, so leaving as a placeholder.
        # If the user wants to use HDF5, this needs to be properly implemented.
        pass

    def _save_individual_format_enhanced(self, record: MaterialRecord):
        """Enhanced individual format with vectorized features"""
        
        jid_dir = self.base_path / 'individual' / record.jid
        jid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. POSCAR file
        poscar_content = self._create_poscar_string(record)
        with open(jid_dir / 'POSCAR', 'w') as f:
            f.write(poscar_content)
        
        # 2. Crystal graph as GraphML
        self._save_graph_as_graphml(record.crystal_graph, jid_dir / 'crystal_graph.graphml')
        
        # Task 3: K-space graph reference instead of full graph
        # This section needs re-evaluation based on the MaterialRecord change
        # For now, I'll save the kspace_graph directly if using this format
        with open(jid_dir / 'kspace_graph.pkl', 'wb') as f:
            pickle.dump(record.kspace_graph, f)
        
        # 3. Original point cloud data as CSV
        asph_df = pd.DataFrame(record.asph_features.reshape(1, -1))
        asph_df.to_csv(jid_dir / 'asph_features.csv', index=False)
        
        # Task 1: Save vectorized features as separate CSVs
        vectorized_dir = jid_dir / 'vectorized_features'
        vectorized_dir.mkdir(exist_ok=True)
        
        for method_name, features in record.vectorized_features.items():
            feature_df = pd.DataFrame(features.reshape(1, -1))
            feature_df.to_csv(vectorized_dir / f'{method_name}.csv', index=False)
        
        # 4. Complete metadata as JSON
        metadata = asdict(record)
        # Convert numpy arrays to lists for JSON serialization
        metadata['lattice_matrix'] = record.lattice_matrix.tolist()
        metadata['atomic_positions'] = record.atomic_positions.tolist()
        metadata['asph_features'] = record.asph_features.tolist()
        metadata['vectorized_features'] = {
            method: features.tolist() if hasattr(features, 'tolist') else features
            for method, features in record.vectorized_features.items()
        }
        
        # Ensure kspace_graph, kspace_physics_features, kspace_connectivity_matrix are JSON serializable
        if 'kspace_graph' in metadata and metadata['kspace_graph'] is not None:
            # Assuming kspace_graph can contain numpy arrays
            # This requires a more robust serialization if it contains complex numpy objects
            # For simplicity, convert all values to lists if they are numpy arrays
            serializable_kspace_graph = {}
            for k, v in metadata['kspace_graph'].items():
                if isinstance(v, np.ndarray):
                    serializable_kspace_graph[k] = v.tolist()
                elif isinstance(v, (list, tuple)) and all(isinstance(i, np.ndarray) for i in v):
                     serializable_kspace_graph[k] = [item.tolist() for item in v]
                else:
                    serializable_kspace_graph[k] = v
            metadata['kspace_graph'] = serializable_kspace_graph
        
        if 'kspace_physics_features' in metadata and metadata['kspace_physics_features'] is not None:
            serializable_kspace_physics = {}
            for k, v in metadata['kspace_physics_features'].items():
                serializable_kspace_physics[k] = v if isinstance(v, list) else v.tolist()
            metadata['kspace_physics_features'] = serializable_kspace_physics
            
        if 'kspace_connectivity_matrix' in metadata and metadata['kspace_connectivity_matrix'] is not None:
            metadata['kspace_connectivity_matrix'] = [row.tolist() if isinstance(row, np.ndarray) else row for row in metadata['kspace_connectivity_matrix']]

        with open(jid_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_vectorized_dataset(self, method: str = 'aggregated') -> pd.DataFrame:
        """Task 1: Create dataset with vectorized features for non-GNN models"""
        
        records = []
        metadata_dir = self.base_path / 'metadata'
        
        for json_file in metadata_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Load the specific vectorized features
            vectorized_dir = self.base_path / metadata['file_locations']['vectorized_features_dir']
            feature_file = vectorized_dir / f"{method}.npy"
            
            if feature_file.exists():
                features = np.load(feature_file)
                
                # Create record with features and metadata
                record = {
                    'jid': metadata['jid'],
                    'formula': metadata['formula'],
                    'space_group_number': metadata['space_group_number'],
                    'topological_binary': metadata['topological_binary'],
                    'band_gap': metadata['band_gap'],
                    'formation_energy': metadata['formation_energy'],
                    'density': metadata.get('density'),
                    'volume': metadata.get('volume')
                }
                
                # Add features as separate columns
                for i, feature_val in enumerate(features):
                    record[f'feature_{i}'] = feature_val
                
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save vectorized dataset
        output_path = self.base_path / 'datasets' / f'vectorized_{method}_dataset.csv'
        df.to_csv(output_path, index=False)
        
        print(f"Created vectorized dataset with {len(df)} materials using {method} method")
        print(f"Feature dimensions: {len([col for col in df.columns if col.startswith('feature_')])}")
        
        return df
    
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
        # Placeholder - implement based on your graph structure
        # This needs a library like networkx to convert your Dict graph to GraphML
        pass

    def create_ml_datasets(self):
        """Placeholder for creating ML-ready datasets, e.g., for GNNs or other models."""
        print("ML dataset creation method needs to be implemented based on your model requirements.")
        print("You can call create_vectorized_dataset for non-GNN models or generate specific GNN data.")
        # Example: self.create_vectorized_dataset(method='combined_features')

class POSCARReader:
    """Utility class to read and parse POSCAR files"""
    
    @staticmethod
    def read_poscar(poscar_path: str) -> Dict:
        """Read POSCAR file and extract structure information"""
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
                atomic_numbers.extend([POSCARReader.element_to_atomic_number(elem)] * count)
            
            # Coordinate mode (line 7)
            coord_mode = lines[7].lower()
            direct_coords = coord_mode.startswith('d')
            
            # Atomic positions (starting from line 8)
            total_atoms = sum(count_line)
            positions = []
            for i in range(8, 8 + total_atoms):
                pos = [float(x) for x in lines[i].split()[:3]]
                positions.append(pos)
            
            positions = np.array(positions)
            
            # Convert direct to cartesian if needed
            if direct_coords:
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

class IntegratedMaterialProcessor:
    """
    Integrated processor that combines Materials Project data with topological analysis.
    Replaces the placeholder logic in generate_and_save_material_record.
    """
    
    def __init__(self, csv_path: str, db_path: str, kspace_graphs_base_dir: str, database_manager):
        """
        Initialize with topological analyzer and database manager.
        
        Args:
            csv_path: Path to topological materials CSV
            db_path: Path to SQLite database with k-space data
            database_manager: Your existing database manager for saving records
        """
        KSPACE_GRAPH_OUTPUT_DIR = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs"
        self.topological_analyzer = TopologicalMaterialAnalyzer(csv_path, db_path, kspace_graphs_base_dir= KSPACE_GRAPH_OUTPUT_DIR)
        self.database = database_manager
        self.mp_rest = MPRester("8O0LxKct7DKVrG2KqE9WhieXWpmsAZuu")

    def generate_and_save_material_record(self, mp_material_data: Dict, save_id: str, 
                                        poscar_file_path: Optional[str] = None, 
                                        format_type: str = 'hybrid') -> Optional[MaterialRecord]:
        """
        Generate complete material record from Materials Project data with topological analysis.
        
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
            normalized_formula = mp_material_data.get('formula_reduced_abc', formula)
            
            # Extract structure (pymatgen Structure object)
            structure: Structure = mp_material_data.get('structure')
            if not structure:
                print(f"No structure found for MP material {material_id}. Skipping.")
                return None
            
            print(f"Processing MP material {material_id} with formula {formula}")
            
            # INTEGRATION POINT: Use TopologicalMaterialAnalyzer for graph/feature generation
            print("Attempting topological analysis...")
            topological_data_block = None
            
            try:
                # Use the new method that accepts pymatgen Structure objects
                topological_data_block = self.topological_analyzer.enhanced_generate_data_block_from_structure(
                    structure=structure,
                    material_id=material_id,
                    formula=formula
                )
            except Exception as e:
                print(f"Topological analysis failed for {material_id}: {e}")
                topological_data_block = None
            
            # Extract MP properties
            lattice_matrix = structure.lattice.matrix
            atomic_positions = np.array(structure.cart_coords)
            elements = [str(site.specie.symbol) for site in structure]
            atomic_numbers = [site.specie.Z for site in structure]

            # Get space group info
            symmetry_data = mp_material_data.get('symmetry', {})
            space_group = symmetry_data.get('symbol', 'Unknown')
            space_group_number = symmetry_data.get('number', 0)
            
            # Extract MP properties
            band_gap = mp_material_data.get('band_gap')
            formation_energy = mp_material_data.get('formation_energy_per_atom')
            energy_above_hull = mp_material_data.get('energy_above_hull', 0.0)
            density = mp_material_data.get('density', structure.density)
            volume = mp_material_data.get('volume', structure.volume)
            nsites = mp_material_data.get('nsites', len(structure))
            total_magnetization = mp_material_data.get('total_magnetization', 0.0)
            magnetic_type = mp_material_data.get('ordering', 'Unknown')
            theoretical = mp_material_data.get('theoretical', True)
            
            # Process topological analysis results or create fallbacks
            if topological_data_block:
                print("✓ Topological analysis successful - using computed features")
                
                # Extract features from topological analysis
                crystal_graph_data = topological_data_block['crystal_graph']
                kspace_graph_data = topological_data_block['kspace_graph'] 
                asph_features_data = topological_data_block['asph_features'] 
                band_rep_features = topological_data_block['band_rep_features'] 
                
                # Extract topological classification
                target_label = topological_data_block['target_label'].item() if isinstance(topological_data_block['target_label'], np.ndarray) else topological_data_block['target_label']
                topological_binary_data = target_label
                
                # Determine topological class from local database entry
                local_topo_data = topological_data_block.get('local_topo_data', {})
                property_str = str(local_topo_data.get('Property', '')).upper()
                
                if 'TI' in property_str:
                    topological_class_data = "Topological Insulator"
                elif 'SM' in property_str:
                    topological_class_data = "Semimetal"
                elif 'WEYL' in property_str:
                    topological_class_data = "Weyl Semimetal"
                elif 'DIRAC' in property_str:
                    topological_class_data = "Dirac Semimetal"
                elif target_label > 0.5:
                    topological_class_data = "Topological"
                else:
                    topological_class_data = "Trivial"
                
                symmetry_operations_data = topological_data_block['symmetry_ops']
                local_database_props = local_topo_data
                local_database_props = local_topo_data or {"_dummy_key": False}

                kspace_physics_features_data = topological_data_block.get('kspace_physics_features')
                kspace_connectivity_matrix_data = topological_data_block.get('kspace_connectivity_matrix')
                
            else:
                print("⚠ Topological analysis failed - using fallback features")
                
                # Create fallback features when topological analysis fails
                crystal_graph_data = self._create_fallback_crystal_graph(structure)
                kspace_graph_data = self._create_fallback_kspace_graph(structure) # This is a Dict fallback
                asph_features_data = self._create_fallback_asph_features(structure)
                band_rep_features = np.zeros(100)  # Placeholder for band_rep_vector
                
                topological_class_data = "Unknown"
                topological_binary_data = 0.0
                symmetry_operations_data = {'rotations': [], 'translations': [], 'num_ops': 0}
                local_database_props = {}

                # Fallback for new kspace fields
                kspace_physics_features_data = {}
                kspace_connectivity_matrix_data = []
            
            # Create comprehensive vectorized features
            basic_properties = np.array([
                band_gap or 0.0, 
                formation_energy or 0.0, 
                density, 
                volume
            ])
            
            structural_properties = np.array([
                space_group_number, 
                nsites, 
                total_magnetization,
                energy_above_hull
            ])
            
            # Combine all features
            combined_features = np.concatenate([
                asph_features_data,
                band_rep_features, # Use the numpy array
                basic_properties,
                structural_properties
            ])
            
            vectorized_features = {
                'asph_features': asph_features_data,
                'band_rep_features': band_rep_features,
                'basic_properties': basic_properties,
                'structural_properties': structural_properties,
                'combined_features': combined_features
            }
            
            # Create stability and electronic structure data
            stability_data = {
                'energy_above_hull': energy_above_hull,
                'formation_energy_per_atom': formation_energy,
                'is_stable': energy_above_hull <= 0.025
            }
            
            electronic_structure_data = {
                'band_gap': band_gap,
                'is_metal': band_gap == 0.0 if band_gap is not None else None,
                'is_gap_direct': mp_material_data.get('is_gap_direct'),
                'cbm': mp_material_data.get('cbm'),
                'vbm': mp_material_data.get('vbm')
            }
            
            mechanical_properties_data = {
                'bulk_modulus': mp_material_data.get('bulk_modulus'),
                'shear_modulus': mp_material_data.get('shear_modulus'),
            }

            shared_graph_rel_path = self.topological_analyzer.shared_graph_rel_path(space_group_number)

            record = MaterialRecord(
                jid=material_id,
                formula=formula,
                normalized_formula=normalized_formula,
                space_group=space_group,
                space_group_number=space_group_number,
                
                lattice_matrix=lattice_matrix,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                elements=elements,
                
                crystal_graph=crystal_graph_data,
                kspace_graph_path= shared_graph_rel_path,
                
                asph_features=asph_features_data,
                vectorized_features=vectorized_features,
                
                topological_class=topological_class_data,
                topological_binary=topological_binary_data,
                band_gap=band_gap,
                formation_energy=formation_energy,
                
                energy_above_hull=energy_above_hull,
                density=density,
                volume=volume,
                nsites=nsites,
                total_magnetization=total_magnetization,
                magnetic_type=magnetic_type,
                theoretical=theoretical,
                stability=stability_data,
                electronic_structure=electronic_structure_data,
                mechanical_properties=mechanical_properties_data,
                
                symmetry_operations=symmetry_operations_data,
                local_database_props=local_database_props,
                processing_timestamp=pd.Timestamp.now().isoformat(),

                kspace_physics_features=kspace_physics_features_data,
                kspace_connectivity_matrix=kspace_connectivity_matrix_data
            )
            
            self.database.save_material_record(record, format_type)
            print(f"✓ Successfully saved material record for {material_id}")
            
            return record

        except Exception as e:
            print(f"Error creating material record for MP entry {save_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_fallback_crystal_graph(self, structure: Structure) -> Dict:
        """Create a basic crystal graph when topological analysis fails."""
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            
            vnn = VoronoiNN(cutoff=8.0, allow_pathological=True)
            
            nodes = []
            edges = []
            
            for i, site in enumerate(structure):
                # Node features: atomic number, coordination
                atomic_number = site.specie.Z
                nodes.append({
                    'id': i,
                    'atomic_number': atomic_number,
                    'position': site.coords.tolist()
                })
                
                # Get neighbors
                try:
                    neighbors = vnn.get_nn_info(structure, i)
                    for neighbor_info in neighbors:
                        j = neighbor_info['site_index']
                        if i != j:
                            edges.append({
                                'source': i,
                                'target': j,
                                'weight': neighbor_info['weight']
                            })
                except:
                    continue
            
            return {
                'nodes': nodes,
                'edges': edges,
                'graph_type': 'fallback_crystal_graph'
            }
            
        except Exception as e:
            print(f"Error creating fallback crystal graph: {e}")
            return {
                'nodes': [{'id': i, 'atomic_number': site.specie.Z} 
                         for i, site in enumerate(structure)],
                'edges': [],
                'graph_type': 'minimal_fallback'
            }
    
    def _create_fallback_kspace_graph(self, structure: Structure) -> Dict:
        """Create a basic k-space graph when topological analysis fails."""
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            analyzer = SpacegroupAnalyzer(structure)
            sg_number = analyzer.get_space_group_number()
            
            # Create basic k-space representation
            high_symmetry_points = ['Γ', 'X', 'M', 'R']  # Generic points
            
            nodes = [{'id': i, 'k_point': point, 'coordinates': [0, 0, 0]} 
                    for i, point in enumerate(high_symmetry_points)]
            
            edges = [{'source': i, 'target': i+1} 
                    for i in range(len(high_symmetry_points)-1)]
            
            return {
                'nodes': nodes,
                'edges': edges,
                'space_group': sg_number,
                'graph_type': 'fallback_kspace_graph'
            }
            
        except Exception as e:
            print(f"Error creating fallback k-space graph: {e}")
            return {
                'nodes': [],
                'edges': [],
                'graph_type': 'minimal_kspace_fallback'
            }
    
    def _create_fallback_asph_features(self, structure: Structure, n_features: int = 128) -> np.ndarray:
        """Create basic structural features when ASPH computation fails."""
        try:
            # Basic structural descriptors
            features = []
            
            # Lattice parameters
            a, b, c = structure.lattice.abc
            alpha, beta, gamma = structure.lattice.angles
            
            features.extend([a, b, c, alpha, beta, gamma])
            features.append(structure.volume)
            features.append(structure.density)
            
            # Composition features
            composition = structure.composition
            features.append(len(composition))  # Number of unique elements
            features.append(sum(composition.values()))  # Total atoms
            
            # Element statistics
            atomic_numbers = [site.specie.Z for site in structure]
            features.extend([
                np.mean(atomic_numbers),
                np.std(atomic_numbers),
                np.min(atomic_numbers),
                np.max(atomic_numbers)
            ])
            
            # Pad or truncate to desired length
            if len(features) < n_features:
                features.extend([0.0] * (n_features - len(features)))
            else:
                features = features[:n_features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating fallback ASPH features: {e}")
            return np.random.rand(n_features).astype(np.float32)
    
    def process_mp_materials_batch(self, mp_materials_list: List[Dict], 
                                 save_prefix: str = "MP") -> List[MaterialRecord]:
        """
        Process a batch of Materials Project materials.
        
        Args:
            mp_materials_list: List of MP material dictionaries
            save_prefix: Prefix for save IDs
            
        Returns:
            List of successfully created MaterialRecord objects
        """
        successful_records = []
        
        for i, mp_data in enumerate(mp_materials_list):
            try:
                material_id = mp_data.get('material_id', f'unknown-{i}')
                save_id = f"{save_prefix}-{material_id}"
                
                print(f"\n--- Processing {i+1}/{len(mp_materials_list)}: {material_id} ---")
                
                record = self.generate_and_save_material_record(
                    mp_material_data=mp_data,
                    save_id=save_id,
                    format_type='hybrid'
                )
                
                if record:
                    successful_records.append(record)
                    print(f"✓ Successfully processed {material_id}")
                else:
                    print(f"✗ Failed to process {material_id}")
                    
            except Exception as e:
                print(f"✗ Error processing material {i}: {e}")
                continue
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Successfully processed: {len(successful_records)}/{len(mp_materials_list)} materials")
        print(f"Topological matches: {self.topological_analyzer.matched_count}")
        
        return successful_records
    
    def get_mp_material_data(self, formula: str, space_group_number: int) -> Optional[Dict]:
        """
        Query Materials Project for material data matching formula and space group.
        Returns the first matching entry or None.
        """
        print(f"Searching Materials Project for Formula: {formula}, Space Group: {space_group_number}")
        try:
            # FIX: Use MPRester.materials.summary.search instead of MPRester.summary.search
            docs = self.mp_rest.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure", "band_gap",
                    "formation_energy_per_atom", "symmetry", "energy_above_hull",
                    "density", "volume", "nsites", "total_magnetization", "ordering",
                    "is_gap_direct", "cbm", "vbm", "bulk_modulus", "shear_modulus",
                   ]
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
                        'energy_above_hull': getattr(doc, 'energy_above_hull', None),
                        'density': getattr(doc, 'density', None),
                        'volume': getattr(doc, 'volume', None),
                        'nsites': getattr(doc, 'nsites', None),
                        'total_magnetization': getattr(doc, 'total_magnetization', None),
                        'ordering': getattr(doc, 'ordering', None), # magnetic_type
                        'theoretical': getattr(doc, 'theoretical', True), # Assuming theoretical if not specified
                        'is_gap_direct': getattr(doc, 'is_gap_direct', None),
                        'cbm': getattr(doc, 'cbm', None),
                        'vbm': getattr(doc, 'vbm', None),
                        'bulk_modulus': getattr(doc, 'bulk_modulus', None),
                        'shear_modulus': getattr(doc, 'shear_modulus', None),
                        'symmetry': {'number': sg_number, 'symbol': getattr(doc.symmetry, 'symbol', 'Unknown')} if sg_number else None
                    }
            
            print(f"No exact match in MP for Formula: {formula} AND Space Group: {space_group_number}")
            return None
            
        except Exception as e:
            print(f"Error querying Materials Project for {formula} (SG: {space_group_number}): {e}")
            import traceback
            traceback.print_exc() # Print full traceback for MP query errors
            return None
        
    
def main():
    """Main execution function with Materials Project API integration."""
    # --- Configuration ---
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    sqlite_db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"
    
    KSPACE_GRAPHS_OUTPUT_DIR = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs"
    MULTIMODAL_DB_OUTPUT_DIR = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/multimodal_materials_db_mp" 

    Path(KSPACE_GRAPHS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MULTIMODAL_DB_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    mp_api_key = "8O0LxKct7DKVrG2KqE9WhieXWpmsAZuu" # put in gitignore lol doesnt matter for now
    print(f"Using MP API Key: {mp_api_key}")
    max_materials = 10 

    print("Pre-generating k-space graphs (if not already present)...")
    kspace_graph_builder_instance = KSpacePhysicsGraphBuilder(csv_path, sqlite_db_path, KSPACE_GRAPHS_OUTPUT_DIR)
    kspace_graph_builder_instance.generate_kspace_graphs_for_all_space_groups()
    print("K-space graphs pre-generation complete (skipped if not explicitly called).")

    database_manager_instance = MultiModalMaterialDatabase(MULTIMODAL_DB_OUTPUT_DIR)

    materials_df = pd.read_csv(csv_path)

    print("Initializing Integrated Material Processor...")
    processor = IntegratedMaterialProcessor(
        csv_path=csv_path, 
        db_path=sqlite_db_path,
        kspace_graphs_base_dir=KSPACE_GRAPHS_OUTPUT_DIR, 
        database_manager=database_manager_instance 
    )
    
    # --- Main Processing Loop ---
    print(f"\nStarting data generation for up to {max_materials} materials from your CSV (using Materials Project)...")
    
    successful_generations = 0 
    SPACE_GROUP_COLUMN_NAME = 'Space Group' 

    processed_icsd_ids = set()

    for index, row in tqdm.tqdm(materials_df.head(max_materials).iterrows(), total=max_materials, desc="Processing Materials"):
        formula = row['Formula']
        icsd_id = row['ICSD_ID']
        
        if icsd_id in processed_icsd_ids:
            continue
        
        if pd.isna(row[SPACE_GROUP_COLUMN_NAME]):
            print(f"✗ Skipping formula '{formula}' (ICSD: {icsd_id}): Missing Space Group in CSV.")
            continue
        
        try:
            space_group_string = str(row[SPACE_GROUP_COLUMN_NAME])
            # Assuming space group string is like "14 P2_1/c" and we want "14"
            space_group_number_str = space_group_string.split(' ')[0]
            space_group_from_csv = int(space_group_number_str)
        except (ValueError, IndexError):
            print(f"✗ Skipping formula '{formula}' (ICSD: {icsd_id}): Could not parse space group number from '{row[SPACE_GROUP_COLUMN_NAME]}'.")
            continue

        unique_save_id = f"ICSD-{icsd_id}"
        existing_metadata_path = Path(MULTIMODAL_DB_OUTPUT_DIR) / 'metadata' / f"{unique_save_id}.json"
        if existing_metadata_path.exists():
            print(f"✓ Skipping {unique_save_id}: Already processed.")
            successful_generations += 1
            processed_icsd_ids.add(icsd_id)
            continue
        
        mp_data = processor.get_mp_material_data(formula, space_group_from_csv) 

        if not mp_data:
            print(f"✗ No Materials Project entry found for Formula: {formula} AND Space Group: {space_group_from_csv}. Skipping.")
            continue
        
        record = processor.generate_and_save_material_record( # Corrected: use 'processor'
            mp_material_data=mp_data,
            save_id=unique_save_id,
        )
        
        if record:
            successful_generations += 1
            processed_icsd_ids.add(icsd_id)
            print(f"✓ Successfully processed and saved {unique_save_id} (MP ID: {record.jid})")
            
    print(f"\n" + "="*60)
    print(f"DATASET GENERATION COMPLETE")
    print(f"="*60)
    print(f"Total materials in CSV: {len(materials_df)}")
    print(f"Successfully processed and saved: {successful_generations}")
    print(f"Database location: {MULTIMODAL_DB_OUTPUT_DIR}")
    print(f"="*60)
    
    if successful_generations > 0:
        print("\nCreating master index and ML datasets...")
        master_df = database_manager_instance.create_master_index()
        if not master_df.empty:
            database_manager_instance.create_ml_datasets() 
        else:
            print("No materials were successfully processed to create a master index or ML datasets.")
    else:
        print("No materials were successfully processed. Skipping master index and ML dataset creation.")

    # Print directory structure
    #inspect_database(MULTIMODAL_DB_OUTPUT_DIR)

if __name__ == "__main__":
    main()