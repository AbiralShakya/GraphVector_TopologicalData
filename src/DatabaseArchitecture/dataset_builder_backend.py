# jarvis data
# graph 1 for poscar
# graph 2 for k space
# point cloud for atom homology

from dataclasses import dataclass
import torch
import pandas as pd
import os
import re
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, List, Tuple, Union, Any
import json
import time
import pickle

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from torch_geometric.data import Data
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy
from sklearn.preprocessing import StandardScaler
import sqlite3
import ast

from jarvis.db.figshare import get_jid_data

class TopologicalMaterialAnalyzer:
    def __init__(self, csv_path: str, db_path: str):
        """
        Initialize the analyzer. This will load the local materials database
        and pre-compute necessary vocabularies for feature generation.
        """
        self.csv_path = csv_path
        self.formula_lookup = {}
        self.master_k_points = []
        self.master_irreps = []
        self.processed_count = 0
        self.matched_count = 0
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._load_and_prepare_databases()

    @dataclass
    class KSpaceTopologyData:
        """Data structure to capture k-space topology physics"""
        space_group_number: int
        ebr_data: Dict[str, Any]  # Elementary Band Representations
        topological_indices: Dict[str, float]  # Chern, Z2, etc.
        decomposition_branches: Dict[str, List]  # Band decomposition info
        kspace_graph: Data  # PyTorch Geometric graph
        connectivity_matrix: np.ndarray  # K-point connectivity
        physics_features: Dict[str, torch.Tensor]  # Physics-informed features


    def normalize_formula(self, formula: str) -> str:
        """Normalizes a chemical formula string for consistent lookups."""
        if pd.isna(formula) or formula == '':
            return ''
        
        formula_clean = str(formula).replace(" ", "").replace("_", "")
        # Handle common variations
        formula_clean = formula_clean.replace("(", "").replace(")", "")
        
        # Extract elements and their counts
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula_clean)
        
        element_counts = {}
        for element, count in matches:
            if count == '' or count == '1':
                count = 1
            else:
                try:
                    count = float(count)
                    if count == int(count):
                        count = int(count)
                except:
                    count = 1
            
            if element in element_counts:
                element_counts[element] += count
            else:
                element_counts[element] = count
        
        # Sort elements alphabetically and create normalized formula
        sorted_elements = sorted(element_counts.keys())
        normalized_parts = []
        for elem in sorted_elements:
            count = element_counts[elem]
            if count == 1:
                normalized_parts.append(elem)
            else:
                normalized_parts.append(f"{elem}{count}")
        
        return "".join(normalized_parts)

    def _load_and_prepare_databases(self):
        """
        Loads the local CSV for labels and queries the SQLite DB
        to build the vocabularies for k-points and irreps.
        """
        print(f"Loading local topological data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        for idx, row in df.iterrows():
            norm_formula = self.normalize_formula(row.get('Formula', ''))
            if norm_formula:
                self.formula_lookup[norm_formula] = row.to_dict()
        print(f"Loaded {len(self.formula_lookup)} unique materials from local CSV.")

        # Build vocabularies by querying the SQLite DB
        print(f"Building vocabularies from SQLite DB: {self.db_path}...")
        try:
            # Query for all unique k-points
            k_points_df = pd.read_sql_query("SELECT DISTINCT k_point FROM irreps", self.conn)
            self.master_k_points = sorted(k_points_df['k_point'].tolist())

            # Query for all unique irrep labels
            irreps_df = pd.read_sql_query("SELECT DISTINCT irrep_label FROM irreps", self.conn)
            self.master_irreps = sorted(irreps_df['irrep_label'].tolist())
            
            print(f"✓ Master K-Point Vocabulary Size: {len(self.master_k_points)}")
            print(f"✓ Master Irrep Vocabulary Size: {len(self.master_irreps)}")
        except Exception as e:
            print(f"✗ ERROR: Could not build vocabularies from SQLite DB. {e}")

    def _connect(self):
        """Establishes the database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

    def _close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _get_kspace_data_from_db(self, space_group_number: int) -> Optional[pd.DataFrame]:
        """
        Queries the SQLite DB to get all k-points, irrep labels,
        and associated decomposition branch information for a given space group number.
        """
        self._connect()

        try:
            query = """
                SELECT
                  ir.k_point,
                  ir.irrep_label,
                  b.branch1_irreps,
                  b.branch2_irreps,
                  b.decomposition_index
                FROM irreps AS ir
                LEFT JOIN ebrs ON ir.ebr_id = ebrs.id
                LEFT JOIN space_groups AS sg ON ebrs.space_group_id = sg.id
                LEFT JOIN ebr_decomposition_branches AS b ON ebrs.id = b.ebr_id
                WHERE
                  sg.number = ?;
            """
            df = pd.read_sql_query(query, self.conn, params=(space_group_number,))
            if not df.empty:
                return df
        except Exception as e:
            print(f"  ✗ Warning: Could not query k-space data for SG {space_group_number}: {e}")
        finally:
            self._close() 
        return None

    def generate_data_block_from_structure(self, structure: Structure, material_id: str, 
                                         formula: str = None) -> Optional[Dict]:
        """
        NEW METHOD: Generate data block directly from pymatgen Structure object.
        This is the key method that replaces generate_data_block_with_sg_check for MP data.
        
        Args:
            structure: pymatgen Structure object
            material_id: unique identifier (e.g., MP-12345)
            formula: chemical formula (optional, will be derived from structure if not provided)
        """
        self.processed_count += 1
        print(f"Processing Material ID: {material_id}")

        if structure is None:
            print(f"  ERROR: No structure provided for {material_id}")
            return None

        # Get formula from structure if not provided
        if formula is None:
            formula = structure.composition.reduced_formula
        
        print(f"  Formula: {formula}")
        norm_formula = self.normalize_formula(formula)
        print(f"  Normalized formula: {norm_formula}")

        # Check if material is in local topological dataset
        if norm_formula not in self.formula_lookup:
            print(f"  Formula {norm_formula} not found in local topological database")
            return None

        local_data_entry = self.formula_lookup[norm_formula]

        # Analyze space group
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            space_group = analyzer.get_space_group_symbol()
            
            # Check space group match with local data
            local_sg = local_data_entry.get('Spacegroup_Number')
            if local_sg and local_sg != sg_number:
                print(f"  Skipping {material_id} ({norm_formula}, SG {sg_number}): "
                      f"Space group mismatch with local DB (local: {local_sg})")
                return None
                
        except Exception as e:
            print(f"  Error in space group analysis: {e}")
            return None

        self.matched_count += 1
        print(f"  ✓ Match found! ({self.matched_count}/{self.processed_count})")
        print(f"  Space group: {space_group} ({sg_number})")

        # Get primitive structure and symmetry operations
        try:
            primitive_structure = analyzer.get_primitive_standard_structure()
            symm_ops = analyzer.get_symmetry_operations(cartesian=True)
            print(f"  Symmetry operations: {len(symm_ops)}")
        except Exception as e:
            print(f"  Error in symmetry analysis: {e}")
            primitive_structure = structure
            symm_ops = []

        # Query k-space data from database
        print("  Querying SQLite DB for k-space data...")
        kspace_df = self._get_kspace_data_from_db(sg_number)
        if kspace_df is None:
            print(f"  Skipping {material_id}: No k-space data found for SG {sg_number}")
            return None
        
        print(f"  ✓ Found {len(kspace_df)} k-space entries for SG {sg_number}")

        # Generate features using existing methods
        band_rep_vector, target_y = self._featurize_kspace_data(local_data_entry, kspace_df)
        
        print("  Generating features...")
        crystal_graph = self._create_crystal_graph(primitive_structure)
        asph_vector = self._compute_asph_features(structure)
        kspace_graph = self._create_kspace_graph(band_rep_vector)

        kspace_topology_data = self._load_pregenerated_kspace_data(sg_number)
        
        if kspace_topology_data:
            kspace_graph = kspace_topology_data.kspace_graph # Get the PyG Data object
            # Also capture other physics features if needed in the data block
            physics_features_tensor = kspace_topology_data.physics_features
            connectivity_matrix = kspace_topology_data.connectivity_matrix
            data_block['kspace_physics_features'] = physics_features_tensor
            data_block['kspace_connectivity_matrix'] = connectivity_matrix
        else:
            print(f"  Warning: No pre-generated k-space topology data found for SG {sg_number}. Creating fallback k-space graph.")
            # Fallback to creating a generic k-space graph if pre-generated data isn't available
            kspace_graph = self._create_fallback_kspace_graph(structure) # Re-using your fallback
            physics_features_tensor = {} # Empty dict for fallback
            connectivity_matrix = np.array([]) # Empty array for fallback

        
        crystal_graph.y = target_y

        # Create basic properties dictionary from structure
        basic_props = {
            'density': structure.density,
            'volume': structure.volume,
            'nsites': len(structure),
            'space_group': space_group,
            'space_group_number': sg_number
        }

        # Assemble the final data block
        data_block = {
            'jid': material_id,  # Using material_id as jid for compatibility
            'formula': formula,
            'normalized_formula': norm_formula,
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'asph_features': asph_vector,
            'band_rep_features': band_rep_vector,
            'target_label': target_y,
            'symmetry_ops': {
                'rotations': torch.stack([torch.tensor(op.rotation_matrix, dtype=torch.float) 
                                        for op in symm_ops]) if symm_ops else torch.empty(0, 3, 3),
                'translations': torch.stack([torch.tensor(op.translation_vector, dtype=torch.float) 
                                           for op in symm_ops]) if symm_ops else torch.empty(0, 3)
            },
            'structure_props': basic_props,
            'local_topo_data': local_data_entry
        }
        
        print(f"  ✓ Data block generated successfully")
        return data_block

    # Keep the original JARVIS method for backward compatibility
    def generate_data_block_with_sg_check(self, jid: str) -> Optional[Dict]:
        """Original method for JARVIS data - kept for backward compatibility"""
        self.processed_count += 1
        print(f"Processing JID: {jid}")

        jid_data = get_jid_data(jid=jid, dataset="dft_3d")
        if not jid_data:
            print(f"  NO data found for JID: {jid}")
            return None
        
        formula = jid_data.get('formula', '')
        print(f"  Formula: {formula}")

        if not formula:
            print(f"  No formula found for JID: {jid}")
            return None
        
        norm_formula = self.normalize_formula(formula)
        print(f"  Normalized formula: {norm_formula}")

        # Process only if material in local dataset 
        if norm_formula not in self.formula_lookup:
            print(f"  Formula {norm_formula} not found in local database")
            return None
        
        local_data_entry = self.formula_lookup[norm_formula]
        
        # Create structure from JARVIS data (existing logic)
        structure = None
        structure_keys_to_try = ['atoms', 'structure', 'final_str', 'initial_structure']
        
        for key in structure_keys_to_try:
            if key in jid_data:
                try:
                    print(f"  Trying to create structure from key: {key}")
                    if isinstance(jid_data[key], dict):
                        structure = Structure.from_dict(jid_data[key])
                        print(f"  ✓ Structure created from '{key}': {len(structure)} atoms")
                        break
                    elif isinstance(jid_data[key], str):
                        from pymatgen.io.vasp.inputs import Poscar
                        poscar = Poscar.from_string(jid_data[key])
                        structure = poscar.structure
                        print(f"  ✓ Structure created from POSCAR string in '{key}': {len(structure)} atoms")
                        break
                except Exception as e:
                    print(f"    Failed to create structure from '{key}': {e}")
                    continue
        
        if structure is None:
            try:
                print("  Attempting to build structure manually from JARVIS format...")    
                if 'atoms' in jid_data:
                    atoms_data = jid_data['atoms']
                    if 'lattice_mat' in atoms_data and 'coords' in atoms_data and 'elements' in atoms_data:
                        from pymatgen.core import Lattice
                        
                        lattice = Lattice(atoms_data['lattice_mat'])
                        coords = atoms_data['coords']
                        elements = atoms_data['elements']
                        
                        coords_are_cartesian = atoms_data.get('cartesian', False)
                        
                        structure = Structure(lattice, elements, coords, coords_are_cartesian=coords_are_cartesian)
                        print(f"  ✓ Structure built from JARVIS format: {len(structure)} atoms")
                
            except Exception as e:
                print(f"    Manual structure building failed: {e}")
                
        if structure is None:
            print(f"  ERROR: Could not create structure for {jid}")
            return None

        # Now use the unified structure processing method
        return self._process_structure_common(structure, jid, formula, local_data_entry, 
                                            additional_props={'jarvis_props': {
                                                'formation_energy': jid_data.get('form_energy_per_atom'),
                                                'band_gap': jid_data.get('optb88vdw_bandgap')
                                            }})

    def _process_structure_common(self, structure: Structure, material_id: str, 
                                formula: str, local_data_entry: dict, 
                                additional_props: dict = None) -> Optional[Dict]:
        """
        Common structure processing logic for both JARVIS and MP data.
        This eliminates code duplication between the two methods.
        """
        # Check space group match
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            space_group = analyzer.get_space_group_symbol()
            
            local_sg = local_data_entry.get('Spacegroup_Number')
            if local_sg and local_sg != sg_number:
                print(f"  Skipping {material_id}: Space group mismatch "
                      f"(computed: {sg_number}, local: {local_sg})")
                return None
                
        except Exception as e:
            print(f"  Error in space group analysis: {e}")
            return None

        self.matched_count += 1
        print(f"  ✓ Match found! ({self.matched_count}/{self.processed_count})")
        
        # Get primitive structure and symmetry operations
        try:
            primitive_structure = analyzer.get_primitive_standard_structure()
            symm_ops = analyzer.get_symmetry_operations(cartesian=True)
            print(f"  Symmetry operations: {len(symm_ops)}")
        except Exception as e:
            print(f"  Error in symmetry analysis: {e}")
            primitive_structure = structure
            symm_ops = []
        
        # Query k-space data
        print("  Querying SQLite DB for k-space data...")
        kspace_df = self._get_kspace_data_from_db(sg_number)
        if kspace_df is None:
            print(f"  Skipping {material_id}: No k-space data found for SG {sg_number}")
            return None
        
        print(f"  ✓ Found {len(kspace_df)} k-space entries for SG {sg_number}")
        
        # Generate features
        band_rep_vector, target_y = self._featurize_kspace_data(local_data_entry, kspace_df)
        
        print("  Generating features...")
        crystal_graph = self._create_crystal_graph(primitive_structure)
        asph_vector = self._compute_asph_features(structure)
        kspace_graph = self._create_kspace_graph(band_rep_vector)
        
        crystal_graph.y = target_y
        
        # Assemble the final data block
        data_block = {
            'jid': material_id,
            'formula': formula,
            'normalized_formula': self.normalize_formula(formula),
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'asph_features': asph_vector,
            'band_rep_features': band_rep_vector,
            'target_label': target_y,
            'symmetry_ops': {
                'rotations': torch.stack([torch.tensor(op.rotation_matrix, dtype=torch.float) 
                                        for op in symm_ops]) if symm_ops else torch.empty(0, 3, 3),
                'translations': torch.stack([torch.tensor(op.translation_vector, dtype=torch.float) 
                                           for op in symm_ops]) if symm_ops else torch.empty(0, 3)
            },
            'structure_props': {
                'space_group': space_group,
                'space_group_number': sg_number,
                'density': structure.density,
                'volume': structure.volume,
                'nsites': len(structure)
            },
            'local_topo_data': local_data_entry
        }
        
        # Add any additional properties
        if additional_props:
            data_block.update(additional_props)
        
        print(f"  ✓ Data block generated successfully")
        return data_block

    def _get_elemental_features(self, atomic_number: int) -> torch.Tensor:
        """Creates a feature vector for a given element using basic atomic properties."""
        periods = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 
                  11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3}
        period = periods.get(atomic_number, 4)
        
        features = [
            atomic_number / 100.0,
            period / 8.0,
            (atomic_number % 18) / 18.0
        ]
        
        return torch.tensor(features, dtype=torch.float)

    def _compute_asph_features(self, structure: Structure, n_bins=20) -> torch.Tensor:
        """Computes Atom-Specific Persistent Homology (ASPH) feature vector.""" 
        try:
            from gtda.homology import VietorisRipsPersistence
            from gtda.diagrams import BettiCurve, PersistenceEntropy
            
            coords = np.array(structure.cart_coords)
            points = coords.reshape(1, -1, 3)
            
            vr = VietorisRipsPersistence(
                homology_dimensions=[0, 1, 2], 
                max_edge_length=10.0,
                n_jobs=1
            )
            diagrams = vr.fit_transform(points)
            
            bc = BettiCurve(n_bins=n_bins)
            betti_curves = bc.fit_transform(diagrams)
            
            pe = PersistenceEntropy()
            entropy_features = pe.fit_transform(diagrams)
            
            combined_features = np.concatenate([
                betti_curves.flatten(),
                entropy_features.flatten()
            ])
            
            return torch.tensor(combined_features, dtype=torch.float)
            
        except Exception as e:
            print(f"Error computing ASPH features: {e}")
            return torch.zeros(n_bins * 3 + 3, dtype=torch.float)
        
    def _featurize_kspace_data(self, local_data_entry: dict, kspace_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts structured k-space data from the DB into a feature vector
        and gets the classification label from the local CSV data.
        """
        # Determine the target label from CSV 'Property' column
        property_str = str(local_data_entry.get('Property', '')).upper()
        is_topological = any(indicator in property_str for indicator in ['TI', 'SM', 'WEYL', 'DIRAC'])
        target_y = torch.tensor([1.0 if is_topological else 0.0], dtype=torch.float)

        # Build the feature vector from the database query
        feature_dim = len(self.master_k_points) + len(self.master_irreps)
        band_rep_vector = torch.zeros(feature_dim, dtype=torch.float)

        if kspace_df is not None:
            present_k_points = set(kspace_df['k_point'])
            present_irreps = set(kspace_df['irrep_label'])

            for i, k_point in enumerate(self.master_k_points):
                if k_point in present_k_points:
                    band_rep_vector[i] = 1.0
            
            offset = len(self.master_k_points)
            for i, irrep in enumerate(self.master_irreps):
                if irrep in present_irreps:
                    band_rep_vector[offset + i] = 1.0
        
        return band_rep_vector, target_y

    def _create_crystal_graph(self, structure: Structure) -> Data:
        """Generates the real-space atomic graph using Voronoi tessellation."""
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            from torch_geometric.data import Data
            
            vnn = VoronoiNN(cutoff=13.0, allow_pathological=True, tol=0.8)
            
            edge_index_list = []
            edge_weight_list = []
            
            for i, site in enumerate(structure):
                try:
                    neighbors = vnn.get_nn_info(structure, i)
                    for neighbor_info in neighbors:
                        j = neighbor_info['site_index']
                        if i != j:
                            edge_index_list.append([i, j])
                            edge_weight_list.append(neighbor_info['weight'])
                except Exception as e:
                    continue
            
            node_features = []
            for site in structure:
                atomic_number = site.specie.Z
                node_features.append(self._get_elemental_features(atomic_number))
            
            node_features = torch.stack(node_features) if node_features else torch.empty(0, 3)
            node_positions = torch.tensor(structure.cart_coords, dtype=torch.float)
            
            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_weight = torch.empty(0, dtype=torch.float)
            
            return Data(
                x=node_features,
                pos=node_positions,
                edge_index=edge_index,
                edge_attr=edge_weight.unsqueeze(1) if len(edge_weight) > 0 else torch.empty(0, 1)
            )
            
        except Exception as e:
            print(f"Error creating crystal graph: {e}")
            node_features = torch.stack([self._get_elemental_features(site.specie.Z) for site in structure])
            return Data(
                x=node_features,
                pos=torch.tensor(structure.cart_coords, dtype=torch.float),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 1)
            )

    def _load_pregenerated_kspace_data(self, space_group_number: int) -> Optional[KSpaceTopologyData]:
        """
        Loads the pre-generated KSpaceTopologyData object for a given space group.
        """
        sg_folder = self.kspace_graphs_base_dir / f"SG_{space_group_number:03d}"
        pkl_path = sg_folder / "topology_data.pkl"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"  Error loading pre-generated k-space data for SG {space_group_number}: {e}")
        else:
            print(f"  Pre-generated k-space data not found for SG {space_group_number} at {pkl_path}")
        return None
    
    def enhanced_generate_data_block_from_structure(self, structure: Structure, material_id: str, 
                                     formula: str = None) -> Optional[Dict]:
        """
        ENHANCED VERSION: Generate data block directly from pymatgen Structure object.
        This version includes better error handling and serialization for database storage.
        
        Args:
            structure: pymatgen Structure object
            material_id: unique identifier (e.g., MP-12345)
            formula: chemical formula (optional, will be derived from structure if not provided)
        """
        self.processed_count += 1
        print(f"Processing Material ID: {material_id}")

        if structure is None:
            print(f"  ERROR: No structure provided for {material_id}")
            return None

        # Get formula from structure if not provided
        if formula is None:
            formula = structure.composition.reduced_formula
        
        print(f"  Formula: {formula}")
        norm_formula = self.normalize_formula(formula)
        print(f"  Normalized formula: {norm_formula}")

        # Check if material is in local topological dataset
        if norm_formula not in self.formula_lookup:
            print(f"  Formula {norm_formula} not found in local topological database")
            return None

        local_data_entry = self.formula_lookup[norm_formula]

        # Analyze space group
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            space_group = analyzer.get_space_group_symbol()
            
            # Check space group match with local data
            local_sg = local_data_entry.get('Spacegroup_Number')
            if local_sg and local_sg != sg_number:
                print(f"  Skipping {material_id} ({norm_formula}, SG {sg_number}): "
                    f"Space group mismatch with local DB (local: {local_sg})")
                return None
                
        except Exception as e:
            print(f"  Error in space group analysis: {e}")
            return None

        self.matched_count += 1
        print(f"  ✓ Match found! ({self.matched_count}/{self.processed_count})")
        print(f"  Space group: {space_group} ({sg_number})")

        # Get primitive structure and symmetry operations
        try:
            primitive_structure = analyzer.get_primitive_standard_structure()
            symm_ops = analyzer.get_symmetry_operations(cartesian=True)
            print(f"  Symmetry operations: {len(symm_ops)}")
        except Exception as e:
            print(f"  Error in symmetry analysis: {e}")
            primitive_structure = structure
            symm_ops = []

        # Query k-space data from database
        print("  Querying SQLite DB for k-space data...")
        kspace_df = self._get_kspace_data_from_db(sg_number)
        if kspace_df is None:
            print(f"  Skipping {material_id}: No k-space data found for SG {sg_number}")
            return None
        
        print(f"  ✓ Found {len(kspace_df)} k-space entries for SG {sg_number}")

        # Generate features using existing methods
        band_rep_vector, target_y = self._featurize_kspace_data(local_data_entry, kspace_df)
        
        print("  Generating features...")
        crystal_graph = self._create_crystal_graph(primitive_structure)
        asph_vector = self._compute_asph_features(structure)
        kspace_graph = self._create_kspace_graph(band_rep_vector)
        
        crystal_graph.y = target_y

        # Create basic properties dictionary from structure
        basic_props = {
            'density': structure.density,
            'volume': structure.volume,
            'nsites': len(structure),
            'space_group': space_group,
            'space_group_number': sg_number
        }
        
        # ENHANCED: Convert torch tensors to serializable formats for database storage
        try:
            # Convert crystal graph to serializable format
            crystal_graph_dict = {
                'x': crystal_graph.x.numpy() if hasattr(crystal_graph.x, 'numpy') else crystal_graph.x,
                'pos': crystal_graph.pos.numpy() if hasattr(crystal_graph.pos, 'numpy') else crystal_graph.pos,
                'edge_index': crystal_graph.edge_index.numpy() if hasattr(crystal_graph.edge_index, 'numpy') else crystal_graph.edge_index,
                'edge_attr': crystal_graph.edge_attr.numpy() if hasattr(crystal_graph.edge_attr, 'numpy') else crystal_graph.edge_attr,
                'y': crystal_graph.y.numpy() if hasattr(crystal_graph.y, 'numpy') else crystal_graph.y,
                'num_nodes': crystal_graph.num_nodes if hasattr(crystal_graph, 'num_nodes') else len(crystal_graph.x)
            }
            
            # Convert k-space graph to serializable format
            kspace_graph_dict = {
                'x': kspace_graph.x.numpy() if hasattr(kspace_graph.x, 'numpy') else kspace_graph.x,
                'edge_index': kspace_graph.edge_index.numpy() if hasattr(kspace_graph.edge_index, 'numpy') else kspace_graph.edge_index,
                'u': kspace_graph.u.numpy() if hasattr(kspace_graph.u, 'numpy') else kspace_graph.u,
                'num_nodes': kspace_graph.num_nodes if hasattr(kspace_graph, 'num_nodes') else len(kspace_graph.x)
            }
            
            # Convert symmetry operations to serializable format
            symmetry_ops_dict = {
                'rotations': symm_ops[0].rotation_matrix.tolist() if symm_ops else [],
                'translations': symm_ops[0].translation_vector.tolist() if symm_ops else [],
                'num_ops': len(symm_ops)
            }
            
        except Exception as e:
            print(f"  Warning: Error converting tensors to serializable format: {e}")
            # Fallback to original torch objects
            crystal_graph_dict = crystal_graph
            kspace_graph_dict = kspace_graph
            symmetry_ops_dict = {
                'rotations': torch.stack([torch.tensor(op.rotation_matrix, dtype=torch.float) 
                                        for op in symm_ops]) if symm_ops else torch.empty(0, 3, 3),
                'translations': torch.stack([torch.tensor(op.translation_vector, dtype=torch.float) 
                                        for op in symm_ops]) if symm_ops else torch.empty(0, 3)
            }

        # Assemble the final data block with enhanced serialization
        data_block = {
            'jid': material_id,
            'formula': formula,
            'normalized_formula': norm_formula,
            'crystal_graph': crystal_graph_dict,  # Now serializable
            'kspace_graph': kspace_graph_dict,    # Now serializable
            'asph_features': asph_vector.numpy() if hasattr(asph_vector, 'numpy') else asph_vector,
            'band_rep_features': band_rep_vector.numpy() if hasattr(band_rep_vector, 'numpy') else band_rep_vector,
            'target_label': target_y.numpy() if hasattr(target_y, 'numpy') else target_y,
            'symmetry_ops': symmetry_ops_dict,    # Now serializable
            'structure_props': basic_props,
            'local_topo_data': local_data_entry,
            
            # Additional metadata for better tracking
            'processing_metadata': {
                'analyzer_version': '1.0',
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'primitive_structure_formula': primitive_structure.composition.reduced_formula,
                'symmetry_precision': 0.1,
                'kspace_entries_count': len(kspace_df),
                'feature_dimensions': {
                    'asph': len(asph_vector) if hasattr(asph_vector, '__len__') else 0,
                    'band_rep': len(band_rep_vector) if hasattr(band_rep_vector, '__len__') else 0,
                    'crystal_graph_nodes': len(crystal_graph.x) if hasattr(crystal_graph, 'x') else 0,
                    'kspace_graph_nodes': len(kspace_graph.x) if hasattr(kspace_graph, 'x') else 0
                }
            }
        }
        
        print(f"  ✓ Data block generated successfully with serializable format")
        return data_block

    def get_topological_summary_stats(self) -> Dict:
        """Get summary statistics about the topological analysis process."""
        return {
            'total_processed': self.processed_count,
            'total_matched': self.matched_count,
            'match_rate': self.matched_count / max(self.processed_count, 1),
            'vocabulary_sizes': {
                'k_points': len(self.master_k_points),
                'irreps': len(self.master_irreps)
            },
            'database_info': {
                'csv_path': self.csv_path,
                'db_path': self.db_path,
                'formula_lookup_size': len(self.formula_lookup)
            }
        }

    def validate_data_block(self, data_block: Dict) -> bool:
        """Validate that a data block contains all required fields."""
        required_fields = [
            'jid', 'formula', 'normalized_formula', 'crystal_graph', 
            'kspace_graph', 'asph_features', 'band_rep_features', 
            'target_label', 'structure_props', 'local_topo_data'
        ]
        
        missing_fields = [field for field in required_fields if field not in data_block]
        
        if missing_fields:
            print(f"Data block validation failed. Missing fields: {missing_fields}")
            return False
        
        # Check for empty or None values in critical fields
        critical_checks = {
            'asph_features': lambda x: x is not None and len(x) > 0,
            'band_rep_features': lambda x: x is not None and len(x) > 0,
            'crystal_graph': lambda x: x is not None,
            'kspace_graph': lambda x: x is not None
        }
        
        for field, check_func in critical_checks.items():
            if not check_func(data_block[field]):
                print(f"Data block validation failed. Invalid {field}")
                return False
        
        return True

    def export_features_for_ml(self, data_blocks: List[Dict], output_path: str = None) -> Dict:
        """
        Export processed data blocks in a format suitable for machine learning.
        
        Args:
            data_blocks: List of data blocks from topological analysis
            output_path: Optional path to save the exported data
            
        Returns:
            Dictionary containing organized ML-ready data
        """
        if not data_blocks:
            print("No data blocks provided for export")
            return {}
        
        # Organize data for ML
        ml_data = {
            'features': {
                'asph': [],
                'band_rep': [],
                'combined': []
            },
            'targets': {
                'binary': [],
                'class_labels': []
            },
            'metadata': {
                'jids': [],
                'formulas': [],
                'space_groups': []
            },
            'graphs': {
                'crystal': [],
                'kspace': []
            }
        }
        
        for data_block in data_blocks:
            if not self.validate_data_block(data_block):
                continue
                
            # Extract features
            asph_features = data_block['asph_features']
            band_rep_features = data_block['band_rep_features']
            
            # Convert to numpy arrays if needed

            
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

class KSpaceGraphBuilder:
    """
    A builder class to create physics-informed k-space graphs.
    It manages vocabularies for categorical features (irreps, decomposition indices)
    to ensure consistent numerical encoding across different materials.
    """
    def __init__(self,
                 initial_df_kspace_data: Optional[pd.DataFrame] = None,
                 irrep_label_vocab: Optional[Dict[str, int]] = None,
                 branch_irrep_vocab: Optional[Dict[str, int]] = None,
                 decomposition_index_vocab: Optional[Dict[Union[int, float, None], int]] = None):
        """
        Initializes the KSpaceGraphBuilder.

        Args:
            initial_df_kspace_data (pd.DataFrame, optional): An initial DataFrame
                from _get_kspace_data_from_db to build vocabularies from.
                Useful if you want to pre-build vocabularies once.
            irrep_label_vocab (Dict, optional): Pre-existing vocabulary for k-point irrep labels.
            branch_irrep_vocab (Dict, optional): Pre-existing vocabulary for branch irrep labels.
            decomposition_index_vocab (Dict, optional): Pre-existing vocabulary for decomposition indices.
        """
        self.irrep_label_vocab = irrep_label_vocab if irrep_label_vocab is not None else {}
        self.branch_irrep_vocab = branch_irrep_vocab if branch_irrep_vocab is not None else {}
        self.decomposition_index_vocab = decomposition_index_vocab if decomposition_index_vocab is not None else {}

        # If vocabularies are not provided, attempt to build them from initial data
        if initial_df_kspace_data is not None and not initial_df_kspace_data.empty:
            self._build_vocabularies(initial_df_kspace_data)
        elif not (self.irrep_label_vocab and self.branch_irrep_vocab and self.decomposition_index_vocab):
            print("Warning: No initial data or pre-built vocabularies provided. Vocabs will be built on the fly, which might lead to inconsistencies if not all possible values are seen during graph creation. Consider building them once from your full dataset.")


    def _build_vocabularies(self, df: pd.DataFrame):
        """
        Builds or extends vocabularies for categorical features from a given DataFrame.
        This should be run once on a representative sample or the entire dataset
        to ensure consistent ID mapping.
        """
        # For irrep_label (local k-point irreps)
        unique_irrep_labels = df['irrep_label'].dropna().unique()
        for label in unique_irrep_labels:
            if label not in self.irrep_label_vocab:
                self.irrep_label_vocab[label] = len(self.irrep_label_vocab)

        # For branch_irreps (can contain multiple comma-separated irreps per entry)
        all_branch_irreps_flat = []
        for col in ['branch1_irreps', 'branch2_irreps']:
            if col in df.columns:
                # Handle potential NaN values and split strings
                for irrep_str in df[col].dropna():
                    all_branch_irreps_flat.extend([s.strip() for s in irrep_str.split(',') if s.strip()])
        unique_branch_irreps = sorted(list(set(all_branch_irreps_flat)))
        for irrep in unique_branch_irreps:
            if irrep not in self.branch_irrep_vocab:
                self.branch_irrep_vocab[irrep] = len(self.branch_irrep_vocab)

        # For decomposition_index
        if 'decomposition_index' in df.columns:
            unique_decomposition_indices = df['decomposition_index'].dropna().unique()
            for idx in unique_decomposition_indices:
                if idx not in self.decomposition_index_vocab:
                    self.decomposition_index_vocab[idx] = len(self.decomposition_index_vocab)
            # Add a category for NaN/None if it's expected to be present
            if df['decomposition_index'].isnull().any() and None not in self.decomposition_index_vocab:
                self.decomposition_index_vocab[None] = len(self.decomposition_index_vocab)


    def _encode_irrep_label(self, label: str) -> torch.Tensor:
        """One-hot encodes a single irrep_label based on the current vocabulary."""
        # Ensure vocab is not empty if it's built on the fly
        if not self.irrep_label_vocab:
            # Fallback or error if vocab is empty (should ideally be pre-built)
            print(f"Error: irrep_label_vocab is empty when encoding '{label}'.")
            return torch.zeros(1) # Return a minimal tensor, but this indicates a problem

        one_hot = torch.zeros(len(self.irrep_label_vocab))
        if label in self.irrep_label_vocab:
            one_hot[self.irrep_label_vocab[label]] = 1.0
        else:
            # Handle unseen labels: add to vocab on the fly, or assign to an 'unknown' index
            # On-the-fly addition:
            new_idx = len(self.irrep_label_vocab)
            self.irrep_label_vocab[label] = new_idx
            new_one_hot = torch.zeros(new_idx + 1) # Expand vector
            new_one_hot[new_idx] = 1.0
            return new_one_hot
            # If you want a fixed vocabulary:
            # print(f"Warning: Unseen irrep_label '{label}' encountered. Using zero vector.")
            # return torch.zeros(len(self.irrep_label_vocab))
        return one_hot

    def _encode_branch_irreps(self, irreps_str: str) -> torch.Tensor:
        """Multi-hot encodes a comma-separated string of branch irreps based on vocab."""
        if not irreps_str:
            return torch.zeros(len(self.branch_irrep_vocab))
        
        individual_irreps = [s.strip() for s in irreps_str.split(',') if s.strip()]
        
        # Ensure vocab is not empty if built on the fly
        if not self.branch_irrep_vocab:
            print(f"Error: branch_irrep_vocab is empty when encoding '{irreps_str}'.")
            return torch.zeros(1)

        multi_hot = torch.zeros(len(self.branch_irrep_vocab))
        for irrep in individual_irreps:
            if irrep in self.branch_irrep_vocab:
                multi_hot[self.branch_irrep_vocab[irrep]] = 1.0
            else:
                # On-the-fly addition:
                new_idx = len(self.branch_irrep_vocab)
                self.branch_irrep_vocab[irrep] = new_idx
                # Create a larger multi_hot if expanding vocab
                temp_multi_hot = torch.zeros(new_idx + 1)
                temp_multi_hot[:multi_hot.size(0)] = multi_hot # Copy old values
                temp_multi_hot[new_idx] = 1.0
                multi_hot = temp_multi_hot # Update reference
                # If you want a fixed vocabulary:
                # print(f"Warning: Unseen branch irrep '{irrep}' encountered. Not encoded.")
        return multi_hot

    def _encode_decomposition_index(self, index: Union[int, float, None]) -> torch.Tensor:
        """Encodes decomposition index (one-hot if categorical)."""
        # Ensure vocab is not empty if built on the fly
        if not self.decomposition_index_vocab:
            print(f"Error: decomposition_index_vocab is empty when encoding '{index}'.")
            return torch.zeros(1) # Return a minimal tensor

        # Handle NaN values explicitly, mapping to 'None' in vocab
        if pd.isna(index):
            index_key = None
        else:
            index_key = index

        one_hot = torch.zeros(len(self.decomposition_index_vocab))
        if index_key in self.decomposition_index_vocab:
            one_hot[self.decomposition_index_vocab[index_key]] = 1.0
        else:
            # On-the-fly addition:
            new_idx = len(self.decomposition_index_vocab)
            self.decomposition_index_vocab[index_key] = new_idx
            new_one_hot = torch.zeros(new_idx + 1) # Expand vector
            new_one_hot[new_idx] = 1.0
            return new_one_hot
            # If you want a fixed vocabulary:
            # print(f"Warning: Unseen decomposition_index '{index}' encountered. Using zero vector.")
            # return torch.zeros(len(self.decomposition_index_vocab))
        return one_hot


    def create_kspace_graph(
        self,
        df_kspace_data: pd.DataFrame,
        band_rep_vector: torch.Tensor,
        electric_field_vector: Optional[torch.Tensor] = None, # e.g., torch.tensor([Ex, Ey, Ez])
        space_group_number: Optional[int] = None
    ) -> Data:
        """
        Generates a physics-informed k-space graph with enriched features for ML.

        Args:
            df_kspace_data (pd.DataFrame): DataFrame containing k_point, irrep_label,
                                          branch1_irreps, branch2_irreps, decomposition_index
                                          for a specific material/space group, typically from
                                          _get_kspace_data_from_db.
            band_rep_vector (torch.Tensor): A tensor representing the overall band
                                            representation, used as a global graph feature.
            electric_field_vector (Optional[torch.Tensor]): 3D vector representing an
                                                               external electric field.
                                                               If None, no field is applied.
            space_group_number (Optional[int]): The space group number of the material,
                                                added as a global feature.

        Returns:
            torch_geometric.data.Data: A PyTorch Geometric Data object with physics-informed
                                       nodes, edges, and global features.
        """
        if df_kspace_data.empty:
            print("Warning: Empty k-space data DataFrame provided. Returning empty graph.")
            # Provide a dummy empty graph with consistent feature dimensions if possible
            # This requires knowing the expected total feature sizes, which depend on vocab size.
            # For simplicity, return a graph with 3D k-coords and 0 global features.
            return Data(x=torch.empty(0, 3 + len(self.irrep_label_vocab) + 2*len(self.branch_irrep_vocab) + len(self.decomposition_index_vocab)),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        u=torch.empty(1, band_rep_vector.size(0) + 3 + 1)) # Dummy based on expected features

        # Ensure vocabularies are built/updated from the current dataframe
        self._build_vocabularies(df_kspace_data)

        # 1. Node Features (x):
        # Group by k_point coordinates to ensure each unique k_point is one node.
        # Aggregate features for k_points that might have multiple irrep_labels/decomposition_indices
        # (e.g., if a k-point corresponds to multiple bands with different irreps).
        grouped_k_points = df_kspace_data.groupby('k_point').agg({
            'irrep_label': lambda x: list(x.dropna().unique()),
            'branch1_irreps': lambda x: [s.strip() for val in x.dropna() for s in val.split(',') if s.strip()],
            'branch2_irreps': lambda x: [s.strip() for val in x.dropna() for s in val.split(',') if s.strip()],
            'decomposition_index': lambda x: list(x.dropna().unique())
        }).reset_index()

        node_features_list = []
        k_point_coords_list = [] # Store numerical k-point coordinates for edge creation

        for idx, row in grouped_k_points.iterrows():
            k_point_str = row['k_point']
            try:
                # Robustly parse k_point string. It might be "(x,y,z)" or "x y z"
                if '(' in k_point_str and ')' in k_point_str:
                    coords = torch.tensor(ast.literal_eval(k_point_str), dtype=torch.float)
                else:
                    coords = torch.tensor([float(x) for x in k_point_str.split()], dtype=torch.float)
                
                # Handle 2D k-points by padding with a zero for the z-coordinate
                if coords.shape[0] == 2:
                    coords = torch.cat((coords, torch.zeros(1, dtype=torch.float)))
                elif coords.shape[0] != 3:
                    raise ValueError(f"K-point coordinates are not 2D or 3D: {coords.shape}")

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing k_point string '{k_point_str}': {e}. Skipping this k-point.")
                continue
            
            k_point_coords_list.append(coords) # Store for edge creation

            # Encode irrep_label(s) - sum their one-hots if multiple irreps for this k-point
            encoded_irrep_labels = torch.zeros(len(self.irrep_label_vocab))
            for label in row['irrep_label']:
                encoded_irrep_labels += self._encode_irrep_label(label)
            # You might normalize or binarize `encoded_irrep_labels` if summing leads to values > 1

            # Encode branch_irreps (multi-hot)
            # Flatten lists of lists and take unique elements from collected branch irreps
            all_branch_irreps_for_k = list(set(row['branch1_irreps'] + row['branch2_irreps']))
            
            encoded_branch_irreps = torch.zeros(len(self.branch_irrep_vocab))
            for irrep in all_branch_irreps_for_k:
                 if irrep in self.branch_irrep_vocab: # Check if it exists in the current vocab
                    encoded_branch_irreps[self.branch_irrep_vocab[irrep]] = 1.0
                 # If not in vocab, it implies a warning was already printed during _encode_branch_irreps call

            # Encode decomposition_index (summing one-hots if multiple, or pick first and encode)
            encoded_decomposition_index = torch.zeros(len(self.decomposition_index_vocab))
            if row['decomposition_index']:
                # For simplicity, let's just encode the first unique decomposition index found
                encoded_decomposition_index += self._encode_decomposition_index(row['decomposition_index'][0])
            else: # Handle case where decomposition_index is missing
                encoded_decomposition_index = self._encode_decomposition_index(None)

            # Concatenate all features for the node
            node_feature = torch.cat([
                coords,
                encoded_irrep_labels,
                encoded_branch_irreps,
                encoded_decomposition_index
            ])
            node_features_list.append(node_feature)

        if not node_features_list:
            print("No valid node features generated. Returning empty graph.")
            return Data(x=torch.empty(0, 3 + len(self.irrep_label_vocab) + len(self.branch_irrep_vocab) + len(self.decomposition_index_vocab)),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        u=torch.empty(1, band_rep_vector.size(0) + 3 + 1)) # Match dummy dimensions

        node_features = torch.stack(node_features_list)
        
        # 2. Edge Index (edge_index): Connect sequential k-points along the path.
        # This assumes `k_point_coords_list` (and thus `grouped_k_points`) is ordered
        # according to the desired path. If not, you'd need explicit path information.
        n_unique_k_points = len(k_point_coords_list)
        edge_index = []
        for i in range(n_unique_k_points - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i]) # Undirected
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)

        # 3. Global Features (u):
        global_features_list = [band_rep_vector.unsqueeze(0)] # Start with band_rep_vector

        if electric_field_vector is not None:
            if electric_field_vector.dim() == 1:
                electric_field_vector = electric_field_vector.unsqueeze(0) # Ensure 2D (1, 3)
            global_features_list.append(electric_field_vector.to(torch.float))
        else:
            # If no electric field is applied, add a zero vector of same dimension for consistency
            global_features_list.append(torch.zeros(1, 3, dtype=torch.float)) # Assuming 3D field

        if space_group_number is not None:
            # Add space group number as a scalar global feature.
            # Could be one-hot encoded if desired, but for simplicity, directly as float.
            global_features_list.append(torch.tensor([[float(space_group_number)]], dtype=torch.float))
        else:
            global_features_list.append(torch.tensor([[0.0]], dtype=torch.float)) # Placeholder for missing SG

        # Concatenate all global features
        global_features = torch.cat(global_features_list, dim=1)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            u=global_features
        )


def main():
    """Main execution function."""
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"
    output_dir = "./graph_vector_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    max_materials = 100  
    
    print("Initializing Topological Material Analyzer...")
    analyzer = TopologicalMaterialAnalyzer(csv_path=csv_path, db_patah = db_path)
    
    if not analyzer.formula_lookup:
        print("ERROR: No materials loaded from local database. Please check the CSV file.")
        return
    
    jids_to_process = get_jarvis_jids(max_jids=max_materials)
    
    print(f"\nStarting data generation for up to {len(jids_to_process)} materials...")
    print(f"Looking for matches in local database with {len(analyzer.formula_lookup)} materials...")
    
    successful_generations = 0
    
    for i, jid in enumerate(tqdm(jids_to_process, desc="Processing JIDs")):
        output_path = os.path.join(output_dir, f"{jid}.pt")
        
        if os.path.exists(output_path):
            successful_generations += 1
            continue
        
        material_block = analyzer.generate_data_block_with_sg_check(jid)
        
        if material_block:
            torch.save(material_block, output_path)
            successful_generations += 1
            
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
        
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i+1}/{len(jids_to_process)} processed, {successful_generations} successful, {analyzer.matched_count} matches found")
        
        time.sleep(0.1)
    
    print(f"\n" + "="*60)
    print(f"DATASET GENERATION COMPLETE")
    print(f"="*60)
    print(f"Total JIDs processed: {analyzer.processed_count}")
    print(f"Matches found in local DB: {analyzer.matched_count}")
    print(f"Successful data blocks: {successful_generations}")
    print(f"Success rate: {successful_generations/analyzer.processed_count*100:.1f}%")
    print(f"Dataset location: {output_dir}")
    print(f"="*60)

if __name__ == "__main__":
    main()