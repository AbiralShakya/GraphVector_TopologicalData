# jarvis data
# graph 1 for poscar
# graph 2 for k space
# point cloud for atom homology

import torch
import pandas as pd
import os
import re
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, List, Tuple
import json
import time

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from torch_geometric.data import Data
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy
from sklearn.preprocessing import StandardScaler
import sqlite3

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
        # This part for loading the CSV remains the same
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        for idx, row in df.iterrows():
            norm_formula = self.normalize_formula(row.get('Formula', ''))
            if norm_formula:
                self.formula_lookup[norm_formula] = row.to_dict()
        print(f"Loaded {len(self.formula_lookup)} unique materials from local CSV.")

        # --- NEW: Build vocabularies by querying the SQLite DB ---
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

    def _get_kspace_data_from_db(self, space_group_number: int) -> Optional[pd.DataFrame]:
        """
        Queries the SQLite DB to get all k-points and irrep labels
        for a given space group number.
        """
        try:
            query = """
                SELECT
                  ir.k_point,
                  ir.irrep_label
                FROM irreps AS ir
                LEFT JOIN ebr ON ir.ebr_id = ebr.id
                LEFT JOIN space_groups AS sg ON ebr.space_group_id = sg.id
                WHERE
                  sg.number = ?;
            """
            # Use pandas to execute the query and return a DataFrame
            df = pd.read_sql_query(query, self.conn, params=(space_group_number,))
            if not df.empty:
                return df
        except Exception as e:
            print(f"  ✗ Warning: Could not query k-space data for SG {space_group_number}: {e}")
        return None
    
    def generate_data_block_with_sg_check(self, jid: str) -> Optional[Dict]:
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
        
        # Get the local data entry first
        local_data_entry = self.formula_lookup[norm_formula]
        
        # Debug: Check what keys are available in jid_data
        print(f"  Available keys in jid_data: {list(jid_data.keys())}")
        
        # Create structure - try different possible keys
        structure = None
        structure_keys_to_try = ['atoms', 'structure', 'final_str', 'initial_structure']
        
        for key in structure_keys_to_try:
            if key in jid_data:
                try:
                    print(f"  Trying to create structure from key: {key}")
                    if isinstance(jid_data[key], dict):
                        print(f"    Structure data keys: {list(jid_data[key].keys())}")
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
                        print(f"    Lattice: {lattice.abc}")
                        print(f"    Elements: {elements}")
                        print(f"    Coordinates type: {'cartesian' if coords_are_cartesian else 'fractional'}")
                
                elif 'lattice_mat' in jid_data and 'coords' in jid_data and 'elements' in jid_data:
                    from pymatgen.core import Lattice
                    lattice = Lattice(jid_data['lattice_mat'])
                    coords = jid_data['coords']
                    elements = jid_data['elements']
                    structure = Structure(lattice, elements, coords)
                    print(f"  ✓ Structure built from top-level JARVIS data: {len(structure)} atoms")
                    
            except Exception as e:
                print(f"    Manual structure building failed: {e}")
                print(f"    Error details: {type(e).__name__}: {str(e)}")
                
                if 'atoms' in jid_data:
                    atoms_data = jid_data['atoms']
                    print(f"    Debug - lattice_mat shape: {np.array(atoms_data.get('lattice_mat', [])).shape if 'lattice_mat' in atoms_data else 'missing'}")
                    print(f"    Debug - coords shape: {np.array(atoms_data.get('coords', [])).shape if 'coords' in atoms_data else 'missing'}")
                    print(f"    Debug - elements length: {len(atoms_data.get('elements', [])) if 'elements' in atoms_data else 'missing'}")
                    print(f"    Debug - cartesian flag: {atoms_data.get('cartesian', 'missing')}")
        
        if structure is None:
            print(f"  ERROR: Could not create structure for {jid}")
            return None

        # Check space group match
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number_from_jarvis = analyzer.get_space_group_number()
            
            # Check if space groups match (assuming your CSV has 'Spacegroup_Number' column)
            local_sg = local_data_entry.get('Spacegroup_Number')
            if local_sg and local_sg != sg_number_from_jarvis:
                print(f"  Skipping {jid} ({norm_formula}, SG {sg_number_from_jarvis}): Space group mismatch with local DB (local: {local_sg})")
                return None
                
        except Exception as e:
            print(f"  Error in space group analysis: {e}")
            return None

        self.matched_count += 1
        print(f"  ✓ Match found! ({self.matched_count}/{self.processed_count})")
        
        # Extract Asymmetric Unit & Symmetry
        try:
            space_group = analyzer.get_space_group_symbol()
            print(f"  Space group: {space_group}")
            
            # Get primitive structure for graph generation
            primitive_structure = analyzer.get_primitive_standard_structure()
            symm_ops = analyzer.get_symmetry_operations(cartesian=True)
            print(f"  Symmetry operations: {len(symm_ops)}")
            
        except Exception as e:
            print(f"  Error in symmetry analysis: {e}")
            primitive_structure = structure
            symm_ops = []
        
        print("  Querying SQLite DB for k-space data...")
        kspace_df = self._get_kspace_data_from_db(sg_number_from_jarvis)
        if kspace_df is None:
            print(f"  Skipping {jid}: No k-space data found for SG {sg_number_from_jarvis}")
            return None
        
        print(f"  ✓ Found {len(kspace_df)} k-space entries for SG {sg_number_from_jarvis}")
        
        # Now call the featurizer with the structured data
        band_rep_vector, target_y = self._featurize_kspace_data(local_data_entry, kspace_df)
        # --- End of major changes ---

        # The rest of the generation process uses these outputs
        print("  Generating features...")
        crystal_graph = self._create_crystal_graph(primitive_structure)
        asph_vector = self._compute_asph_features(structure)
        kspace_graph = self._create_kspace_graph(band_rep_vector) # This now gets the clean vector
        
        crystal_graph.y = target_y # Set the correct label
        
        # Assemble the final data block
        data_block = {
            'jid': jid,
            'formula': formula,
            'normalized_formula': norm_formula,
            'crystal_graph': crystal_graph,
            'kspace_graph': kspace_graph,
            'asph_features': asph_vector,
            'band_rep_features': band_rep_vector,
            'target_label': target_y,
            'symmetry_ops': {
                'rotations': torch.stack([torch.tensor(op.rotation_matrix, dtype=torch.float) for op in symm_ops]) if symm_ops else torch.empty(0, 3, 3),
                'translations': torch.stack([torch.tensor(op.translation_vector, dtype=torch.float) for op in symm_ops]) if symm_ops else torch.empty(0, 3)
            },
            'jarvis_props': {
                'formation_energy': jid_data.get('form_energy_per_atom'),
                'band_gap': jid_data.get('optb88vdw_bandgap'),
                'space_group': space_group if 'space_group' in locals() else None
            },
            'local_topo_data': local_data_entry
        }
        
        print(f"  ✓ Data block generated successfully")
        return data_block

    def _get_elemental_features(self, atomic_number: int) -> torch.Tensor:
        """Creates a feature vector for a given element using basic atomic properties."""
        # Basic atomic properties - you can expand this with more sophisticated features
        # For now, using atomic number, period, and group as basic features
        periods = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3}
        period = periods.get(atomic_number, 4)  # Default to period 4 for higher elements
        
        # Simple features: atomic number, period, group approximation
        features = [
            atomic_number / 100.0,  # Normalized atomic number
            period / 8.0,           # Normalized period
            (atomic_number % 18) / 18.0  # Rough group approximation
        ]
        
        return torch.tensor(features, dtype=torch.float)

    def _compute_asph_features(self, structure: Structure, n_bins=20) -> torch.Tensor:
        """Computes Atom-Specific Persistent Homology (ASPH) feature vector."""
        try:
            # Get atomic coordinates
            coords = np.array(structure.cart_coords)
            
            # Reshape for giotto-tda (expects shape: n_samples, n_points, n_features)
            points = coords.reshape(1, -1, 3)
            
            # Compute persistent homology
            vr = VietorisRipsPersistence(
                homology_dimensions=[0, 1, 2], 
                max_edge_length=10.0,
                n_jobs=1
            )
            diagrams = vr.fit_transform(points)
            
            # Convert to Betti curves
            bc = BettiCurve(n_bins=n_bins)
            betti_curves = bc.fit_transform(diagrams)
            
            # Also compute persistence entropy for additional topological features
            pe = PersistenceEntropy()
            entropy_features = pe.fit_transform(diagrams)
            
            # Combine features
            combined_features = np.concatenate([
                betti_curves.flatten(),
                entropy_features.flatten()
            ])
            
            return torch.tensor(combined_features, dtype=torch.float)
            
        except Exception as e:
            print(f"Error computing ASPH features: {e}")
            # Return zero vector if computation fails
            return torch.zeros(n_bins * 3 + 3, dtype=torch.float)  # 3 homology dims + 3 entropy features
        
    def _featurize_kspace_data(self, local_data_entry: dict, kspace_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts structured k-space data from the DB into a feature vector
        and gets the classification label from the local CSV data.
        """
        # --- Part 1: Determine the target label (same as before) ---
        # This still comes from your high-level 'Property' column in the CSV
        property_str = str(local_data_entry.get('Property', '')).upper()
        is_topological = any(indicator in property_str for indicator in ['TI', 'SM', 'WEYL', 'DIRAC'])
        target_y = torch.tensor([1.0 if is_topological else 0.0], dtype=torch.float)

        # --- Part 2: Build the feature vector from the database query ---
        feature_dim = len(self.master_k_points) + len(self.master_irreps)
        band_rep_vector = torch.zeros(feature_dim, dtype=torch.float)

        if kspace_df is not None:
            # Create sets for fast lookups
            present_k_points = set(kspace_df['k_point'])
            present_irreps = set(kspace_df['irrep_label'])

            # Populate the vector based on the queried data
            for i, k_point in enumerate(self.master_k_points):
                if k_point in present_k_points:
                    band_rep_vector[i] = 1.0
            
            offset = len(self.master_k_points)
            for i, irrep in enumerate(self.master_irreps):
                if irrep in present_irreps:
                    band_rep_vector[offset + i] = 1.0
        
        return band_rep_vector, target_y

    def _featurize_bilbao_data(self, local_data_entry: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts raw topological data into numerical vectors."""
        try:
            # Initialize feature vector
            feature_dim = len(self.master_k_points) + len(self.master_irreps)
            band_rep_vector = torch.zeros(feature_dim, dtype=torch.float)
            
            # Look for topological data in various columns
            topological_data = ""
            for col in ['Property', 'BandReps', 'Band_Representations', 'Topology']:
                if col in local_data_entry and pd.notna(local_data_entry[col]):
                    topological_data += str(local_data_entry[col]) + " "
            
            # Parse k-points and irreps from the data
            if topological_data:
                # Count occurrences of k-points and irreps
                for i, k_point in enumerate(self.master_k_points):
                    if k_point in topological_data:
                        band_rep_vector[i] = 1.0
                
                for i, irrep in enumerate(self.master_irreps):
                    if irrep in topological_data:
                        band_rep_vector[len(self.master_k_points) + i] = 1.0
            
            # Create binary classification target
            property_str = topological_data.upper()
            
            # Look for topological indicators
            topological_indicators = ['TI', 'SM', 'ESFD', 'TOPOLOGICAL', 'NONTRIVIAL', 'WEYL', 'DIRAC']
            trivial_indicators = ['TRIVIAL', 'NORMAL', 'INSULATOR']
            
            is_topological = any(indicator in property_str for indicator in topological_indicators)
            is_trivial = any(indicator in property_str for indicator in trivial_indicators)
            
            if is_topological and not is_trivial:
                binary_label = 1.0  # Topological
            elif is_trivial and not is_topological:
                binary_label = 0.0  # Trivial
            else:
                # Default classification based on any non-empty topological data
                binary_label = 1.0 if topological_data.strip() else 0.0
            
            return band_rep_vector, torch.tensor([binary_label], dtype=torch.float)
            
        except Exception as e:
            print(f"Error featurizing Bilbao data: {e}")
            # Return default vectors
            feature_dim = len(self.master_k_points) + len(self.master_irreps)
            return torch.zeros(feature_dim, dtype=torch.float), torch.tensor([0.0], dtype=torch.float)

    def _create_crystal_graph(self, structure: Structure) -> Data:
        """Generates the real-space atomic graph using Voronoi tessellation."""
        try:
            vnn = VoronoiNN(cutoff=13.0, allow_pathological=True, tol=0.8)
            
            edge_index_list = []
            edge_weight_list = []
            
            # Get all neighbors for each site
            for i, site in enumerate(structure):
                try:
                    neighbors = vnn.get_nn_info(structure, i)
                    for neighbor_info in neighbors:
                        j = neighbor_info['site_index']
                        if i != j:  # Avoid self-loops
                            edge_index_list.append([i, j])
                            edge_weight_list.append(neighbor_info['weight'])
                except Exception as e:
                    # Skip problematic sites
                    continue
            
            # Create node features
            node_features = []
            for site in structure:
                atomic_number = site.specie.Z
                node_features.append(self._get_elemental_features(atomic_number))
            
            node_features = torch.stack(node_features) if node_features else torch.empty(0, 3)
            node_positions = torch.tensor(structure.cart_coords, dtype=torch.float)
            
            # Create edge tensors
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
            # Return minimal graph with just node features
            node_features = torch.stack([self._get_elemental_features(site.specie.Z) for site in structure])
            return Data(
                x=node_features,
                pos=torch.tensor(structure.cart_coords, dtype=torch.float),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 1)
            )

    def _create_kspace_graph(self, band_rep_vector: torch.Tensor) -> Data:
        """Generates the reciprocal-space k-point graph."""
        # Create a simple graph where each k-point is a node
        n_k_points = len(self.master_k_points)
        n_irreps = len(self.master_irreps)
        
        # Node features: k-point features + associated irrep information
        node_features = []
        for i in range(n_k_points):
            # Basic k-point feature (one-hot encoding)
            k_point_feature = torch.zeros(n_k_points)
            k_point_feature[i] = 1.0
            
            # Associated irrep features for this k-point
            irrep_features = band_rep_vector[n_k_points:]  # Irrep part of the vector
            
            # Combine features
            combined_feature = torch.cat([k_point_feature, irrep_features])
            node_features.append(combined_feature)
        
        if node_features:
            node_features = torch.stack(node_features)
        else:
            node_features = torch.empty(0, n_k_points + n_irreps)
        
        # Create a simple connectivity (can be enhanced with actual k-space connectivity)
        edge_index = []
        for i in range(n_k_points):
            for j in range(i+1, n_k_points):
                edge_index.extend([[i, j], [j, i]])  # Bidirectional edges
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            global_features=band_rep_vector.unsqueeze(0)  # Store full band rep as global feature
        )
    

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

def main():
    """Main execution function."""
    # --- Configuration ---
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"
    output_dir = "./graph_vector_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    max_materials = 100  # Start with a smaller number for testing
    
    # --- Initialize Analyzer ---
    print("Initializing Topological Material Analyzer...")
    analyzer = TopologicalMaterialAnalyzer(csv_path=csv_path, db_patah = db_path)
    
    if not analyzer.formula_lookup:
        print("ERROR: No materials loaded from local database. Please check the CSV file.")
        return
    
    # --- Get JIDs to process ---
    jids_to_process = get_jarvis_jids(max_jids=max_materials)
    
    # --- Main Processing Loop ---
    print(f"\nStarting data generation for up to {len(jids_to_process)} materials...")
    print(f"Looking for matches in local database with {len(analyzer.formula_lookup)} materials...")
    
    successful_generations = 0
    
    for i, jid in enumerate(tqdm(jids_to_process, desc="Processing JIDs")):
        output_path = os.path.join(output_dir, f"{jid}.pt")
        
        # Skip if already processed
        if os.path.exists(output_path):
            successful_generations += 1
            continue
        
        # Process the material
        material_block = analyzer.generate_data_block_with_sg_check(jid)
        
        if material_block:
            # Save the data block
            torch.save(material_block, output_path)
            successful_generations += 1
            
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
        
        # Print progress every 10 materials
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i+1}/{len(jids_to_process)} processed, {successful_generations} successful, {analyzer.matched_count} matches found")
        
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    # --- Final Report ---
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