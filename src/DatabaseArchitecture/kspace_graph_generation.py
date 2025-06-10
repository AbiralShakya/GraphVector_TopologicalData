
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import sqlite3
from pathlib import Path
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import pickle

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
    
class KSpacePhysicsGraphBuilder:
    """
    Builds physics-informed k-space graphs from topological material data
    """
    
    def __init__(self, csv_path: str, db_path: str, output_dir: str = "./kspace_graphs"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each space group
        for sg in range(1, 231):
            (self.output_dir / f"SG_{sg:03d}").mkdir(exist_ok=True)
    
    def _get_kspace_data_from_db(self, space_group_number: int) -> pd.DataFrame:
        """
        Get k-space data for a specific space group from database by joining
        the new table names: space_groups, ebr_decomposition_branches, irreps, ebrs.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # SQL query to join the tables
            query = """
                SELECT
                  ir.k_point,
                  ir.irrep_label,
                  b.branch1_irreps,        -- NEW: Irreps for decomposition branch 1
                  b.branch2_irreps,        -- NEW: Irreps for decomposition branch 2
                  b.decomposition_index    -- NEW: Index of the specific EBR decomposition
                FROM irreps AS ir
                LEFT JOIN ebrs ON ir.ebr_id = ebrs.id
                LEFT JOIN space_groups AS sg ON ebrs.space_group_id = sg.id
                LEFT JOIN ebr_decomposition_branches AS b ON ebrs.id = b.ebr_id -- NEW JOIN to get branch info
                WHERE
                  sg.number = ?;
            """
            
            df = pd.read_sql_query(query, conn, params=(space_group_number,))
            conn.close()
            
            if df.empty:
                print(f"No k-space data found for space group {space_group_number}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error querying database for SG {space_group_number}: {e}")
            return pd.DataFrame()
    
    def _extract_ebr_features(self, kspace_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract Elementary Band Representation features from k-space data
        """
        ebr_features = {}
        
        # Extract EBR decomposition data
        if 'ebr_label' in kspace_df.columns:
            ebr_labels = kspace_df['ebr_label'].dropna().unique()
            ebr_features['ebr_labels'] = ebr_labels.tolist()
            ebr_features['num_ebr'] = len(ebr_labels)
            
            # EBR multiplicity analysis
            ebr_counts = kspace_df['ebr_label'].value_counts()
            ebr_features['ebr_multiplicities'] = ebr_counts.to_dict()
        
        # Extract irreducible representations
        if 'irrep' in kspace_df.columns or 'irrep_label' in kspace_df.columns:
            irrep_col = 'irrep' if 'irrep' in kspace_df.columns else 'irrep_label'
            irreps = kspace_df[irrep_col].dropna().unique()
            ebr_features['irreps'] = irreps.tolist()
            ebr_features['irrep_multiplicities'] = kspace_df[irrep_col].value_counts().to_dict()
        
        # Process branch information if available
        if 'branch1_irreps' in kspace_df.columns:
            branch1_data = kspace_df['branch1_irreps'].dropna()
            if len(branch1_data) > 0:
                ebr_features['branch1_irreps'] = branch1_data.tolist()
        
        if 'branch2_irreps' in kspace_df.columns:
            branch2_data = kspace_df['branch2_irreps'].dropna()
            if len(branch2_data) > 0:
                ebr_features['branch2_irreps'] = branch2_data.tolist()
        
        # Wyckoff position information
        if 'wyckoff_position' in kspace_df.columns:
            wyckoff_positions = kspace_df['wyckoff_position'].dropna().unique()
            ebr_features['wyckoff_positions'] = wyckoff_positions.tolist()
        
        # Site symmetry data
        if 'site_symmetry' in kspace_df.columns:
            site_symmetries = kspace_df['site_symmetry'].dropna().unique()
            ebr_features['site_symmetries'] = site_symmetries.tolist()
        
        return ebr_features
    
    def _extract_topological_indices(self, kspace_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract topological indices from k-space data
        """
        topo_indices = {}
        
        # Z2 invariants
        z2_columns = ['z2_0', 'z2_1', 'z2_2', 'z2_3']
        for col in z2_columns:
            if col in kspace_df.columns:
                # Take the most common value or mean for continuous values
                values = kspace_df[col].dropna()
                if len(values) > 0:
                    if values.dtype in ['int64', 'int32', 'bool']:
                        topo_indices[col] = int(values.mode().iloc[0] if not values.mode().empty else 0)
                    else:
                        topo_indices[col] = float(values.mean())
        
        # Chern numbers
        if 'chern_number' in kspace_df.columns:
            chern_values = kspace_df['chern_number'].dropna()
            if len(chern_values) > 0:
                topo_indices['chern_number'] = int(chern_values.mode().iloc[0] if not chern_values.mode().empty else 0)
        
        # Mirror Chern numbers
        mirror_chern_cols = [col for col in kspace_df.columns if 'mirror_chern' in col.lower()]
        for col in mirror_chern_cols:
            values = kspace_df[col].dropna()
            if len(values) > 0:
                topo_indices[col] = float(values.mean())
        
        # Topological crystalline invariants
        if 'tci' in kspace_df.columns:
            tci_values = kspace_df['tci'].dropna()
            if len(tci_values) > 0:
                topo_indices['tci'] = float(tci_values.mean())
        
        # Berry curvature related
        berry_cols = [col for col in kspace_df.columns if 'berry' in col.lower()]
        for col in berry_cols:
            values = kspace_df[col].dropna()
            if len(values) > 0:
                topo_indices[col] = float(values.mean())
        
        return topo_indices
    
    def _extract_decomposition_branches(self, kspace_df: pd.DataFrame) -> Dict[str, List]:
        """
        Extract band decomposition branch information
        """
        decomp_branches = {}
        
        # Process decomposition indices
        if 'decomposition_index' in kspace_df.columns:
            decomp_indices = sorted(kspace_df['decomposition_index'].dropna().unique())
            decomp_branches['decomposition_indices'] = decomp_indices
        
        # Band indices and their decomposition
        if 'band_index' in kspace_df.columns:
            band_indices = sorted(kspace_df['band_index'].dropna().unique())
            decomp_branches['band_indices'] = band_indices
        
        # Energy levels and band connectivity
        if 'energy' in kspace_df.columns and ('kpoint' in kspace_df.columns or 'k_point' in kspace_df.columns):
            kpoint_col = 'kpoint' if 'kpoint' in kspace_df.columns else 'k_point'
            # Group by k-points to understand band connectivity
            kpoint_groups = kspace_df.groupby(kpoint_col)
            
            band_connectivity = {}
            for kpoint, group in kpoint_groups:
                if 'band_index' in group.columns and 'energy' in group.columns:
                    # Sort bands by energy at this k-point
                    sorted_bands = group.sort_values('energy')
                    band_connectivity[str(kpoint)] = {
                        'band_indices': sorted_bands['band_index'].tolist(),
                        'energies': sorted_bands['energy'].tolist()
                    }
            
            decomp_branches['band_connectivity'] = band_connectivity
        
        # Irrep decomposition per band
        irrep_col = 'irrep' if 'irrep' in kspace_df.columns else 'irrep_label'
        if 'band_index' in kspace_df.columns and irrep_col in kspace_df.columns:
            band_irrep_decomp = {}
            for band_idx in kspace_df['band_index'].dropna().unique():
                band_data = kspace_df[kspace_df['band_index'] == band_idx]
                irreps = band_data[irrep_col].dropna().tolist()
                band_irrep_decomp[str(int(band_idx))] = irreps
            
            decomp_branches['band_irrep_decomposition'] = band_irrep_decomp
        
        # Character decomposition
        char_cols = [col for col in kspace_df.columns if 'character' in col.lower()]
        if char_cols:
            char_decomp = {}
            for col in char_cols:
                values = kspace_df[col].dropna().tolist()
                if values:
                    char_decomp[col] = values
            decomp_branches['character_decomposition'] = char_decomp
        
        return decomp_branches
    
    def _build_kspace_connectivity_graph(self, kspace_df: pd.DataFrame, 
                                       ebr_features: Dict, 
                                       topo_indices: Dict,
                                       decomp_branches: Dict) -> Tuple[Data, np.ndarray]:
        """
        Build physics-informed k-space connectivity graph - FIXED VERSION
        """
        # Create nodes based on k-points and their properties
        nodes = []
        node_features = []
        kpoint_to_idx = {}
        
        # Process k-points as nodes - handle both 'kpoint' and 'k_point' columns
        kpoint_col = None
        if 'kpoint' in kspace_df.columns:
            kpoint_col = 'kpoint'
        elif 'k_point' in kspace_df.columns:
            kpoint_col = 'k_point'
        
        if kpoint_col:
            unique_kpoints = kspace_df[kpoint_col].dropna().unique()
            
            for idx, kpoint in enumerate(unique_kpoints):
                kpoint_to_idx[kpoint] = idx
                nodes.append(kpoint)
                
                # Extract features for this k-point
                kpoint_data = kspace_df[kspace_df[kpoint_col] == kpoint]
                features = self._extract_kpoint_features(kpoint_data, ebr_features, topo_indices)
                node_features.append(features)
        
        # If no k-points, create nodes based on irreps or EBR labels
        elif 'irrep' in kspace_df.columns or 'irrep_label' in kspace_df.columns:
            irrep_col = 'irrep' if 'irrep' in kspace_df.columns else 'irrep_label'
            unique_irreps = kspace_df[irrep_col].dropna().unique()
            
            for idx, irrep in enumerate(unique_irreps):
                kpoint_to_idx[irrep] = idx
                nodes.append(irrep)
                
                irrep_data = kspace_df[kspace_df[irrep_col] == irrep]
                features = self._extract_irrep_features(irrep_data, ebr_features, topo_indices)
                node_features.append(features)
        
        # Fallback: create nodes based on decomposition indices
        elif 'decomposition_index' in kspace_df.columns:
            unique_decomp = kspace_df['decomposition_index'].dropna().unique()
            
            for idx, decomp_idx in enumerate(unique_decomp):
                kpoint_to_idx[decomp_idx] = idx
                nodes.append(f"decomp_{decomp_idx}")
                
                decomp_data = kspace_df[kspace_df['decomposition_index'] == decomp_idx]
                features = self._extract_decomp_features(decomp_data, ebr_features, topo_indices)
                node_features.append(features)
        
        # Ensure we have at least some nodes
        if not nodes:
            print("Warning: No nodes could be created, creating default node")
            nodes = ["default_node"]
            node_features = [[1.0] * 10]
            kpoint_to_idx = {"default_node": 0}
        
        print(f"  Created {len(nodes)} nodes: {nodes[:5]}..." if len(nodes) > 5 else f"  Created {len(nodes)} nodes: {nodes}")
        
        # Create edges based on physical connectivity
        edge_indices = []
        edge_attributes = []
        
        # Method 1: Band connectivity edges
        if 'band_connectivity' in decomp_branches and decomp_branches['band_connectivity']:
            print("  Creating band connectivity edges...")
            connectivity = decomp_branches['band_connectivity']
            edges, edge_attrs = self._create_band_connectivity_edges(
                connectivity, kpoint_to_idx, kspace_df
            )
            edge_indices.extend(edges)
            edge_attributes.extend(edge_attrs)
            print(f"    Added {len(edges)} band connectivity edges")
        
        # Method 2: Symmetry-based edges
        print("  Creating symmetry-based edges...")
        symmetry_edges, symmetry_attrs = self._create_symmetry_edges(
            kspace_df, kpoint_to_idx, ebr_features
        )
        edge_indices.extend(symmetry_edges)
        edge_attributes.extend(symmetry_attrs)
        print(f"    Added {len(symmetry_edges)} symmetry edges")
        
        # Method 3: Topological proximity edges
        print("  Creating topological proximity edges...")
        topo_edges, topo_attrs = self._create_topological_edges(
            nodes, node_features, topo_indices
        )
        edge_indices.extend(topo_edges)
        edge_attributes.extend(topo_attrs)
        print(f"    Added {len(topo_edges)} topological edges")
        
        # Method 4: If still no edges, create a basic connectivity structure
        if not edge_indices and len(nodes) > 1:
            print("  No edges found, creating basic linear connectivity...")
            for i in range(len(nodes) - 1):
                edge_indices.append([i, i + 1])
                edge_attributes.append([1.0])
            print(f"    Added {len(edge_indices)} basic connectivity edges")
        
        # Convert to tensors
        if not node_features:
            # Fallback: create minimal features
            node_features = [[1.0] * 10 for _ in nodes]
        
        # Ensure we have valid tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float32) if edge_attributes else None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        
        print(f"  Final graph: {len(nodes)} nodes, {len(edge_indices)} edges")
        
        # Create connectivity matrix - FIXED VERSION
        num_nodes = len(nodes)
        connectivity_matrix = np.zeros((num_nodes, num_nodes))
        
        if edge_indices:
            for edge in edge_indices:
                if len(edge) >= 2:
                    i, j = int(edge[0]), int(edge[1])
                    if 0 <= i < num_nodes and 0 <= j < num_nodes:
                        connectivity_matrix[i, j] = 1
                        connectivity_matrix[j, i] = 1  # Undirected graph
        
        # Debug: Print connectivity matrix info
        total_connections = np.sum(connectivity_matrix)
        print(f"  Connectivity matrix: {num_nodes}x{num_nodes}, total connections: {int(total_connections)}")
        if total_connections == 0:
            print("  WARNING: Connectivity matrix is all zeros!")
        
        # Create PyG Data object
        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Add global features
        sg_num = kspace_df.iloc[0].get('space_group_number', 0) if not kspace_df.empty else 0
        num_bands = len(kspace_df['band_index'].dropna().unique()) if 'band_index' in kspace_df.columns else 0
        
        pyg_data.space_group = torch.tensor([sg_num])
        pyg_data.num_bands = torch.tensor([num_bands])
        
        return pyg_data, connectivity_matrix
    
    def _extract_kpoint_features(self, kpoint_data: pd.DataFrame, 
                               ebr_features: Dict, 
                               topo_indices: Dict) -> List[float]:
        """Extract features for a specific k-point"""
        features = []
        
        # Energy-related features
        if 'energy' in kpoint_data.columns:
            energies = kpoint_data['energy'].dropna()
            if len(energies) > 0:
                features.extend([
                    float(energies.mean()),
                    float(energies.std() if len(energies) > 1 else 0),
                    float(energies.min()),
                    float(energies.max())
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Band count at this k-point
        if 'band_index' in kpoint_data.columns:
            num_bands = len(kpoint_data['band_index'].dropna().unique())
            features.append(float(num_bands))
        else:
            features.append(0.0)
        
        # Irrep diversity
        irrep_col = 'irrep' if 'irrep' in kpoint_data.columns else 'irrep_label'
        if irrep_col in kpoint_data.columns:
            num_irreps = len(kpoint_data[irrep_col].dropna().unique())
            features.append(float(num_irreps))
        else:
            features.append(0.0)
        
        # EBR participation
        if 'ebr_label' in kpoint_data.columns:
            num_ebr = len(kpoint_data['ebr_label'].dropna().unique())
            features.append(float(num_ebr))
        else:
            features.append(0.0)
        
        # Topological contribution (averaged local topological indices)
        topo_contribution = 0.0
        topo_count = 0
        for key, value in topo_indices.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                topo_contribution += abs(float(value))
                topo_count += 1
        
        if topo_count > 0:
            features.append(topo_contribution / topo_count)
        else:
            features.append(0.0)
        
        # Symmetry indicator
        if 'site_symmetry' in kpoint_data.columns:
            site_syms = kpoint_data['site_symmetry'].dropna().unique()
            features.append(float(len(site_syms)))
        else:
            features.append(0.0)
        
        # Pad to fixed size
        target_size = 10
        while len(features) < target_size:
            features.append(0.0)
        
        return features[:target_size]
    
    def _extract_irrep_features(self, irrep_data: pd.DataFrame,
                              ebr_features: Dict,
                              topo_indices: Dict) -> List[float]:
        """Extract features for irreducible representations"""
        features = []
        
        # Similar to k-point features but focused on irrep properties
        if 'energy' in irrep_data.columns:
            energies = irrep_data['energy'].dropna()
            if len(energies) > 0:
                features.extend([energies.mean(), energies.std() if len(energies) > 1 else 0])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Multiplicity
        features.append(float(len(irrep_data)))
        
        # Band participation
        if 'band_index' in irrep_data.columns:
            features.append(float(len(irrep_data['band_index'].dropna().unique())))
        else:
            features.append(0.0)
        
        # Decomposition index info
        if 'decomposition_index' in irrep_data.columns:
            features.append(float(len(irrep_data['decomposition_index'].dropna().unique())))
        else:
            features.append(0.0)
        
        # Branch information
        if 'branch1_irreps' in irrep_data.columns:
            features.append(float(len(irrep_data['branch1_irreps'].dropna())))
        else:
            features.append(0.0)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def _extract_decomp_features(self, decomp_data: pd.DataFrame,
                               ebr_features: Dict,
                               topo_indices: Dict) -> List[float]:
        """Extract features for decomposition indices"""
        features = []
        
        # Basic statistics
        features.append(float(len(decomp_data)))  # Number of entries for this decomposition
        
        # K-point diversity
        kpoint_col = 'k_point' if 'k_point' in decomp_data.columns else 'kpoint'
        if kpoint_col in decomp_data.columns:
            features.append(float(len(decomp_data[kpoint_col].dropna().unique())))
        else:
            features.append(0.0)
        
        # Irrep diversity
        irrep_col = 'irrep_label' if 'irrep_label' in decomp_data.columns else 'irrep'
        if irrep_col in decomp_data.columns:
            features.append(float(len(decomp_data[irrep_col].dropna().unique())))
        else:
            features.append(0.0)
        
        # Branch information
        if 'branch1_irreps' in decomp_data.columns:
            features.append(float(len(decomp_data['branch1_irreps'].dropna())))
        else:
            features.append(0.0)
        
        if 'branch2_irreps' in decomp_data.columns:
            features.append(float(len(decomp_data['branch2_irreps'].dropna())))
        else:
            features.append(0.0)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def _create_band_connectivity_edges(self, connectivity: Dict, 
                                      kpoint_to_idx: Dict, 
                                      kspace_df: pd.DataFrame) -> Tuple[List, List]:
        """Create edges based on band connectivity between k-points"""
        edges = []
        edge_attrs = []
        
        kpoints = list(connectivity.keys())
        
        # Connect adjacent k-points in the path
        for i in range(len(kpoints) - 1):
            kp1, kp2 = kpoints[i], kpoints[i + 1]
            
            if kp1 in kpoint_to_idx and kp2 in kpoint_to_idx:
                idx1, idx2 = kpoint_to_idx[kp1], kpoint_to_idx[kp2]
                
                # Calculate edge weight based on band overlap or energy difference
                conn1 = connectivity[kp1]
                conn2 = connectivity[kp2]
                
                if 'energies' in conn1 and 'energies' in conn2:
                    # Energy difference as edge weight
                    energy_diff = abs(np.mean(conn1['energies']) - np.mean(conn2['energies']))
                    edge_weight = 1.0 / (1.0 + energy_diff)  # Inverse relationship
                else:
                    edge_weight = 1.0
                
                edges.append([idx1, idx2])
                edge_attrs.append([edge_weight])
        
        return edges, edge_attrs
    
    def _create_symmetry_edges(self, kspace_df: pd.DataFrame, 
                             kpoint_to_idx: Dict, 
                             ebr_features: Dict) -> Tuple[List, List]:
        """Create edges based on symmetry relationships"""
        edges = []
        edge_attrs = []
        
        # Connect nodes with shared symmetry properties
        irrep_col = 'irrep' if 'irrep' in kspace_df.columns else 'irrep_label'
        kpoint_col = 'kpoint' if 'kpoint' in kspace_df.columns else 'k_point'
        
        if irrep_col in kspace_df.columns:
            # Group by irrep and connect k-points with same irrep
            irrep_groups = kspace_df.groupby(irrep_col)
            
            for irrep, group in irrep_groups:
                if kpoint_col in group.columns:
                    kpoints_with_irrep = group[kpoint_col].dropna().unique()
                    
                    # Create clique among k-points with same irrep
                    for i, kp1 in enumerate(kpoints_with_irrep):
                        for kp2 in kpoints_with_irrep[i+1:]:
                            if kp1 in kpoint_to_idx and kp2 in kpoint_to_idx:
                                idx1, idx2 = kpoint_to_idx[kp1], kpoint_to_idx[kp2]
                                edges.append([idx1, idx2])
                                edge_attrs.append([0.8])  # Symmetry connection strength
                elif irrep in kpoint_to_idx:
                    # If nodes are irreps themselves, connect based on co-occurrence
                    continue
        
        # Connect based on branch information
        if 'branch1_irreps' in kspace_df.columns or 'branch2_irreps' in kspace_df.columns:
            # Group by decomposition index and connect related entries
            if 'decomposition_index' in kspace_df.columns:
                decomp_groups = kspace_df.groupby('decomposition_index')
                
                for decomp_idx, group in decomp_groups:
                    if kpoint_col in group.columns:
                        kpoints_in_decomp = group[kpoint_col].dropna().unique()
                        
                        # Connect k-points within same decomposition
                        for i, kp1 in enumerate(kpoints_in_decomp):
                            for kp2 in kpoints_in_decomp[i+1:]:
                                if kp1 in kpoint_to_idx and kp2 in kpoint_to_idx:
                                    idx1, idx2 = kpoint_to_idx[kp1], kpoint_to_idx[kp2]
                                    edges.append([idx1, idx2])
                                    edge_attrs.append([0.6])  # Decomposition connection strength
        
        return edges, edge_attrs
    
    def _create_topological_edges(self, nodes: List, 
                                node_features: List, 
                                topo_indices: Dict) -> Tuple[List, List]:
        """Create edges based on topological proximity"""
        edges = []
        edge_attrs = []
        
        if len(nodes) < 2 or not node_features:
            return edges, edge_attrs
        
        # Calculate feature similarity matrix
        features_array = np.array(node_features)
        
        if features_array.shape[0] > 1:
            # Use correlation or distance-based connectivity
            distances = pdist(features_array, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # Connect nodes with high similarity (low distance)
            threshold = np.percentile(distances, 30)  # Connect top 30% most similar
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if distance_matrix[i, j] <= threshold:
                        weight = 1.0 / (1.0 + distance_matrix[i, j])
                        edges.append([i, j])
                        edge_attrs.append([weight])
        
        return edges, edge_attrs
    
    def generate_kspace_graphs_for_all_space_groups(self):
        """
        Generate k-space topology graphs for all space groups 1-230
        """
        results = {}
        
        for sg in range(1, 231):
            print(f"Processing Space Group {sg}...")
            
            try:
                # Get k-space data for this space group
                kspace_df = self._get_kspace_data_from_db(sg)
                
                if kspace_df.empty:
                    print(f"  No data found for SG {sg}, skipping...")
                    continue
                
                # Extract physics features
                ebr_features = self._extract_ebr_features(kspace_df)
                topo_indices = self._extract_topological_indices(kspace_df)
                decomp_branches = self._extract_decomposition_branches(kspace_df)
                
                # Build k-space graph
                kspace_graph, connectivity_matrix = self._build_kspace_connectivity_graph(
                    kspace_df, ebr_features, topo_indices, decomp_branches
                )
                
                # Create physics features tensor
                physics_features = self._create_physics_features_tensor(
                    ebr_features, topo_indices, decomp_branches
                )
                
                # Create topology data structure
                topology_data = KSpaceTopologyData(
                    space_group_number=sg,
                    ebr_data=ebr_features,
                    topological_indices=topo_indices,
                    decomposition_branches=decomp_branches,
                    kspace_graph=kspace_graph,
                    connectivity_matrix=connectivity_matrix,
                    physics_features=physics_features
                )
                
                # Save to appropriate folder
                self._save_topology_data(topology_data, sg)
                
                results[sg] = topology_data
                print(f"  Successfully processed SG {sg} - {len(kspace_df)} records")
                
            except Exception as e:
                print(f"  Error processing SG {sg}: {e}")
                continue
        
        print(f"\nCompleted processing. Generated graphs for {len(results)} space groups.")
        return results
    
    def _create_physics_features_tensor(self, ebr_features: Dict, 
                                      topo_indices: Dict, 
                                      decomp_branches: Dict) -> Dict[str, torch.Tensor]:
        """Create physics-informed feature tensors"""
        features = {}
        
        # EBR feature tensor
        ebr_vector = []
        ebr_vector.append(ebr_features.get('num_ebr', 0))
        ebr_vector.extend([len(ebr_features.get('ebr_labels', [])),
                          len(ebr_features.get('irreps', [])),
                          len(ebr_features.get('wyckoff_positions', [])),
                          len(ebr_features.get('site_symmetries', []))])
        
        features['ebr_features'] = torch.tensor(ebr_vector, dtype=torch.float32)
        
        # Topological indices tensor
        topo_vector = []
        for key in ['z2_0', 'z2_1', 'z2_2', 'z2_3', 'chern_number']:
            topo_vector.append(topo_indices.get(key, 0))
        
        features['topological_indices'] = torch.tensor(topo_vector, dtype=torch.float32)
        
        # Decomposition features
        decomp_vector = []
        decomp_vector.append(len(decomp_branches.get('band_indices', [])))
        decomp_vector.append(len(decomp_branches.get('band_connectivity', {})))
        
        features['decomposition_features'] = torch.tensor(decomp_vector, dtype=torch.float32)
        
        return features
    
    def _save_topology_data(self, topology_data: KSpaceTopologyData, space_group: int):
        """Save topology data to appropriate space group folder"""
        sg_folder = self.output_dir / f"SG_{space_group:03d}"
        
        # Save PyTorch Geometric graph
        torch.save(topology_data.kspace_graph, sg_folder / "kspace_graph.pt")
        
        # Save connectivity matrix
        np.save(sg_folder / "connectivity_matrix.npy", topology_data.connectivity_matrix)
        
        # Save physics features
        torch.save(topology_data.physics_features, sg_folder / "physics_features.pt")
        
        # Save metadata as JSON
        metadata = {
            'space_group_number': topology_data.space_group_number,
            'ebr_data': topology_data.ebr_data,
            'topological_indices': topology_data.topological_indices,
            'decomposition_branches': topology_data.decomposition_branches,
            'graph_info': {
                'num_nodes': topology_data.kspace_graph.num_nodes,
                'num_edges': topology_data.kspace_graph.edge_index.shape[1] if topology_data.kspace_graph.edge_index.numel() > 0 else 0,
                'node_feature_dim': topology_data.kspace_graph.x.shape[1],
            }
        }
        
        with open(sg_folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save complete data structure as pickle for easy loading
        with open(sg_folder / "topology_data.pkl", 'wb') as f:
            pickle.dump(topology_data, f)
        
        print(f"  Saved topology data for SG {space_group} to {sg_folder}")

# Usage example with your actual paths
def main():
    """Main execution function"""
    
    # Your actual paths
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"
    output_dir = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs"
    
    # Initialize the graph builder
    graph_builder = KSpacePhysicsGraphBuilder(csv_path, db_path, output_dir)
    
    # Generate graphs for all space groups
    results = graph_builder.generate_kspace_graphs_for_all_space_groups()
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Generated k-space topology graphs for {len(results)} space groups")
    print(f"Output directory: {output_dir}")
    print(f"Each space group folder contains:")
    print("  - kspace_graph.pt (PyTorch Geometric graph)")
    print("  - connectivity_matrix.npy (adjacency matrix)")
    print("  - physics_features.pt (physics-informed tensors)")
    print("  - metadata.json (all extracted features)")
    print("  - topology_data.pkl (complete KSpaceTopologyData object)")
    
    return results

if __name__ == "__main__":
    main()
