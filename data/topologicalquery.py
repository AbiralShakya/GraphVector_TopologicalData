# single query handling

import torch # Retained if future extensions might use it
import torch.nn as nn # Retained if future extensions might use it
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from sklearn.manifold import UMAP
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import fields # For accessing dataclass fields
from topologicalmaterial import TopologicalMaterial

#TODO: implement with api key like in other repo

class TopologicalQueryEngine:
    """Advanced query engine for topological materials database."""

    def __init__(self, database: 'TopologicalMaterialsDatabase'):
        self.db = database
        # topology_map can be complex; for now, keeping it simple as its usage isn't fully defined in queries
        self.topology_map = self._build_topology_map()
        self.invariant_weights = self._calculate_invariant_weights()

    def _build_topology_map(self) -> Dict:
        """Build hierarchical topology classification map (can be expanded)."""
        return {
            'TI': {
                'strong': ['Z2=1', 'all_Z2w=1'], # Example criteria
                'weak': ['Z2=0', 'any_Z2w=1'],
            },
            'TSM': {
                'weyl': ['chiral_nodes'],
            }
            # Further details can be added based on how this map is used in queries.
        }

    def _calculate_invariant_weights(self) -> Dict[str, float]:
        """Calculate importance weights for different properties/invariants."""
        weights = {
            # Topological Invariants
            'Z2': 1.0,
            'Z2w_1': 0.8,
            'Z2w_2': 0.8,
            'Z2w_3': 0.8,
            'Z4': 0.9,
            'Z4pi': 0.7,
            'Z8': 0.6,
            # Structural & Electronic
            'space_group': 0.5,
            'smallest_gap': 0.6, 
            'soc_enabled': 0.4,  
            'ebr_class': 0.4,
            'fragile_bands_count': 0.3,

            'lattice_a': 0.2, 'lattice_b': 0.2, 'lattice_c': 0.2,
            'lattice_alpha': 0.1, 'lattice_beta': 0.1, 'lattice_gamma': 0.1,
        }
        return weights

    def _get_material_vector_for_similarity(self, material: 'TopologicalMaterial') -> Dict[str, Any]:
        """Helper to extract features for similarity calculation."""
        vec = {}
        vec.update(material.z2_invariants)
        vec.update(material.higher_order_invariants) 
        vec['space_group'] = material.space_group
        vec['smallest_gap'] = material.smallest_gap
        vec['soc_enabled'] = 1.0 if material.soc_enabled else 0.0
        vec['ebr_class'] = material.ebr_class
        vec['fragile_bands_count'] = material.fragile_bands

        if material.lattice_params:
            for p, v in material.lattice_params.items():
                vec[f'lattice_{p}'] = v
        return vec

    def _calculate_topological_similarity(self, mat1: 'TopologicalMaterial', mat2: 'TopologicalMaterial') -> float:
        """Calculate weighted topological and electronic similarity."""
        similarity = 0.0
        total_weight = 0.0

        vec1 = self._get_material_vector_for_similarity(mat1)
        vec2 = self._get_material_vector_for_similarity(mat2)

        # Compare Invariants and discrete properties
        discrete_props = ['Z2', 'Z2w_1', 'Z2w_2', 'Z2w_3', 'Z4', 'Z4pi', 'Z8',
                          'space_group', 'soc_enabled', 'ebr_class', 'fragile_bands_count']
        for prop_name in discrete_props:
            w = self.invariant_weights.get(prop_name, 0.0)
            if w == 0.0: continue

            val1 = vec1.get(prop_name)
            val2 = vec2.get(prop_name)

            if val1 is not None and val2 is not None:
                total_weight += w
                if val1 == val2:
                    similarity += w
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)): # Numeric diff
                    # For invariants like Z4, Z8, allow graded similarity if not exact match
                    if prop_name in ['Z4', 'Z8', 'fragile_bands_count', 'ebr_class']:
                         sim_score = 1.0 / (1.0 + abs(val1 - val2)) # Higher penalty for larger diff
                         similarity += w * sim_score
        
        # Compare continuous properties (e.g., smallest_gap)
        if mat1.smallest_gap is not None and mat2.smallest_gap is not None:
            w_gap = self.invariant_weights.get('smallest_gap', 0.0)
            total_weight += w_gap
            # Normalize difference by average gap or a characteristic energy scale
            avg_gap = (mat1.smallest_gap + mat2.smallest_gap) / 2.0 + 1e-3 # Add epsilon to avoid div by zero
            gap_similarity = 1.0 / (1.0 + abs(mat1.smallest_gap - mat2.smallest_gap) / avg_gap)
            similarity += w_gap * gap_similarity
            
        # Compare topological type (simple match)
        if mat1.topological_type and mat2.topological_type:
            total_weight += 0.5 # Default weight for type matching
            if mat1.topological_type == mat2.topological_type:
                similarity += 0.5
                # Bonus if subtype also matches
                if mat1.topological_subtype == mat2.topological_subtype:
                    similarity += 0.25 # Bonus weight
                    total_weight += 0.25

        return similarity / total_weight if total_weight > 0 else 0.0

    def _find_shared_invariants(self, mat1: 'TopologicalMaterial', mat2: 'TopologicalMaterial') -> List[str]:
        """Find shared non-trivial topological invariants."""
        shared = []
        for inv_name, value1 in mat1.z2_invariants.items():
            if value1 != 0 and mat2.z2_invariants.get(inv_name) == value1:
                shared.append(f"{inv_name}={value1}")
        for inv_name, value1 in mat1.higher_order_invariants.items():
            if value1 != 0 and mat2.higher_order_invariants.get(inv_name) == value1:
                shared.append(f"{inv_name}={value1}")
        return shared

    def _find_key_differences(self, mat1: 'TopologicalMaterial', mat2: 'TopologicalMaterial') -> List[str]:
        """Find key topological, electronic, and structural differences."""
        differences = []
        if mat1.topological_type != mat2.topological_type:
            differences.append(f"Type: '{mat1.topological_type} ({mat1.topological_subtype})' vs '{mat2.topological_type} ({mat2.topological_subtype})'")

        all_inv_keys = set(mat1.z2_invariants.keys()) | set(mat2.z2_invariants.keys()) | \
                       set(mat1.higher_order_invariants.keys()) | set(mat2.higher_order_invariants.keys())
        for inv_name in all_inv_keys:
            val1 = mat1.z2_invariants.get(inv_name, mat1.higher_order_invariants.get(inv_name))
            val2 = mat2.z2_invariants.get(inv_name, mat2.higher_order_invariants.get(inv_name))
            if val1 is None and val2 is None: continue
            val1 = val1 if val1 is not None else "N/A"
            val2 = val2 if val2 is not None else "N/A"
            if str(val1) != str(val2): # Compare as string to handle N/A
                differences.append(f"{inv_name}: {val1} vs {val2}")
        
        if mat1.space_group != mat2.space_group:
            differences.append(f"Space Group: {mat1.space_group_symbol} ({mat1.space_group}) vs {mat2.space_group_symbol} ({mat2.space_group})")
        if abs((mat1.smallest_gap or 0) - (mat2.smallest_gap or 0)) > 0.01 : # Threshold for gap difference
             differences.append(f"Smallest Gap: {mat1.smallest_gap:.3f} eV vs {mat2.smallest_gap:.3f} eV")
        if mat1.ebr_class != mat2.ebr_class:
            differences.append(f"EBR Class: {mat1.ebr_class} vs {mat2.ebr_class}")
        if mat1.fragile_bands != mat2.fragile_bands:
            differences.append(f"Fragile Bands: {mat1.fragile_bands} vs {mat2.fragile_bands}")
            
        # Basic band group representation difference (count of groups)
        if len(mat1.band_group_representations) != len(mat2.band_group_representations):
            differences.append(f"Num Band Groups: {len(mat1.band_group_representations)} vs {len(mat2.band_group_representations)}")
        return differences

    def find_topological_analogues(self, query_material_id: str,
                                 similarity_threshold: float = 0.7) -> List[Dict]:
        """Find materials with analogous topological properties.
           More detailed output with key differences and shared invariants.
        """
        if query_material_id not in self.db.materials:
            print(f"Error: Query material ID '{query_material_id}' not found in database.")
            return []

        query_material = self.db.materials[query_material_id]
        analogues = []

        for mat_id, material in self.db.materials.items():
            if mat_id == query_material_id:
                continue

            similarity_score = self._calculate_topological_similarity(query_material, material)

            if similarity_score >= similarity_threshold:
                analogues.append({
                    'material_id': mat_id,
                    'formula': material.formula,
                    'similarity_score': round(similarity_score, 3),
                    'topological_type': material.topological_type,
                    'topological_subtype': material.topological_subtype,
                    'space_group': material.space_group_symbol,
                    'smallest_gap': material.smallest_gap,
                    'shared_invariants': self._find_shared_invariants(query_material, material),
                    'key_differences': self._find_key_differences(query_material, material)
                })
        return sorted(analogues, key=lambda x: x['similarity_score'], reverse=True)

    def search_by_invariant_pattern(self, pattern: Dict[str, int],
                                    match_all: bool = True) -> List[Dict]:
        """Search for materials matching specific invariant patterns.
        Args:
            pattern: Dict of {"invariant_name": value}.
            match_all: If True, all patterns must match. If False, any pattern match suffices.
        """
        matches = []
        for mat_id, material in self.db.materials.items():
            if self._matches_pattern(material, pattern, match_all):
                matches.append({
                    'material_id': mat_id,
                    'formula': material.formula,
                    'topological_type': material.topological_type,
                    'space_group': material.space_group_symbol,
                    'invariants': {**material.z2_invariants, **material.higher_order_invariants}
                })
        return matches

    def _matches_pattern(self, material: 'TopologicalMaterial', pattern: Dict[str, int], match_all: bool) -> bool:
        """Check if material matches invariant pattern."""
        all_material_invariants = {**material.z2_invariants, **material.higher_order_invariants}
        match_count = 0
        for inv_name, required_value in pattern.items():
            material_value = all_material_invariants.get(inv_name)
            if material_value is not None and material_value == required_value:
                if not match_all: return True # Any match is enough
                match_count +=1
            elif match_all: # If any pattern fails for match_all mode
                return False
        return match_count == len(pattern) if match_all else False


    def _calculate_structural_similarity(self, mat1: 'TopologicalMaterial', mat2: 'TopologicalMaterial') -> float:
        """Calculate a heuristic structural similarity score."""
        score = 0.0
        total_weight = 0.0

        # Space group exact match
        w_sg = self.invariant_weights.get('space_group', 0.5)
        total_weight += w_sg
        if mat1.space_group is not None and mat1.space_group == mat2.space_group:
            score += w_sg
        elif mat1.space_group is not None and mat2.space_group is not None: # Graded similarity for SG family
             if (mat1.space_group // 10) == (mat2.space_group // 10) : # Same family (approx)
                 score += w_sg * 0.5

        # Lattice parameters
        params1, params2 = mat1.lattice_params, mat2.lattice_params
        if params1 and params2:
            for p_name in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
                w_p = self.invariant_weights.get(f'lattice_{p_name}', 0.0)
                if w_p == 0.0 : continue
                
                val1, val2 = params1.get(p_name), params2.get(p_name)
                if val1 is not None and val2 is not None:
                    total_weight += w_p
                    # Normalized difference
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 1e-6: # Avoid division by zero for angles if they are zero
                        param_similarity = 1.0 - (abs(val1 - val2) / max_val)
                        score += w_p * max(0, param_similarity) # Ensure similarity is not negative
        
        return score / total_weight if total_weight > 0 else 0.0

    def find_topological_transitions(self, material_id: str, structural_sim_threshold: float = 0.6) -> Dict:
        """Find possible topological phase transitions by looking at symmetry breaking
           data within the material and structurally similar materials with different topology.
        """
        if material_id not in self.db.materials:
            print(f"Error: Material ID '{material_id}' not found.")
            return {}

        material = self.db.materials[material_id]
        transitions = {
            'internal_symmetry_breaking': [], # Transitions listed within the material's data
            'parameter_tuning_candidates': [], # Structurally similar materials with different topology
        }

        # 1. Internal Symmetry Breaking (from material.symmetry_transitions)
        if material.symmetry_transitions:
            for trans_data in material.symmetry_transitions:
                transitions['internal_symmetry_breaking'].append({
                    'target_sg_number': trans_data.target_symmetry_group_number,
                    'target_sg_symbol': trans_data.target_symmetry_group_symbol,
                    'new_topological_indices': trans_data.topological_indices,
                    'notes': trans_data.band_representations_info or "N/A"
                })

        # 2. Parameter Tuning Candidates (structurally similar materials)
        for other_id, other_material in self.db.materials.items():
            if other_id == material_id:
                continue

            struct_similarity = self._calculate_structural_similarity(material, other_material)

            if struct_similarity >= structural_sim_threshold:
                # Check if topology is meaningfully different
                topo_is_different = False
                if material.topological_type != other_material.topological_type:
                    topo_is_different = True
                else: # Same type, check key invariants
                    shared_inv = self._find_shared_invariants(material, other_material)
                    all_inv_count_mat1 = len(material.z2_invariants) + len(material.higher_order_invariants)
                    if len(shared_inv) < all_inv_count_mat1 * 0.5 : # Heuristic: significantly different invariants
                         topo_is_different = True

                if topo_is_different:
                    transitions['parameter_tuning_candidates'].append({
                        'target_material_id': other_id,
                        'target_formula': other_material.formula,
                        'current_topology': f"{material.topological_type} ({material.topological_subtype})",
                        'target_topology': f"{other_material.topological_type} ({other_material.topological_subtype})",
                        'structural_similarity': round(struct_similarity, 3),
                        'key_differences': self._find_key_differences(material, other_material)
                    })
        
        # Sort parameter tuning candidates by structural similarity
        transitions['parameter_tuning_candidates'] = sorted(
            transitions['parameter_tuning_candidates'],
            key=lambda x: x['structural_similarity'],
            reverse=True
        )
        return transitions

    def search_by_band_features(self, num_bands_in_group: Optional[int] = None,
                                subclass: Optional[str] = None,
                                irrep_at_kpoint: Optional[Tuple[str, str]] = None, # (kpoint_name, irrep_label)
                                min_degeneracy: Optional[int] = None) -> List[Dict]:
        """Search materials based on features from their BandGroupRepresentations."""
        matches = []
        for mat_id, material in self.db.materials.items():
            if not material.band_group_representations:
                continue
            
            match_found_for_material = False
            for bg_rep in material.band_group_representations:
                current_bg_match = True # Assume this band group matches criteria initially
                if num_bands_in_group is not None and bg_rep.num_bands != num_bands_in_group:
                    current_bg_match = False
                if subclass is not None and bg_rep.subclass != subclass:
                    current_bg_match = False
                
                if irrep_at_kpoint is not None and current_bg_match:
                    kpt_name, ir_label = irrep_at_kpoint
                    found_irrep_in_bg = False
                    for rep_detail in bg_rep.representations_at_k_points:
                        if rep_detail.k_point == kpt_name and rep_detail.irrep_label == ir_label:
                            if min_degeneracy is not None:
                                if rep_detail.degeneracy is not None and rep_detail.degeneracy >= min_degeneracy:
                                    found_irrep_in_bg = True
                                    break
                            else: # No min_degeneracy check needed
                                found_irrep_in_bg = True
                                break
                    if not found_irrep_in_bg:
                        current_bg_match = False # Required irrep not found in this band group
                
                if current_bg_match: # If all criteria for this band group passed
                    match_found_for_material = True
                    break # No need to check other band groups for this material
            
            if match_found_for_material:
                matches.append({
                    'material_id': mat_id,
                    'formula': material.formula,
                    'topological_type': material.topological_type,
                    # Optionally add matched band group details
                })
        return matches

    def get_material_summary_vector(self, material: 'TopologicalMaterial', max_len: int = 128) -> np.ndarray:
        """Creates a fixed-size numerical vector for a material (for UMAP, etc.).
           This is a simplified vectorizer; a more sophisticated one might be external.
        """
        feature_vector = []
        
        # Invariants (normalize or use directly if range is small)
        for inv_name in ['Z2', 'Z2w_1', 'Z2w_2', 'Z2w_3', 'Z4', 'Z4pi', 'Z8']:
            val = material.z2_invariants.get(inv_name, material.higher_order_invariants.get(inv_name, 0))
            feature_vector.append(val)

        # Space group, EBR class, fragile bands
        feature_vector.append(material.space_group if material.space_group is not None else -1)
        feature_vector.append(material.ebr_class if material.ebr_class is not None else -1)
        feature_vector.append(material.fragile_bands if material.fragile_bands is not None else -1)
        
        # Smallest gap and SOC
        feature_vector.append(material.smallest_gap if material.smallest_gap is not None else -1) # -1 for missing gap
        feature_vector.append(1.0 if material.soc_enabled else 0.0)
        
        # Lattice parameters (a, b, c, vol) - simplified
        lp = material.lattice_params
        feature_vector.append(lp.get('a', 0))
        feature_vector.append(lp.get('b', 0))
        feature_vector.append(lp.get('c', 0))
        feature_vector.append(material.volume if material.volume is not None else 0)

        # Count of band groups and symmetry transitions
        feature_vector.append(len(material.band_group_representations))
        feature_vector.append(len(material.symmetry_transitions))

        # Convert to numpy array
        vec = np.array(feature_vector, dtype=float)
        
        # Pad or truncate to max_len (simple approach)
        if len(vec) < max_len:
            padded_vec = np.zeros(max_len)
            padded_vec[:len(vec)] = vec
            return padded_vec
        return vec[:max_len]


    def visualize_materials_landscape(self, material_ids: Optional[List[str]] = None,
                                      n_neighbors_umap: int = 15, min_dist_umap: float = 0.1,
                                      eps_dbscan: float = 0.5, min_samples_dbscan: int = 5,
                                      color_by: str = 'topological_type', fig_size: Tuple[int, int]=(12,10)):
        """
        Visualizes a landscape of specified materials using UMAP and optionally DBSCAN for clustering.
        Args:
            material_ids: List of material IDs to include. If None, uses all materials.
            n_neighbors_umap, min_dist_umap: UMAP parameters.
            eps_dbscan, min_samples_dbscan: DBSCAN parameters.
            color_by: Property to color points by (e.g., 'topological_type', 'space_group', 'ebr_class').
            fig_size: Figure size.
        """
        if material_ids is None:
            target_materials = list(self.db.materials.values())
            target_ids = list(self.db.materials.keys())
        else:
            target_materials = [self.db.materials[mid] for mid in material_ids if mid in self.db.materials]
            target_ids = [mid for mid in material_ids if mid in self.db.materials]

        if not target_materials:
            print("No materials found for visualization.")
            return

        # Vectorize materials
        material_vectors = np.array([self.get_material_summary_vector(mat) for mat in target_materials])

        # UMAP dimensionality reduction
        print(f"Running UMAP on {len(material_vectors)} materials...")
        reducer = UMAP(n_neighbors=n_neighbors_umap, min_dist=min_dist_umap, random_state=42, n_components=2)
        embedding = reducer.fit_transform(material_vectors)

        # Prepare data for plotting
        plot_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        plot_df['material_id'] = target_ids
        plot_df['formula'] = [mat.formula for mat in target_materials]
        
        # Get color data
        color_labels = []
        if color_by in ['topological_type', 'topological_subtype', 'space_group_symbol', 'cryst_sys']: # Categorical
            color_labels = [getattr(mat, color_by, 'N/A') for mat in target_materials]
            # Convert N/A to string for consistent type
            color_labels = ['N/A' if lab is None else str(lab) for lab in color_labels]
        elif color_by in ['space_group', 'ebr_class', 'smallest_gap', 'valence_electrons']: # Potentially numeric
            raw_labels = [getattr(mat, color_by, np.nan) for mat in target_materials]
            # Handle None before converting to float for isnan check
            color_labels = [np.nan if lab is None else float(lab) for lab in raw_labels]
        else: # Default to formula if color_by is not recognized
            print(f"Warning: color_by field '{color_by}' not directly supported, using formula for hover.")
            color_labels = plot_df['formula'] # Fallback or for hover only
        
        plot_df[color_by] = color_labels

        # Optional: DBSCAN clustering
        print("Running DBSCAN for clustering...")
        clustering = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan).fit(embedding)
        plot_df['cluster'] = [str(c) for c in clustering.labels_] # Convert cluster labels to string for discrete legend

        # Plotting
        plt.figure(figsize=fig_size)
        unique_colors = plot_df[color_by].unique()
        
        # Heuristic to decide if legend is needed
        show_legend = len(unique_colors) < 20 or (pd.api.types.is_numeric_dtype(plot_df[color_by]) and len(unique_colors) > 1)


        sns.scatterplot(
            x='UMAP1', y='UMAP2',
            hue=color_by if show_legend else None, # Only use hue if legend is manageable
            size='cluster' if 'cluster' in plot_df.columns else None, # Example: size by cluster
            sizes=(50, 200) if 'cluster' in plot_df.columns else None,
            palette=sns.color_palette("viridis", n_colors=len(unique_colors) if show_legend and not pd.api.types.is_numeric_dtype(plot_df[color_by]) else None), # Only set n_colors for categorical
            data=plot_df,
            legend="full" if show_legend else False,
            alpha=0.7
        )
        plt.title(f'UMAP Projection of Topological Materials (Colored by {color_by})')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        if show_legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.85 if show_legend else 1, 1]) # Adjust layout for legend
        plt.show()