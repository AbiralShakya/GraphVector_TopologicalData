import os
import pandas as pd
import requests
import warnings
import numpy as np
from pymatgen.core import Structure 
from typing import Dict, Optional, List
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import data
import networkx as nx
import matplotlib.pyplot as plt
import json

class UnifiedMaterialGraphDB:
    """
    Unified database structure linking:
    1. Vector properties from JARVIS
    2. Real-space atomistic graph
    3. K-space band connectivity (when available)
    4. Topological/symmetry data
    """
    
    def __init__(self):
        self.jarvis_data = None
        self.load_jarvis_database()
    
    def load_jarvis_database(self):
        """Load JARVIS database for property lookup"""
        try:
            # Load DFT-3D database (most comprehensive)
            self.jarvis_data = data('dft_3d')
            print(f"Loaded JARVIS database with {len(self.jarvis_data)} materials")
        except Exception as e:
            print(f"Warning: Could not load JARVIS database: {e}")
            self.jarvis_data = None

    def parse_poscar_with_jarvis_graph(
        self,
        poscar_string_content: str,
        source_identifier: str = "N/A",
        cutoff: float = 8.0,
        use_canonize: bool = True
    ) -> Optional[Dict]:
        """Parse POSCAR and create JARVIS-style graphs with enhanced features"""
        try:
            if not isinstance(poscar_string_content, str) or not poscar_string_content.strip():
                return None

            # Create PyMatGen structure first
            structure = Structure.from_str(poscar_string_content, fmt="poscar")
            
            # Convert to JARVIS atoms
            jarvis_atoms = Atoms.from_dict(structure.as_dict())
            
            # Create JARVIS graph with multiple representations
            graph_data = self._create_jarvis_graphs(jarvis_atoms, cutoff, use_canonize)
            
            # Get material properties from JARVIS if available
            jarvis_properties = self._get_jarvis_properties(structure)
            
            # Get symmetry and topological data
            symmetry_data = self._get_symmetry_data(jarvis_atoms)
            
            # Extract comment
            comment_line = ""
            if poscar_string_content:
                lines = poscar_string_content.splitlines()
                if lines:
                    comment_line = lines[0].strip()

            return {
                "source_identifier": source_identifier,
                "comment": comment_line,
                "formula": structure.composition.reduced_formula,
                "natoms": len(structure),
                "lattice_vectors": structure.lattice.matrix.tolist(),
                "sites": [{"species": site.species_string,
                          "xyz_frac": site.frac_coords.tolist(),
                          "xyz_cart": site.coords.tolist()}
                         for site in structure.sites],
                "pymatgen_structure": structure,
                "jarvis_atoms": jarvis_atoms,
                
                # Graph representations
                "atomistic_graph": graph_data["atomistic_graph"],
                "line_graph": graph_data["line_graph"],
                "undirected_graph": graph_data["undirected_graph"],
                
                # JARVIS vector properties
                "jarvis_properties": jarvis_properties,
                
                # Symmetry and topological data
                "symmetry_data": symmetry_data,
                
                # Placeholder for k-space data (would need band structure calculation)
                "k_space_graph": None,
                
                # Combined feature vector for ML
                "feature_vector": self._create_feature_vector(
                    jarvis_properties, symmetry_data, graph_data
                )
            }
            
        except Exception as e:
            print(f"Error parsing POSCAR content from '{source_identifier}': {e}")
            return None

    def _create_jarvis_graphs(self, atoms: Atoms, cutoff: float, use_canonize: bool) -> Dict:
        """Create multiple JARVIS graph representations"""
        try:
            # Standard atomistic graph
            g = Graph.atom_dgl_multigraph(
                atoms, 
                cutoff=cutoff,
                atom_features='atomic_number',
                use_canonize=use_canonize
            )
            
            # Line graph (important for some ML models like ALIGNN)
            lg = Graph.get_line_graph_from_graph(g)
            
            # Undirected version for some analyses
            ug = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=cutoff,
                atom_features='atomic_number',
                use_canonize=use_canonize,
                id_tag='jid'
            )
            
            # Convert to networkx for compatibility with your existing code
            atomistic_nx = self._dgl_to_networkx(g, atoms)
            
            return {
                "atomistic_graph": atomistic_nx,
                "line_graph": lg,
                "undirected_graph": ug,
                "dgl_graph": g  # Keep original DGL graph for deep learning
            }
            
        except Exception as e:
            print(f"Error creating JARVIS graphs: {e}")
            return {"atomistic_graph": None, "line_graph": None, "undirected_graph": None}

    def _dgl_to_networkx(self, dgl_graph, atoms: Atoms) -> Dict:
        """Convert DGL graph to NetworkX format for compatibility"""
        try:
            import dgl
            graph_dict = {"nodes": [], "edges": []}
            
            # Add nodes with features
            for i, site in enumerate(atoms.cart_coords):
                graph_dict["nodes"].append({
                    "id": i,
                    "species": atoms.elements[i],
                    "coords": site.tolist(),
                    "atomic_number": atoms.atomic_numbers[i]
                })
            
            # Add edges
            src, dst = dgl_graph.edges()
            for s, d in zip(src.numpy(), dst.numpy()):
                # Calculate distance
                coords1 = atoms.cart_coords[s]
                coords2 = atoms.cart_coords[d]
                distance = np.linalg.norm(coords1 - coords2)
                graph_dict["edges"].append((int(s), int(d), {"distance": distance}))
            
            return graph_dict
            
        except Exception as e:
            print(f"Error converting DGL to NetworkX: {e}")
            return {"nodes": [], "edges": []}

    def _get_jarvis_properties(self, structure: Structure) -> Dict:
        """Get material properties from JARVIS database"""
        if self.jarvis_data is None:
            return {}
        
        try:
            formula = structure.composition.reduced_formula
            
            # Find matching entry in JARVIS database
            matching_entries = []
            for entry in self.jarvis_data:
                if entry.get('formula', '') == formula:
                    matching_entries.append(entry)
            
            if not matching_entries:
                print(f"No JARVIS entry found for formula: {formula}")
                return {"formula": formula, "jarvis_match": False}
            
            # Take the first match (you might want to implement better matching)
            entry = matching_entries[0]
            
            # Extract key properties for ML
            properties = {
                "formula": formula,
                "jarvis_match": True,
                "jid": entry.get('jid', ''),
                
                # Electronic properties
                "bandgap_opt": entry.get('optb88vdw_bandgap', None),
                "bandgap_tbmbj": entry.get('mbj_bandgap', None),
                "formation_energy": entry.get('formation_energy_peratom', None),
                
                # Mechanical properties
                "bulk_modulus": entry.get('kv', None),  # Voigt bulk modulus
                "shear_modulus": entry.get('gv', None),  # Voigt shear modulus
                
                # Electronic structure
                "n_electrons": entry.get('n_electrons', None),
                "n_dos": entry.get('n_dos', None),
                
                # Magnetic properties
                "magmom": entry.get('magmom_oszicar', None),
                
                # Dielectric properties
                "epsilon_x": entry.get('epsilon_x', None),
                "epsilon_y": entry.get('epsilon_y', None),
                "epsilon_z": entry.get('epsilon_z', None),
                
                # Additional properties
                "density": entry.get('density', None),
                "spacegroup": entry.get('spg_symbol', None),
                "spacegroup_number": entry.get('spg_number', None)
            }
            
            # Clean None values
            properties = {k: v for k, v in properties.items() if v is not None}
            
            return properties
            
        except Exception as e:
            print(f"Error getting JARVIS properties: {e}")
            return {"formula": structure.composition.reduced_formula, "jarvis_match": False}

    def _get_symmetry_data(self, atoms: Atoms) -> Dict:
        """Extract symmetry and topological data"""
        try:
            spg = Spacegroup3D(atoms)
            
            symmetry_data = {
                "space_group": spg.space_group_symbol,
                "space_group_number": spg.space_group_number,
                "crystal_system": spg.crystal_system,
                "point_group": spg.point_group_symbol,
                "lattice_system": spg.lattice_system,
                
                # Topological invariants (basic)
                "wyckoff_positions": len(spg.wyckoff_symbols),
                "multiplicities": spg.wyckoff_multiplicities,
                
                # For TQC integration - placeholders
                "topological_invariants": {},  # Would compute Z2, Chern numbers, etc.
                "symmetry_indicators": {},     # Would compute symmetry-based indicators
            }
            
            return symmetry_data
            
        except Exception as e:
            print(f"Error getting symmetry data: {e}")
            return {}

    def _create_feature_vector(self, jarvis_props: Dict, symmetry_data: Dict, graph_data: Dict) -> np.ndarray:
        """Create unified feature vector for ML predictions"""
        features = []
        
        # JARVIS numerical properties
        jarvis_features = [
            jarvis_props.get('formation_energy', 0),
            jarvis_props.get('bandgap_opt', 0),
            jarvis_props.get('bulk_modulus', 0),
            jarvis_props.get('shear_modulus', 0),
            jarvis_props.get('density', 0),
            jarvis_props.get('spacegroup_number', 0),
            jarvis_props.get('n_electrons', 0),
        ]
        features.extend([f if f is not None else 0 for f in jarvis_features])
        
        # Symmetry features
        symmetry_features = [
            symmetry_data.get('space_group_number', 0),
            symmetry_data.get('wyckoff_positions', 0),
        ]
        features.extend(symmetry_features)
        
        # Graph-based features
        if graph_data.get('atomistic_graph'):
            graph = graph_data['atomistic_graph']
            graph_features = [
                len(graph.get('nodes', [])),  # Number of atoms
                len(graph.get('edges', [])),  # Number of bonds
            ]
            features.extend(graph_features)
        
        return np.array(features, dtype=float)

    def process_csv_with_unified_graphs(
        self,
        csv_filepath: str,
        link_column_name: str = 'POSCAR_link',
        max_materials: int = 10,
        disable_ssl_verify: bool = False
    ) -> List[Dict]:
        """Process multiple materials from CSV and create unified database"""
        
        try:
            df = pd.read_csv(csv_filepath, nrows=max_materials)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []
        
        if link_column_name not in df.columns:
            print(f"Column '{link_column_name}' not found in CSV")
            return []
        
        unified_database = []
        
        for idx, row in df.iterrows():
            poscar_url = row[link_column_name]
            
            if pd.isna(poscar_url) or not isinstance(poscar_url, str):
                continue
            
            print(f"Processing material {idx + 1}/{len(df)}: {poscar_url}")
            
            # Fetch POSCAR content
            try:
                if disable_ssl_verify:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                response = requests.get(poscar_url, timeout=30, verify=not disable_ssl_verify)
                response.raise_for_status()
                poscar_content = response.text
                
            except Exception as e:
                print(f"Error fetching {poscar_url}: {e}")
                continue
            
            # Parse with unified approach
            material_data = self.parse_poscar_with_jarvis_graph(
                poscar_content, 
                source_identifier=poscar_url
            )
            
            if material_data:
                unified_database.append(material_data)
                print(f"Successfully processed: {material_data['formula']}")
            else:
                print(f"Failed to process material from {poscar_url}")
        
        return unified_database

    def save_database(self, database: List[Dict], output_path: str):
        """Save the unified database"""
        # Prepare data for JSON serialization
        serializable_db = []
        for entry in database:
            serializable_entry = {}
            for key, value in entry.items():
                if key in ['pymatgen_structure', 'jarvis_atoms', 'dgl_graph']:
                    # Skip non-serializable objects
                    continue
                elif isinstance(value, np.ndarray):
                    serializable_entry[key] = value.tolist()
                else:
                    serializable_entry[key] = value
            serializable_db.append(serializable_entry)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_db, f, indent=2)
        print(f"Database saved to {output_path}")

    def visualize_material(self, material_data: Dict):
        """Visualize the atomistic graph for a material"""
        graph_dict = material_data["atomistic_graph"]
        if not graph_dict or not graph_dict.get("nodes"):
            print("No graph data to visualize")
            return
        
        G = nx.Graph()
        
        # Add nodes with species information
        node_colors = []
        for node in graph_dict["nodes"]:
            G.add_node(node["id"], **node)
            # Color by element type
            species = node.get("species", "C")
            color_map = {"C": "gray", "Si": "blue", "O": "red", "N": "green", "H": "white"}
            node_colors.append(color_map.get(species, "purple"))
        
        # Add edges
        for u, v, attrs in graph_dict["edges"]:
            G.add_edge(u, v, **attrs)
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Graph visualization
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_color=node_colors, node_size=100, 
                edge_color="gray", with_labels=True, ax=ax1)
        ax1.set_title(f"Atomistic Graph: {material_data['formula']}")
        ax1.axis("off")
        
        # Property visualization
        props = material_data.get('jarvis_properties', {})
        if props and props.get('jarvis_match'):
            prop_names = []
            prop_values = []
            for key, value in props.items():
                if isinstance(value, (int, float)) and key != 'spacegroup_number':
                    prop_names.append(key.replace('_', ' ').title())
                    prop_values.append(value)
            
            if prop_names:
                ax2.barh(prop_names[:10], prop_values[:10])  # Show top 10 properties
                ax2.set_title("JARVIS Properties")
        
        # Feature vector visualization
        feature_vec = material_data.get('feature_vector')
        if feature_vec is not None:
            ax3.plot(feature_vec, 'o-')
            ax3.set_title("Unified Feature Vector")
            ax3.set_xlabel("Feature Index")
            ax3.set_ylabel("Feature Value")
        
        # Symmetry information
        sym_data = material_data.get('symmetry_data', {})
        if sym_data:
            info_text = f"""
            Space Group: {sym_data.get('space_group', 'N/A')}
            Crystal System: {sym_data.get('crystal_system', 'N/A')}
            Point Group: {sym_data.get('point_group', 'N/A')}
            Wyckoff Positions: {sym_data.get('wyckoff_positions', 'N/A')}
            """
            ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
            ax4.set_title("Symmetry Information")
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    # Initialize the unified database system
    db_system = UnifiedMaterialGraphDB()
    
    csv_path = "/Users/abiralshakya/Downloads/materials_database.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    # Process materials (start with a small number for testing)
    print("Processing materials with unified graph approach...")
    unified_database = db_system.process_csv_with_unified_graphs(
        csv_path,
        link_column_name='POSCAR_link',
        max_materials=5,  # Start small for testing
        disable_ssl_verify=True
    )
    
    if unified_database:
        print(f"\nSuccessfully processed {len(unified_database)} materials")
        
        # Save the database
        db_system.save_database(unified_database, "unified_materials_database.json")
        
        # Visualize the first material
        if unified_database:
            print("\nVisualizing first material...")
            db_system.visualize_material(unified_database[0])
            
            # Print summary of the first material
            first_material = unified_database[0]
            print(f"\nFirst Material Summary:")
            print(f"Formula: {first_material['formula']}")
            print(f"JARVIS Match: {first_material['jarvis_properties'].get('jarvis_match', False)}")
            print(f"Feature Vector Length: {len(first_material.get('feature_vector', []))}")
            print(f"Graph Nodes: {len(first_material['atomistic_graph']['nodes'])}")
            print(f"Graph Edges: {len(first_material['atomistic_graph']['edges'])}")
    else:
        print("No materials were successfully processed")


if __name__ == '__main__':
    main()