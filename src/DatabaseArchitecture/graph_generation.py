
import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from jarvis.db.figshare import get_jid_data
from torch_geometric.data import Data

class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        # GNN for the atomic crystal structure
        self.crystal_encoder = CrystalGNN(...) 
        # A separate, likely simpler GNN for the k-space graph
        self.kspace_encoder = KSpaceGNN(...)
        # A final network to combine the results
        self.combiner_head = nn.Sequential(...)

    def forward(self, crystal_batch, kspace_batch):
        # 1. Process each graph through its own encoder
        crystal_embedding = self.crystal_encoder(crystal_batch)
        kspace_embedding = self.kspace_encoder(kspace_batch)

        # 2. Combine the two resulting vectors
        combined_embedding = torch.cat([crystal_embedding, kspace_embedding], dim=-1)

        # 3. Make the final prediction
        return self.combiner_head(combined_embedding)



# # --- Helper function for Element Feature Vectorization ---
# def get_elemental_features(atomic_number):
#     # This should be a pre-computed dictionary or a more sophisticated function
#     # For now, just a placeholder
#     # In reality, this would be a vector of electronegativity, radius, valence electrons, etc.
#     return torch.tensor([atomic_number], dtype=torch.float)

# # --- Helper function for Persistent Homology (New, inspired by Rasul et al.) ---
# def compute_asph_features(structure: Structure, n_bins=100) -> torch.tensor:
#     """Computes a feature vector using persistent homology."""
#     points = structure.cart_coords.reshape(1, -1, 3) # Reshape for giotto-tda

#     # 1. Compute persistence diagrams (Betti 0, 1, 2)
#     vr = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
#     diagrams = vr.fit_transform(points)

#     # 2. Convert diagrams to a fixed-size feature vector (Betti curves)
#     bc = BettiCurve(n_bins=n_bins)
#     betti_curves = bc.fit_transform(diagrams) # Shape: (1, 3, n_bins)
    
#     # 3. Flatten into a single feature vector
#     return torch.tensor(betti_curves.flatten(), dtype=torch.float)


# # --- The Main Data Generation Function ---
# def create_enhanced_graph_from_jid(jid: str, local_topo_data: dict) -> Data:
#     """
#     Creates a single, enhanced graph object for a given JARVIS ID.

#     Args:
#         jid (str): The JARVIS ID (e.g., "JVASP-12345").
#         local_topo_data (dict): Your pre-parsed dict containing band reps
#                                 and topological indices for this material.
#     Returns:
#         A torch_geometric.data.Data object.
#     """
#     # 1. FETCH DATA
#     jid_data = get_jid_data(jid=jid, dataset="dft_3d")
#     structure = Structure.from_dict(jid_data["atoms"])

#     # 2. SYMMETRY ANALYSIS (Core of our Generative Plan)
#     analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
#     asym_unit = analyzer.get_symmetrized_structure().equivalent_structures[0]
#     symm_ops = analyzer.get_symmetry_operations(cartesian=True)
    
#     # 3. GRAPH CONNECTIVITY (Voronoi Method from Rasul et al.)
#     # This is a more physically robust way to define edges
#     vnn = VoronoiNN(cutoff=10) # Cutoff for finding neighbors
#     bonds = vnn.get_all_nn_info(asym_unit)
    
#     edge_index_list = []
#     for i, neighbors in enumerate(bonds):
#         for neighbor_info in neighbors:
#             j = neighbor_info['site_index']
#             # Avoid self-loops if structure has only 1 atom
#             if i != j:
#                 edge_index_list.append([i, j])

#     edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()


#     # 4. FEATURE ENGINEERING
#     # 4a. Node Features (Atoms in the asymmetric unit)
#     node_features = torch.stack([get_elemental_features(site.specie.Z) for site in asym_unit])
#     node_positions = torch.tensor(asym_unit.cart_coords, dtype=torch.float)

#     # 4b. Persistent Homology Features (Global structural fingerprint)
#     asph_vector = compute_asph_features(structure)

#     # 4c. Electronic Topology & Symmetry Features (From your local files)
#     # This is what makes our dataset unique!
#     band_rep_vector = torch.tensor(local_topo_data.get(jid, {}).get("band_rep_vector", []), dtype=torch.float)
#     topo_indices_vector = torch.tensor(local_topo_data.get(jid, {}).get("topo_indices", []), dtype=torch.float)


#     # 5. ASSEMBLE THE "SUPER-GRAPH"
#     graph = Data(
#         # --- Node/Atom Information ---
#         x=node_features,                  # Features for each atom in the asym_unit
#         pos=node_positions,               # 3D coordinates of atoms in the asym_unit

#         # --- Graph Structure ---
#         edge_index=edge_index,            # Connectivity from Voronoi analysis

#         # --- Target Properties (for prediction tasks) ---
#         y_formation_energy=jid_data.get("form_energy_per_atom"),
#         y_band_gap=jid_data.get("optb88vdw_bandgap"),

#         # --- GLOBAL FEATURES (Multi-scale physics) ---
#         # A. For Equivariance & Generation (Our Plan)
#         lattice=torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0),
#         rotations=torch.stack([torch.tensor(op.rotation_matrix, dtype=torch.float) for op in symm_ops]),
#         translations=torch.stack([torch.tensor(op.translation_vector, dtype=torch.float) for op in symm_ops]),

#         # B. For Structural Topology (Rasul et al. idea)
#         asph_features=asph_vector.unsqueeze(0),
        
#         # C. For Electronic Topology (Your specialized data)
#         band_rep_features=band_rep_vector.unsqueeze(0),
#         topological_indices=topo_indices_vector.unsqueeze(0)
#     )

#     return graph

# #
# # --- Example Usage ---
# #
# # if __name__ == '__main__':
# #     # 1. Get a list of JIDs you want to process
# #     all_jids = ["JVASP-1005", "JVASP-130", ...] # Your list of 33,800 materials
# #
# #     # 2. Load your pre-parsed local topological data
# #     # my_local_data = load_my_band_rep_and_topo_indices_file()
# #
# #     # 3. Loop, process, and save
# #     for jid in all_jids:
# #         try:
# #             enhanced_graph = create_enhanced_graph_from_jid(jid, my_local_data)
# #             # Save the processed graph object
# #             torch.save(enhanced_graph, f"processed_dataset/{jid}.pt")
# #             print(f"Successfully processed and saved {jid}")
# #         except Exception as e:
# #             print(f"Failed to process {jid}: {e}")