import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

def create_graph_representation(sym_info: dict, jid_data: dict) -> Data:
    """
    Converts the processed symmetry data into a PyG graph object.
    """
    asym_unit = sym_info["asymmetric_unit"]
    
    # 1. Node Features (Element Vectors)
    # Start simple, you can add more physics later (electronegativity, etc.)
    atomic_numbers = torch.tensor([site.specie.Z for site in asym_unit], dtype=torch.long)
    
    # 2. Positional Information
    positions = torch.tensor(asym_unit.cart_coords, dtype=torch.float)
    
    # 3. Global / Conditioning Features
    # For now, let's target formation energy
    formation_energy = torch.tensor([jid_data["form_energy_per_atom"]], dtype=torch.float)

    graph_data = Data(
        x=atomic_numbers,      # Node features (will be embedded)
        pos=positions,         # Node positions
        y=formation_energy,    # Target variable to predict
        
        # Store the physics for later stages!
        lattice=torch.tensor(sym_info["lattice"], dtype=torch.float).unsqueeze(0),
        rotations=torch.tensor(sym_info["rotations"], dtype=torch.float),
        translations=torch.tensor(sym_info["translations"], dtype=torch.float)
    )

    # 4. Edge Creation
    # Let PyG handle creating edges based on a cutoff radius
    radius_transformer = RadiusGraph(r=6.0) 
    return radius_transformer(graph_data)

# --- Full Pipeline Test ---
jid = "JVASP-1005"
struct = fetch_structure(jid)
jid_data = get_jid_data(jid=jid, dataset="dft_3d")
sym_info = get_symmetry_info(struct)
graph = create_graph_representation(sym_info, jid_data)
print(graph)