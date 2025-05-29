#TODO: add graph embedding (deepwalk)

import networkx as nx
import matplotlib.pyplot as plt # For plotting
from topologicalmaterial import TopologicalMaterial, process_h2sc1_example

def create_kpoint_irrep_graph(material: TopologicalMaterial) -> nx.Graph:
    # creates graph --> noddess are high symmetry k points
    # node attributes store list of band group representations at k point
    # edges can represent std BZ paths
   
    G = nx.Graph(material_formula=material.formula)
    
    # define k-points from data structure
    # infer k-points from the parsed band_group_representations
    all_kpoints_in_data = set()
    for bg_rep in material.band_group_representations:
        for rep_detail in bg_rep.representations_at_k_points:
            all_kpoints_in_data.add(rep_detail.k_point)
    
    for k_point_name in sorted(list(all_kpoints_in_data)):
        G.add_node(k_point_name, representations=[])

    # populate node attributes with representation details
    for k_point_name in G.nodes():
        representations_at_kpoint = []
        for bg_rep_idx, bg_rep in enumerate(material.band_group_representations):
            for rep_detail in bg_rep.representations_at_k_points:
                if rep_detail.k_point == k_point_name:
                    representations_at_kpoint.append({
                        "band_group_index": bg_rep_idx,
                        "num_bands_in_group": bg_rep.num_bands,
                        "subclass_of_group": bg_rep.subclass,
                        "irrep_label": rep_detail.irrep_label,
                        "degeneracy": rep_detail.degeneracy
                    })
        G.nodes[k_point_name]['representations'] = representations_at_kpoint
        G.nodes[k_point_name]['num_distinct_irreps'] = len(representations_at_kpoint)

    # Add edges (example: a simple linear path for visualization)
    # This should ideally come from actual BZ path information for the material's space group
    # for demonstration, let's connect them if they appear in a common order:
    # Highly simplified.
    ordered_kpoints = sorted(list(all_kpoints_in_data)) # Example order
    if len(ordered_kpoints) > 1:
        for i in range(len(ordered_kpoints) - 1):
            G.add_edge(ordered_kpoints[i], ordered_kpoints[i+1], path_segment=f"{ordered_kpoints[i]}-{ordered_kpoints[i+1]}")
            
    return G

if __name__ == "__main__":
    material_obj, _ = process_h2sc1_example()

    if material_obj:
        kpoint_graph = create_kpoint_irrep_graph(material_obj)
        print("\n--- K-Point Irrep Graph ---")
        print(f"Nodes: {kpoint_graph.nodes(data=True)}")
        print(f"Edges: {kpoint_graph.edges(data=True)}")

        # Basic plot (optional)
        # For better visualization, use graphviz or adjust layout parameters
        # plt.figure(figsize=(10, 7))
        # pos = nx.spring_layout(kpoint_graph) # or nx.kamada_kawai_layout(kpoint_graph)
        # nx.draw(kpoint_graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
        # edge_labels = nx.get_edge_attributes(kpoint_graph, 'path_segment')
        # nx.draw_networkx_edge_labels(kpoint_graph, pos, edge_labels=edge_labels)
        # plt.title(f"K-Point Irrep Graph for {material_obj.formula}")
        # plt.show()