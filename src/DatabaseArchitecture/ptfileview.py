import torch 
file_path = "graph_vector_dataset/JVASP-5.pt"
loaded = torch.load(file_path, weights_only= False)['kspace_graph']

print(loaded)