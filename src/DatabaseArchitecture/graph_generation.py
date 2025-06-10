from dataset_builder_backend import TopologicalMaterialAnalyzer

csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
db_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/pebr_tr_nonmagnetic_rev4.db"

topo = TopologicalMaterialAnalyzer(csv_path, db_path)
print(topo._get_kspace_data_from_db(12))