import spglib
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from pymatgen.core import Lattice, Structure

jvasp_input = "JVASP-125448"

entry  = get_jid_data(jid=jvasp_input, dataset="dft_3d")
atoms  = Atoms.from_dict(entry["atoms"])
latt   = Lattice(atoms.lattice_mat)
struct = Structure(latt, atoms.elements, atoms.coords)

magmoms = entry.get("magmom_outcar") or entry.get("magmom_oszicar")
if isinstance(magmoms, (int, float)):
    magmoms = [magmoms] * len(struct)

lattice   = struct.lattice.matrix
positions = struct.frac_coords
numbers   = [site.specie.Z for site in struct]
magmoms_vec = [[0, 0, m] for m in magmoms]

cell = (lattice, positions, numbers, magmoms_vec)

mag_dataset = spglib.get_magnetic_symmetry_dataset(cell, symprec=1e-5)

if mag_dataset is None:
    print("No magnetic dataset found—check your spins!")
else:
    # uni_number is the BNS/UNI serial 1…1651
    print("BNS (UNI) number:", mag_dataset.uni_number)
    # hall_number is the Hall serial for the underlying non-magnetic subgroup
    print("Hall number:",      mag_dataset.hall_number)
    # msg_type tells you Type I/II/III/IV
    print("MSG family type:",  mag_dataset.msg_type)

from spglib import get_magnetic_spacegroup_type

msg_info = get_magnetic_spacegroup_type(mag_dataset.uni_number)
# msg_info.bns_number is the “225.2.2”-style label
# msg_info.og_number is the Opechowsky-Guccione label
print("BNS label:", msg_info.bns_number)
print("OG label: ", msg_info.og_number)

print("SG symbol:", entry["spg_symbol"]) 
print("SG number:", entry["spg_number"])  
print("Formula: ", entry['formula'])
print(jvasp_input)
