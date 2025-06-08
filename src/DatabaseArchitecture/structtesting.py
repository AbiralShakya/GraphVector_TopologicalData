# from jarvis.db.figshare import get_jid_data

# entry = get_jid_data("JVASP-14839", dataset="dft_3d")
# print(entry)

# from jarvis.db.figshare import data

# all3d = data("dft_3d")           
# entry = next(x for x in all3d if x["jid"] == "JVASP-14839")
# print(entry)

# from jarvis.db.figshare import get_jid_data
# from jarvis.core.atoms import Atoms
# from pymatgen.core import Lattice, Structure
# from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

# # 1) Fetch the entry dict
# entry = get_jid_data(jid="JVASP-14839", dataset="dft_3d")

# # 2) Build Atoms → pymatgen.Structure
# atoms = Atoms.from_dict(entry["atoms"])
# latt    = Lattice(atoms.lattice_mat)
# species = atoms.elements
# coords  = atoms.coords        # fractional coords
# struct  = Structure(latt, species, coords)

# # 3) Pull out the magmoms (could be float or list)
# magmoms = entry.get("magmom_outcar") or entry.get("magmom_oszicar")
# if magmoms is None:
#     raise ValueError("No magnetic moment data found in entry.")
# # broadcast if a single float/int
# if isinstance(magmoms, (int, float)):
#     magmoms = [magmoms] * len(struct)

# # 4) Attach to each site
# for site, m in zip(struct, magmoms):
#     site.properties["magmom"] = m

# # 5) Analyze exchange‐Hamiltonian symmetry
# cmsa = CollinearMagneticStructureAnalyzer(
#     struct,
#     overwrite_magmom_mode="none",
#     threshold_ordering=1e-8,
#     make_primitive=False
# )
# sg_symbol, sg_number = cmsa.get_exchange_group_info(symprec=0.01, angle_tolerance=5)

# print("Exchange‐Hamiltonian SG:", sg_symbol, sg_number)
# print("Magnetic ordering guess:", cmsa.ordering)
# print("Inequivalent mag sites:", cmsa.number_of_unique_magnetic_sites(
#     symprec=0.001, angle_tolerance=5))

from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

try:
    import spglib
    SPGLIB_AVAILABLE = True
    # Check if magnetic functions are available (spglib v2.0+)
    SPGLIB_MAGNETIC = hasattr(spglib, 'get_magnetic_dataset')
    if not SPGLIB_MAGNETIC:
        print("Note: spglib version < 2.0 detected. Magnetic space group analysis limited.")
        print("For full BNS analysis, upgrade with: pip install spglib>=2.0")
except ImportError:
    SPGLIB_AVAILABLE = False
    SPGLIB_MAGNETIC = False
    print("Warning: spglib not available. Install with: pip install spglib")

def get_bns_magnetic_space_group(jid, dataset="dft_3d"):
    """
    Get BNS magnetic space group number for a JARVIS entry
    """
    # 1) Fetch the entry dict
    entry = get_jid_data(jid=jid, dataset=dataset)
    
    # 2) Build Atoms → pymatgen.Structure
    atoms = Atoms.from_dict(entry["atoms"])
    latt = Lattice(atoms.lattice_mat)
    species = atoms.elements
    coords = atoms.coords
    struct = Structure(latt, species, coords)
    
    # 3) Get magnetic moments
    magmoms = entry.get("magmom_outcar") or entry.get("magmom_oszicar")
    if magmoms is None:
        raise ValueError("No magnetic moment data found in entry.")
    
    if isinstance(magmoms, (int, float)):
        magmoms = [magmoms] * len(struct)
    
    # 4) Attach to each site
    for site, m in zip(struct, magmoms):
        site.properties["magmom"] = m
    
    results = {
        "jid": jid,
        "formula": struct.composition.reduced_formula,
        "magnetic_moments": magmoms,
        "bns_number": None,
        "bns_symbol": None,
        "og_number": None,  # Opechowski-Guccione number
        "method": None
    }
    
    # Method 1: Use spglib for magnetic space group analysis (v2.0+ only)
    if SPGLIB_AVAILABLE and SPGLIB_MAGNETIC:
        try:
            # Convert structure to spglib format
            lattice = struct.lattice.matrix
            positions = struct.frac_coords
            numbers = [struct[i].specie.Z for i in range(len(struct))]
            
            # Create magnetic structure for spglib
            # spglib expects magnetic moments as vectors (for collinear: [0, 0, mz])
            magmoms_vec = []
            for mag in magmoms:
                if abs(mag) > 1e-6:
                    magmoms_vec.append([0, 0, mag])  # collinear along z
                else:
                    magmoms_vec.append([0, 0, 0])
            
            # Get magnetic space group
            cell = (lattice, positions, numbers, magmoms_vec)
            mag_dataset = spglib.get_magnetic_dataset(cell, symprec=1e-5)
            
            if mag_dataset:
                results["bns_number"] = mag_dataset.get("uni_number")
                results["bns_symbol"] = mag_dataset.get("msg_type", {}).get("bns_number")
                results["og_number"] = mag_dataset.get("msg_type", {}).get("og_number") 
                results["method"] = "spglib_v2"
                
        except Exception as e:
            print(f"spglib magnetic analysis failed: {e}")
    elif SPGLIB_AVAILABLE:
        # Use regular spglib for conventional space group analysis
        try:
            lattice = struct.lattice.matrix
            positions = struct.frac_coords
            numbers = [struct[i].specie.Z for i in range(len(struct))]
            cell = (lattice, positions, numbers)
            
            # Get conventional space group info
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            if dataset:
                print(f"Conventional SG from spglib: {dataset['international']} (#{dataset['number']})")
                
        except Exception as e:
            print(f"spglib conventional analysis failed: {e}")
    
    # Method 2: Use pymatgen magnetic analyzer + lookup table
    if results["bns_number"] is None:
        try:
            cmsa = CollinearMagneticStructureAnalyzer(
                struct,
                overwrite_magmom_mode="none",
                threshold_ordering=1e-8,
                make_primitive=False
            )
            
            # Get conventional space group
            sga = SpacegroupAnalyzer(struct)
            conventional_sg = sga.get_space_group_number()
            
            # Get magnetic ordering type
            ordering = cmsa.ordering
            
            # Use lookup table for common cases
            bns_estimate = estimate_bns_from_conventional_sg(
                conventional_sg, ordering, magmoms
            )
            
            if bns_estimate:
                results.update(bns_estimate)
                results["method"] = "lookup_table"
                
        except Exception as e:
            print(f"Pymatgen analysis failed: {e}")
    
    return results, struct

def estimate_bns_from_conventional_sg(conventional_sg, ordering, magmoms):
    """
    Estimate BNS number from conventional space group and magnetic ordering
    Enhanced lookup table with more comprehensive coverage
    """
    mag_moments = np.array(magmoms)
    total_moment = np.sum(mag_moments)
    abs_moments = np.abs(mag_moments)
    non_zero_moments = abs_moments[abs_moments > 1e-6]
    
    # Determine magnetic type
    if len(non_zero_moments) == 0:
        mag_type = "paramagnetic"
        return None  # No magnetic space group for paramagnetic materials
    elif np.all(mag_moments >= -1e-6):
        mag_type = "ferromagnetic"
    elif abs(total_moment) < 1e-6:
        mag_type = "antiferromagnetic"
    else:
        mag_type = "ferrimagnetic"
    
    # Enhanced BNS lookup for common space groups
    # Format: BNS.family.variant
    bns_lookup = {
        # Cubic systems - Face-centered (Fm-3m, #225)
        (225, "ferromagnetic"): {
            "bns_number": "225.1.1", 
            "bns_symbol": "Fm-3m", 
            "description": "Ferromagnetic face-centered cubic (like Fe, Ni)"
        },
        (225, "antiferromagnetic"): {
            "bns_number": "225.2.2", 
            "bns_symbol": "Fm'-3m'", 
            "description": "Antiferromagnetic Type-II (checkerboard AFM)"
        },
        
        # Cubic systems - Primitive (Pm-3m, #221)
        (221, "ferromagnetic"): {
            "bns_number": "221.1.1", 
            "bns_symbol": "Pm-3m", 
            "description": "Ferromagnetic primitive cubic"
        },
        (221, "antiferromagnetic"): {
            "bns_number": "221.2.2", 
            "bns_symbol": "Pm'-3m'", 
            "description": "Antiferromagnetic primitive cubic"
        },
        
        # Cubic systems - Body-centered (Im-3m, #229)
        (229, "ferromagnetic"): {
            "bns_number": "229.1.1", 
            "bns_symbol": "Im-3m", 
            "description": "Ferromagnetic body-centered cubic (like α-Fe)"
        },
        (229, "antiferromagnetic"): {
            "bns_number": "229.2.2", 
            "bns_symbol": "Im'-3m'", 
            "description": "Antiferromagnetic body-centered cubic (like Cr)"
        },
        
        # Hexagonal systems (P63/mmc, #194)
        (194, "ferromagnetic"): {
            "bns_number": "194.1.1", 
            "bns_symbol": "P6₃/mmc", 
            "description": "Ferromagnetic hexagonal close-packed (like Co)"
        },
        (194, "antiferromagnetic"): {
            "bns_number": "194.2.2", 
            "bns_symbol": "P6₃/mm'c'", 
            "description": "Antiferromagnetic hexagonal"
        },
        
        # Tetragonal systems (P4/mmm, #123)
        (123, "ferromagnetic"): {
            "bns_number": "123.1.1", 
            "bns_symbol": "P4/mmm", 
            "description": "Ferromagnetic tetragonal"
        },
        (123, "antiferromagnetic"): {
            "bns_number": "123.2.2", 
            "bns_symbol": "P4/mm'm'", 
            "description": "Antiferromagnetic tetragonal"
        },
        
        # Tetragonal systems (P4/nmm, #129)
        (129, "ferromagnetic"): {
            "bns_number": "129.1.1", 
            "bns_symbol": "P4/nmm", 
            "description": "Ferromagnetic tetragonal"
        },
        (129, "antiferromagnetic"): {
            "bns_number": "129.2.2", 
            "bns_symbol": "P4/nm'm'", 
            "description": "Antiferromagnetic tetragonal"
        },
        
        # Orthorhombic systems (Pnma, #62)
        (62, "ferromagnetic"): {
            "bns_number": "62.1.1", 
            "bns_symbol": "Pnma", 
            "description": "Ferromagnetic orthorhombic (like many perovskites)"
        },
        (62, "antiferromagnetic"): {
            "bns_number": "62.2.2", 
            "bns_symbol": "Pn'ma'", 
            "description": "Antiferromagnetic orthorhombic"
        },
        
        # Monoclinic systems (P21/c, #14)
        (14, "ferromagnetic"): {
            "bns_number": "14.1.1", 
            "bns_symbol": "P2₁/c", 
            "description": "Ferromagnetic monoclinic"
        },
        (14, "antiferromagnetic"): {
            "bns_number": "14.2.2", 
            "bns_symbol": "P2₁/c'", 
            "description": "Antiferromagnetic monoclinic"
        },
        
        # Triclinic systems (P-1, #2)
        (2, "ferromagnetic"): {
            "bns_number": "2.1.1", 
            "bns_symbol": "P-1", 
            "description": "Ferromagnetic triclinic"
        },
        (2, "antiferromagnetic"): {
            "bns_number": "2.2.2", 
            "bns_symbol": "P-1'", 
            "description": "Antiferromagnetic triclinic"
        },
        
        # Triclinic systems (P1, #1)
        (1, "ferromagnetic"): {
            "bns_number": "1.1.1", 
            "bns_symbol": "P1", 
            "description": "Ferromagnetic triclinic (no inversion)"
        },
    }
    
    key = (conventional_sg, mag_type)
    if key in bns_lookup:
        return bns_lookup[key]
    
    # If not found in lookup, provide general guidance
    if mag_type == "ferromagnetic":
        return {
            "bns_number": f"{conventional_sg}.1.1", 
            "bns_symbol": f"SG#{conventional_sg} (FM)", 
            "description": f"Likely ferromagnetic variant of space group #{conventional_sg}"
        }
    elif mag_type in ["antiferromagnetic", "ferrimagnetic"]:
        return {
            "bns_number": f"{conventional_sg}.2.2", 
            "bns_symbol": f"SG#{conventional_sg} (AFM)", 
            "description": f"Likely magnetic variant of space group #{conventional_sg} with broken symmetry"
        }
    
    return None

def analyze_bns_magnetic_space_group(jid="JVASP-14839"):
    """
    Complete BNS magnetic space group analysis
    """
    try:
        results, struct = get_bns_magnetic_space_group(jid)
        
        print(f"=== BNS Magnetic Space Group Analysis for {jid} ===")
        print(f"Formula: {results['formula']}")
        print(f"Method: {results['method']}")
        
        if results['bns_number']:
            print(f"BNS Number: {results['bns_number']}")
            print(f"BNS Symbol: {results['bns_symbol']}")
            if results['og_number']:
                print(f"OG Number: {results['og_number']}")
        else:
            print("BNS number could not be determined automatically")
            print("Consider using specialized magnetic crystallography software")
        
        # Show magnetic moments
        print(f"\nMagnetic moments: {results['magnetic_moments']}")
        
        # Additional analysis
        sga = SpacegroupAnalyzer(struct)
        print(f"Conventional space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")
        
        return results, struct
        
    except Exception as e:
        print(f"Error analyzing {jid}: {str(e)}")
        return None, None

def get_magnetic_space_group_database_info():
    """
    Information about magnetic space group databases and tools
    """
    info = """
    === Magnetic Space Group Databases and Tools ===
    
    1. BNS Notation (Belov-Neronova-Smirnova):
       - 1,651 magnetic space groups total
       - Format: SGnumber.family.variant (e.g., 225.2.2)
       
    2. OG Notation (Opechowski-Guccione):
       - Alternative numbering system
       - Sometimes used in crystallographic software
    
    3. Databases:
       - MAGNDATA: https://www.cryst.ehu.es/magndata/
       - Bilbao Crystallographic Server
       - ISOTROPY Software Suite
    
    4. Software Tools:
       - spglib (Python): Basic magnetic space group detection
       - FINDSYM: Complete magnetic space group determination
       - SARAh: Magnetic structure analysis
       - FullProf: Refinement with magnetic space groups
    
    5. For definitive BNS determination:
       - Use specialized crystallographic software
       - Compare with experimental magnetic structures
       - Consult magnetic structure databases
    """
    print(info)

# Example usage and testing
if __name__ == "__main__":
    # Show database information
    get_magnetic_space_group_database_info()
    
    print("\n" + "="*50)
    
    # Analyze specific structure
    results, struct = analyze_bns_magnetic_space_group("JVASP-93786")
    
    # Test with other magnetic materials if available
    # test_jids = ["JVASP-816", "JVASP-23", "JVASP-1002"]  # Example magnetic materials
    
    # for test_jid in test_jids:
    #     try:
    #         print(f"\n" + "="*50)
    #         results, struct = analyze_bns_magnetic_space_group(test_jid)
    #     except:
    #         print(f"Could not analyze {test_jid}")

# Additional utility function for BNS lookup
def lookup_bns_by_conventional_sg_and_magnetic_type(sg_number, magnetic_type):
    """
    Quick lookup for common BNS numbers
    """
    common_bns = {
        # Format: (space_group, magnetic_type): BNS_info
        (225, "FM"): "225.1.1 - Fm-3m (ferromagnetic face-centered cubic)",
        (225, "AFM"): "225.2.2 - Fm'-3m' (antiferromagnetic, Type-II)",
        (221, "FM"): "221.1.1 - Pm-3m (ferromagnetic primitive cubic)",
        (221, "AFM"): "221.2.2 - Pm'-3m' (antiferromagnetic cubic)",
        (194, "FM"): "194.1.1 - P6₃/mmc (ferromagnetic hexagonal)",
        (194, "AFM"): "194.2.2 - P6₃/mm'c' (antiferromagnetic hexagonal)",
    }
    
    key = (sg_number, magnetic_type)
    return common_bns.get(key, f"BNS for SG{sg_number}+{magnetic_type} not in lookup table")