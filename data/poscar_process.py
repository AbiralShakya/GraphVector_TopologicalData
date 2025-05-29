import os
from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.io.vasp.inputs import Poscar
from typing import List, Dict, Tuple # Added typing for clarity

# --- (get_lines_for_one_poscar_block function remains the same as provided previously) ---
def get_lines_for_one_poscar_block(all_lines: list, start_line_index: int) -> Tuple[List[str], int]:
    """
    Identifies and returns the lines belonging to a single POSCAR block
    starting from start_line_index in all_lines.

    Args:
        all_lines (list): A list of all lines from the input file (pre-stripped).
        start_line_index (int): The starting index in all_lines for the current POSCAR.

    Returns:
        tuple: (list_of_lines_for_poscar, number_of_lines_consumed)
               Returns (None, 0) if a full POSCAR cannot be read (e.g., EOF or insufficient header).
    """
    if start_line_index >= len(all_lines):
        return [], 0 # Return empty list for lines if no block can be formed

    current_poscar_lines = []
    idx = start_line_index

    try:
        # 1. Comment line
        current_poscar_lines.append(all_lines[idx]); idx += 1
        # 2. Scaling factor
        current_poscar_lines.append(all_lines[idx]); idx += 1
        # 3. Lattice vectors (3 lines)
        for _ in range(3):
            current_poscar_lines.append(all_lines[idx]); idx += 1

        # 4. Element symbols (optional, VASP 5+) or counts
        line_for_symbols_or_counts = all_lines[idx].strip()
        is_vasp5_format = False
        if any(not s.isdigit() and s != '.' and s != '-' for s in line_for_symbols_or_counts.split()):
            is_vasp5_format = True
        
        if is_vasp5_format:
            current_poscar_lines.append(all_lines[idx]); idx += 1 # Element symbols
            counts_line_content = all_lines[idx].strip()
            current_poscar_lines.append(all_lines[idx]); idx += 1 # Element counts
        else: 
            counts_line_content = line_for_symbols_or_counts
            current_poscar_lines.append(all_lines[idx]); idx += 1 # Element counts
        
        current_poscar_lines.append(all_lines[idx]); idx += 1 # Coordinate type line
        coord_type_line_content = all_lines[idx-1].strip().lower()

        try:
            num_atoms = sum(map(int, counts_line_content.split()))
        except ValueError:
            return [], 0 

        if num_atoms == 0: 
            return current_poscar_lines, (idx - start_line_index)

        for _ in range(num_atoms):
            current_poscar_lines.append(all_lines[idx]); idx += 1
        
        lines_consumed_so_far = idx - start_line_index
        
        # Note: Selective dynamics and velocities are implicitly handled by Pymatgen's Poscar.from_string
        # if the lines are included in the block. The `lines_consumed_so_far` currently
        # counts up to the end of atom coordinates. Pymatgen's parser might interpret
        # further lines if they conform to velocity or selective dynamics flags following coordinates.
        # For strict block definition based on this function, we stop counting lines after coordinates.

        return current_poscar_lines, lines_consumed_so_far

    except IndexError:
        return [], 0 # Return empty list for lines if EOF reached prematurely
    except Exception:
        return [], 0 # Return empty list for lines on other errors


def process_multi_poscar_file(filepath: str, parse_only_first: bool = False) -> List[Dict]: # Added parse_only_first flag
    """
    Processes a single file containing multiple concatenated POSCAR structures.

    Args:
        filepath (str): The path to the local file.
        parse_only_first (bool): If True, only parses the first successfully found structure.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains
                    parsed information for POSCAR structure(s) found in the file.
    """
    parsed_structures_data = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return parsed_structures_data

    try:
        with open(filepath, 'r') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return parsed_structures_data

    if not all_lines:
        print(f"Warning: File '{filepath}' is empty or contains only whitespace.")
        return parsed_structures_data

    current_line_offset = 0
    structure_index_in_file = 0

    while current_line_offset < len(all_lines):
        poscar_lines_block, num_lines_in_block = get_lines_for_one_poscar_block(all_lines, current_line_offset)

        if not poscar_lines_block or num_lines_in_block == 0:
            if current_line_offset < len(all_lines):
                print(f"Info: Could not read a complete POSCAR header starting near line {current_line_offset + 1}. End of readable structures or formatting issue.")
            break 

        poscar_string = "\n".join(poscar_lines_block)

        try:
            poscar_obj = Poscar.from_string(poscar_string)
            structure = poscar_obj.structure

            parsed_structures_data.append({
                "index_in_file": structure_index_in_file,
                "comment": poscar_obj.comment,
                "formula": structure.composition.reduced_formula,
                "natoms": len(structure),
                "lattice_vectors": structure.lattice.matrix.tolist(),
                "sites": [{"species": site.species_string,
                           "xyz_frac": site.frac_coords.tolist(),
                           "xyz_cart": site.coords.tolist()}
                          for site in structure.sites],
                "selective_dynamics": poscar_obj.selective_dynamics,
                "velocities": poscar_obj.velocities.tolist() if poscar_obj.velocities is not None else None,
                "pymatgen_structure": structure
            })
            structure_index_in_file += 1
            current_line_offset += num_lines_in_block

            if parse_only_first: # MODIFICATION: Exit after the first successful parse
                print("Info: Successfully parsed the first structure. Stopping as requested.")
                break

        except Exception as e:
            print(f"Error parsing POSCAR block #{structure_index_in_file} (lines {current_line_offset + 1} - {current_line_offset + num_lines_in_block +1}): {e}")
            print(f"Problematic block content:\n---\n{poscar_string}\n---")
            if num_lines_in_block > 0:
                 current_line_offset += num_lines_in_block
            else:
                 current_line_offset += 1 
            print(f"Attempting to continue from next potential block (line offset {current_line_offset + 1}).")
            if parse_only_first: # If an error occurs on the first attempt and we only want the first
                print("Info: Error occurred on the first structure attempt. Stopping as requested.")
                break


    return parsed_structures_data

if __name__ == '__main__':
    # --- Create a dummy multi-POSCAR file for testing ---
    dummy_file_content = """\
Cubic Diamond (First Structure)
1.0
3.56 0.00 0.00
0.00 3.56 0.00
0.00 0.00 3.56
C
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
FCC Al (Second Structure)
1.0
4.05 0.00 0.00
0.00 4.05 0.00
0.00 0.00 4.05
Al
4
Direct
0.00 0.00 0.00
0.50 0.50 0.00
0.00 0.50 0.50
0.50 0.00 0.50
NaCl Structure with Selective Dynamics (Third Structure)
1.0
5.64 0.00 0.00
0.00 5.64 0.00
0.00 0.00 5.64
Na Cl
4 4
Selective dynamics
Direct
0.00 0.00 0.00 T T F  ! Na1
0.50 0.50 0.00 F F F  ! Na2
0.00 0.50 0.50 T F T  ! Na3
0.50 0.00 0.50 F T F  ! Na4
0.50 0.00 0.00         ! Cl1
0.00 0.50 0.00         ! Cl2
0.50 0.50 0.50         ! Cl3
0.00 0.00 0.50         ! Cl4
"""
    test_filepath = "concatenated_poscars_test_first.vasp"
    with open(test_filepath, "w") as f:
        f.write(dummy_file_content)

    print(f"--- Processing dummy file ({test_filepath}) to get ONLY THE FIRST structure ---")
    # MODIFICATION: Call with parse_only_first=True
    first_structure_info_list = process_multi_poscar_file(test_filepath, parse_only_first=True)

    if first_structure_info_list: # Should contain one item if successful
        info = first_structure_info_list[0]
        print(f"\n--- Parsed First Structure (Index #{info['index_in_file']}) ---")
        print(f"Comment: {info['comment']}")
        print(f"Formula: {info['formula']}")
        print(f"Number of atoms: {info['natoms']}")
        print(f"Lattice Vectors: {info['lattice_vectors'][0]}")
        print(f"First site: {info['sites'][0]['species']} @ {info['sites'][0]['xyz_frac']}")
    elif len(first_structure_info_list) == 0:
        print("No structures were parsed (or an error occurred on the first attempt).")


    print(f"\n--- Processing dummy file ({test_filepath}) to get ALL structures (for comparison) ---")
    all_structures_info_list = process_multi_poscar_file(test_filepath, parse_only_first=False)
    print(f"Total structures parsed when parse_only_first=False: {len(all_structures_info_list)}")
    if len(all_structures_info_list) > 0:
        print(f"Comment of first structure: {all_structures_info_list[0]['comment']}")
    if len(all_structures_info_list) > 1:
        print(f"Comment of second structure: {all_structures_info_list[1]['comment']}")


    # Clean up dummy file
    # os.remove(test_filepath)