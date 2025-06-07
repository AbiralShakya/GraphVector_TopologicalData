import sqlite3
import argparse
import re
import sys
import os
from pathlib import Path

def parse_kpoint_cells(irreps_txt):
    """
    Parses a string containing multiple k-point irrep cells into a list of individual cell strings.
    Handles cells like "A1(2)", "2 Γ3Γ4(2)", "D3D3(2) ⊕ D4D4(2)", 
    "Γ1(1) ⊕ Γ2(1) ⊕ Γ3(1) ⊕ Γ4(1)", "2 X2X3(2) ⊕ 2 X4X5(2)",
    and a placeholder like "-" as single tokens.
    """
    if not irreps_txt:
        return []

    # Define a pattern for a single irrep unit
    irrep_unit_pattern = r"[A-ZΓa-z0-9+\-↑↓]+?\([^\)]+\)" 

    # Define a "component" pattern: a simple irrep_unit OR a numbered irrep_unit.
    component_pattern = rf"(?:\d+\s+{irrep_unit_pattern}|{irrep_unit_pattern})"
    
    # Define the full cell pattern:
    # Starts with an initial component, followed by zero or more "⊕ component" sequences.
    # OR it's our special placeholder for a "button" / missing decomposable irrep string.
    placeholder_pattern = r"(-)" # Placeholder is a hyphen

    full_cell_pattern_str = rf"((?:{component_pattern})(?:\s*⊕\s*(?:{component_pattern}))*|{placeholder_pattern})"
    
    cell_parser_regex = re.compile(full_cell_pattern_str, re.UNICODE)
    
    cells = []
    remaining_text = irreps_txt.strip()
    while remaining_text:
        match = cell_parser_regex.match(remaining_text)
        if match:
            token = match.group(0).strip() 
            cells.append(token)
            remaining_text = remaining_text[match.end():].strip()
        else:
            if remaining_text:
                # Fallback if main regex fails: try to take the next non-whitespace chunk
                # This helps with potential edge cases or slightly malformed simple tokens
                # not perfectly caught by the more complex regex.
                parts = remaining_text.split(maxsplit=1)
                if parts:
                    cells.append(parts[0])
                    remaining_text = parts[1].strip() if len(parts) > 1 else ""
                    # print(f"Warning: Used fallback split for token '{parts[0]}' near '{remaining_text[:30]}'", file=sys.stderr)
                else: # Should not happen if remaining_text is non-empty
                    raise ValueError(f"Could not parse k-point cell structure (fallback failed) near: '{remaining_text[:50]}'")

            else: # No text left
                break
    return cells

class MagneticEBRDatabaseManager:
    def __init__(self, db_path="pebr_magnetic.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        print(f"Magnetic DB Manager connected to '{db_path}'.")
        self.create_tables()

    def close(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS magnetic_space_groups (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        bns_number  TEXT UNIQUE NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS magnetic_ebrs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        space_group_id INTEGER NOT NULL REFERENCES magnetic_space_groups(id) ON DELETE CASCADE,
        wyckoff_letter TEXT    NOT NULL,
        site_symmetry  TEXT    NOT NULL,
        orbital_label  TEXT    NOT NULL,
        orbital_multiplicity INTEGER NOT NULL,
        notes          TEXT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("DROP TABLE IF EXISTS magnetic_ebr_decomposition_branches;") # Keep this for clean slate
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS magnetic_ebr_decomposition_items (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        space_group_id      INTEGER NOT NULL REFERENCES magnetic_space_groups(id) ON DELETE CASCADE,
        ebr_id              INTEGER NOT NULL REFERENCES magnetic_ebrs(id) ON DELETE CASCADE,
        decomposition_index INTEGER NOT NULL,
        branch_index        INTEGER NOT NULL,
        irreps_string       TEXT    NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS magnetic_irreps (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        space_group_id INTEGER NOT NULL REFERENCES magnetic_space_groups(id) ON DELETE CASCADE,
        ebr_id         INTEGER NOT NULL REFERENCES magnetic_ebrs(id) ON DELETE CASCADE,
        k_point        TEXT    NOT NULL,
        irrep_label    TEXT    NOT NULL,
        multiplicity   INTEGER
        );
        """)
        self.conn.commit()
        
        # --- FIX 2: Point triggers to the correct MAGNETIC table names ---
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_magnetic_space_groups_updated_at
        AFTER UPDATE ON magnetic_space_groups FOR EACH ROW
        BEGIN UPDATE magnetic_space_groups SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
        """)
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_magnetic_ebrs_updated_at
        AFTER UPDATE ON magnetic_ebrs FOR EACH ROW
        BEGIN UPDATE magnetic_ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
        """)
        self.conn.commit()
    
    def get_all_scraped_bns_numbers(self) -> set:
        cursor = self.conn.cursor()
        cursor.execute("SELECT bns_number FROM magnetic_space_groups")
        return {row[0] for row in cursor.fetchall()}

    def parse_wyckoff(self, entry):
        m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
        return (m.group(1), m.group(2)) if m else (entry, "")

    def parse_orbital(self, entry):
        m = re.match(r"^(.+)\((\d+)\)$", entry)
        return (m.group(1), int(m.group(2))) if m else (entry, 1)

    def parse_multiplicity(self, entry):
        m = re.search(r"\((\d+)\)$", entry)
        return int(m.group(1)) if m else None

    def parse_irrep_label(self, entry):
        m = re.match(r"^(.+)\(\d+\)$", entry)
        return m.group(1) if m else entry

    def _parse_file_content(self, raw_lines): # Assumed from previous, ensure it's correct
        joined = "\n".join(raw_lines)
        joined = re.sub(r'\\\s*\n\s*', ' ', joined)
        lines = [line.rstrip("\n") for line in joined.splitlines() if line.strip()]
        wyckoff_line, orbital_line, notes_line, kpoint_lines = None, None, None, []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("Wyckoff pos."):
                i += 1
                while i < len(lines) and not lines[i].strip(): i += 1
                if i < len(lines): wyckoff_line = lines[i]
            elif line.startswith("Band-Rep."):
                i += 1
                while i < len(lines) and not lines[i].strip(): i += 1
                if i < len(lines): orbital_line = lines[i]
            elif line.startswith("Decomposable"):
                i += 1 
                while i < len(lines) and not lines[i].strip(): i += 1
                if i < len(lines):
                    potential_header_line = lines[i]
                    if i + 1 < len(lines):
                        potential_notes_line = lines[i+1]
                        if len(potential_header_line.split()) <= 2 and len(potential_notes_line.split()) > len(potential_header_line.split()):
                             if any(kw in potential_header_line.lower() for kw in ["decomposable", "indecomposable"]):
                                i += 1 
                while i < len(lines) and not lines[i].strip(): i+=1
                if i < len(lines): notes_line = lines[i]
                i += 1
                while i < len(lines):
                    if lines[i].strip(): kpoint_lines.append(lines[i])
                    i += 1
                break 
            i += 1
        if not all([wyckoff_line, orbital_line]):
            missing = [p for p, v in [("Wyckoff pos.", wyckoff_line), ("Band-Rep.", orbital_line)] if not v]
            raise ValueError(f"Could not locate required sections in file: {', '.join(missing)}")
        return wyckoff_line, orbital_line, notes_line if notes_line else "", kpoint_lines
    
    def _parse_and_insert_from_text(self, sg_number, raw_text_data):
        """Takes the structured text block and performs parsing and database insertion."""
        lines = [line.strip() for line in raw_text_data.strip().split('\n') if line.strip()]
        
        sections = {}
        current_header = None
        header_pattern = re.compile(r'^(Wyckoff pos\.|Band-Rep\.|Decomposable)')
        
        # This logic correctly groups lines under their respective headers
        for line in lines:
            match = header_pattern.match(line)
            if match:
                current_header = match.group(1)
                sections[current_header] = [line[len(current_header):].strip()]
            elif re.match(r'^[A-ZΓ]+:', line):
                current_header = "kpoints"
                if current_header not in sections: sections[current_header] = []
                sections[current_header].append(line)
            elif current_header and current_header != "kpoints":
                sections[current_header].append(line)

        wyckoff_entries = " ".join(sections.get('Wyckoff pos.', [])).split()
        orbital_entries = " ".join(sections.get('Band-Rep.', [])).split()
        notes_entries = " ".join(sections.get('Decomposable', [])).split()
        kpoint_lines = sections.get('kpoints', [])

        if not wyckoff_entries or not orbital_entries:
            raise ValueError("Failed to parse Wyckoff or Band-Rep lines from scraped text.")

        num_cols = len(wyckoff_entries)

        kpoint_data_final = []
        for kp_line_str in kpoint_lines:
            parts = kp_line_str.split(':', 1)
            kp_label = parts[0].strip()
            irreps_text = parts[1].strip() if len(parts) > 1 else ""
            cells = parse_kpoint_cells(irreps_text)
            if len(cells) != num_cols:
                raise ValueError(f"Mismatch at k-point {kp_label}. Expected {num_cols} irrep columns, found {len(cells)} in '{irreps_text}'.")
            kpoint_data_final.append((kp_label, cells))

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
        sg_row = cursor.fetchone()
        sg_id = sg_row[0] if sg_row else cursor.execute("INSERT INTO space_groups(number) VALUES (?)", (sg_number,)).lastrowid
        
        cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
        
        for j in range(num_cols):
            wyck_letter, site_sym = self.parse_wyckoff(wyckoff_entries[j])
            orb_label, orb_mult = self.parse_orbital(orbital_entries[j])
            note = notes_entries[j] if j < len(notes_entries) else "indecomposable"

            cursor.execute("""
                INSERT INTO ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes, time_reversal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sg_id, wyck_letter, site_sym, orb_label, orb_mult, note, 1))
            ebr_id = cursor.lastrowid
            
            for (kp_label, cells) in kpoint_data_final:
                full_irrep_str = cells[j]
                mult = self.parse_multiplicity(full_irrep_str)
                label_no_mult = self.parse_irrep_label(full_irrep_str)
                cursor.execute("""
                    INSERT INTO irreps (ebr_id, k_point, irrep_label, multiplicity)
                    VALUES (?, ?, ?, ?)
                """, (ebr_id, kp_label, label_no_mult, mult))
        
        self.conn.commit()
        print(f"✅ Successfully ingested {num_cols} EBRs for SG {sg_number} into the database.")
        return True

    def _parse_raw_text(self, raw_text): # Assumed from previous, ensure it's correct
        text = re.sub(r'\\\s*\n\s*', ' ', raw_text.strip())
        wyckoff_match = re.search(r'Wyckoff pos\.(.*?)(?=Band-Rep\.)', text, re.IGNORECASE | re.DOTALL)
        orbital_match = re.search(r'Band-Rep\.(.*?)(?=Decomposable)', text, re.IGNORECASE | re.DOTALL)
        if not wyckoff_match: raise ValueError("Could not find: Wyckoff pos.")
        if not orbital_match: raise ValueError("Could not find: Band-Rep. (ensure 'Decomposable' follows)")

        wyckoff_line = re.sub(r'\s+', ' ', wyckoff_match.group(1).strip())
        orbital_line = re.sub(r'\s+', ' ', orbital_match.group(1).strip())
        expected_cols = len(wyckoff_line.split())
        notes_line = ""
        kpoint_section_start_match = re.search(r'([A-ZΓ]+:\s*\([^)]+\))', text, re.IGNORECASE)
        
        if kpoint_section_start_match:
            kpoint_start_index = kpoint_section_start_match.start()
            decomp_keyword_pattern = r'Decomposable\\?\s*'
            decomposable_intro_match = re.search(decomp_keyword_pattern, text[:kpoint_start_index], re.IGNORECASE | re.DOTALL)
            if decomposable_intro_match:
                potential_notes_section = text[decomposable_intro_match.end():kpoint_start_index].strip()
                decomp_lines = [ln.strip() for ln in potential_notes_section.split('\n') if ln.strip()]
                found_notes = None
                if decomp_lines:
                    if len(decomp_lines) == 1 and (len(decomp_lines[0].split()) == expected_cols or expected_cols == 0):
                        found_notes = decomp_lines[0]
                    elif len(decomp_lines) > 1:
                        if len(decomp_lines[-1].split()) == expected_cols: found_notes = decomp_lines[-1]
                        else:
                            for ln_idx_rev in range(len(decomp_lines) - 1, -1, -1):
                                if len(decomp_lines[ln_idx_rev].split()) == expected_cols:
                                    found_notes = decomp_lines[ln_idx_rev]; break
                notes_line = found_notes if found_notes else ""
                if not notes_line and decomp_lines and not (len(decomp_lines[-1].split()) == 1 and decomp_lines[-1].lower() in ["indecomposable", "decomposable"]):
                    notes_line = decomp_lines[-1]
        else: raise ValueError("K-point data not found (needed to delimit Decomposable section).")

        if not kpoint_section_start_match: raise ValueError("K-point data not found (logic error).")
        kpoint_text_block = text[kpoint_section_start_match.start():]
        kpoint_raw_lines = [ln.strip() for ln in kpoint_text_block.split('\n') if ln.strip()]
        kpoint_data = []
        for kp_raw_line in kpoint_raw_lines:
            match_label_coords = re.match(r'([A-ZΓ]+):\s*\(([^)]+)\)\s*', kp_raw_line)
            if match_label_coords:
                kpt_label = match_label_coords.group(1)
                irreps_txt_for_kpoint = kp_raw_line[match_label_coords.end():].strip()
                cells = parse_kpoint_cells(irreps_txt_for_kpoint)
                if expected_cols > 0 and len(cells) != expected_cols:
                    raise ValueError(f"K-point {kpt_label}: {len(cells)} irreps, expected {expected_cols}. Line: '{kp_raw_line}'")
                kpoint_data.append((kpt_label, cells))
        if not kpoint_data and expected_cols > 0: raise ValueError("No k-point data lines parsed.")
        return wyckoff_line, orbital_line, notes_line, kpoint_data
        
    def get_decomposable_ebrs(self, sg_id_or_num):
        cursor = self.conn.cursor()
        sg_id = sg_id_or_num
        if isinstance(sg_id_or_num, int) and sg_id_or_num <= 230: 
            cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_id_or_num,))
            sg_row = cursor.fetchone()
            if not sg_row: return []
            sg_id = sg_row[0]
        # Fetch the new orbital_multiplicity column
        cursor.execute("""
            SELECT id, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes 
            FROM ebrs 
            WHERE space_group_id = ? AND lower(notes) = 'decomposable' 
            ORDER BY id
        """, (sg_id,))
        return cursor.fetchall()
    
    def _calculate_irrep_string_dimension(self, irrep_string): 
        """Calculates the total dimension of a comma-separated irrep string."""
        if not irrep_string or not irrep_string.strip():
            return 0
        
        total_dimension = 0
        irrep_parts = [part.strip() for part in irrep_string.split(',')]
        
        for part in irrep_parts:
            # Check for leading multiplicity, e.g., "2 A1(1)"
            match = re.match(r"(\d+)\s+(.*)", part)
            if match:
                leading_mult = int(match.group(1))
                irrep_name = match.group(2)
            else:
                leading_mult = 1
                irrep_name = part

            # Get dimension of the irrep itself, e.g., the (2) in "G(2)"
            dim = self.parse_multiplicity(irrep_name)
            if dim is None:
                dim = 1 # Assume dimension is 1 if not specified (e.g., "A1")
            
            total_dimension += leading_mult * dim
            
        return total_dimension

    def add_decomposition_item(self, sg_id, ebr_id, decomp_idx, branch_idx, irrep_string):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO magnetic_ebr_decomposition_items
            (space_group_id, ebr_id, decomposition_index, branch_index, irreps_string)
            VALUES (?, ?, ?, ?, ?)
        """, (sg_id, ebr_id, decomp_idx, branch_idx, irrep_string))
        self.conn.commit()


    def get_ebr_decomposition_branches(self, ebr_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT decomposition_index, branch1_irreps, branch2_irreps, created_at FROM ebr_decomposition_branches WHERE ebr_id = ? ORDER BY decomposition_index", (ebr_id,))
        return cursor.fetchall()

    def delete_ebr_decomposition_branch(self, ebr_id, decomposition_index):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM ebr_decomposition_branches WHERE ebr_id = ? AND decomposition_index = ?", (ebr_id, decomposition_index))
        rc = cursor.rowcount
        self.conn.commit()
        # if rc > 0: cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)); self.conn.commit()
        return rc > 0

    def delete_all_ebr_decomposition_branches(self, ebr_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM ebr_decomposition_branches WHERE ebr_id = ?", (ebr_id,))
        rc = cursor.rowcount
        self.conn.commit()
        # if rc > 0: cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)); self.conn.commit()
        return rc > 0

    def _insert_data(self, bns_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data_list):
        from PEBR_TR_nonmagnetic_query import EBRDatabaseManager # To access parsing helpers
        parser_helpers = EBRDatabaseManager() # A bit of a hack to reuse parsing functions
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM magnetic_space_groups WHERE bns_number = ?", (bns_number,))
        sg_row = cursor.fetchone()
        sg_id = sg_row[0] if sg_row else cursor.execute(
            "INSERT INTO magnetic_space_groups(bns_number) VALUES (?)", (bns_number,)
        ).lastrowid

        cursor.execute("DELETE FROM magnetic_ebrs WHERE space_group_id = ?", (sg_id,))
        self.conn.commit()

        inserted_ebrs = []
        for j, wyck_entry in enumerate(wyckoff_entries):
            wyck_letter, site_sym = parser_helpers.parse_wyckoff(wyck_entry)
            orb_label, orb_mult = parser_helpers.parse_orbital(orbital_entries[j])
            note = notes_entries[j] if j < len(notes_entries) else "indecomposable"

            cursor.execute("""
                INSERT INTO magnetic_ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sg_id, wyck_letter, site_sym, orb_label, orb_mult, note))
            ebr_id = cursor.lastrowid
            inserted_ebrs.append({'ebr_id': ebr_id, 'note': note})

            for kp_label, cells in kpoint_data_list:
                full_irrep_str = cells[j]
                mult = parser_helpers.parse_multiplicity(full_irrep_str)
                label_no_mult = parser_helpers.parse_irrep_label(full_irrep_str)
                cursor.execute("""
                    INSERT INTO magnetic_irreps (space_group_id, ebr_id, k_point, irrep_label, multiplicity)
                    VALUES (?, ?, ?, ?, ?)
                """, (sg_id, ebr_id, kp_label, label_no_mult, mult))
        
        self.conn.commit()
        parser_helpers.close()
        return inserted_ebrs, sg_id
    
    def validate_input_file(self, filepath):
        """Validate that input file exists and has expected format."""
        if not os.path.exists(filepath):
            return False, f"File '{filepath}' not found."
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for required sections
            required_sections = ["Wyckoff pos.", "Band-Rep.", "Decomposable"]
            missing = [section for section in required_sections if section not in content]
            
            if missing:
                return False, f"Missing required sections: {', '.join(missing)}"
            
            return True, "File format appears valid."
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def ingest_file(self, sg_number, filepath, prompt_for_branches_immediately=False):
        if not (1 <= sg_number <= 230): raise ValueError(f"SG number {sg_number} out of range 1-230")
        is_valid, msg = self.validate_input_file(filepath)
        if not is_valid: raise ValueError(msg)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                # Filter out empty lines before passing to _parse_file_content
                raw_lines = [line.rstrip("\n") for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Error reading file '{filepath}': {str(e)}")

        if not raw_lines:
            raise ValueError(f"Input file '{filepath}' is empty or contains only whitespace.")

        wyckoff_line, orbital_line, notes_line, kpoint_lines_parsed = self._parse_file_content(raw_lines)
        
        wyckoff_entries = wyckoff_line.split()
        orbital_entries = orbital_line.split()
        # notes_line can be empty if not found, handle it
        notes_entries = notes_line.split() if notes_line else [] 
        
        num_cols = len(wyckoff_entries)

        # Column consistency checks
        if not (len(orbital_entries) == num_cols):
            raise ValueError(f"Column count mismatch: Wyckoff({num_cols}), Orbital({len(orbital_entries)})")
        
        # Notes entries consistency (allow notes_entries to be empty if notes_line was empty)
        if notes_line and len(notes_entries) != num_cols:
            if len(notes_entries) > num_cols:
                notes_entries = notes_entries[-num_cols:]
            # If notes_entries is shorter but not empty, it's an issue unless it's a single token meant for all
            elif len(notes_entries) > 0 and len(notes_entries) < num_cols and not (len(notes_entries) == 1 and num_cols > 1):
                 raise ValueError(f"Notes line has {len(notes_entries)} tokens, expected {num_cols} or 1 (if applying to all). Notes: '{notes_line}'")

        kpoint_data_final = []
        for kp_line_str in kpoint_lines_parsed:
            tokens = kp_line_str.split()
            if tokens: 
                # tokens[1:] should be a list of cell strings for this k-point row
                kpoint_data_final.append((tokens[0].rstrip(":"), tokens[1:]))
        
        inserted_ebrs = self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data_final)
        
        if prompt_for_branches_immediately:
            print(f"\n--- Checking {len(inserted_ebrs)} imported EBRs for branch input (SG {sg_number}) ---")
            for ebr_info in inserted_ebrs:
                # Trigger if note is "decomposable" (add other conditions like specific placeholders if needed)
                if ebr_info['note'].lower() == 'decomposable': 
                    self._manage_branches_interactively_for_ebr(
                        ebr_info['ebr_id'], 
                        f"{ebr_info['wyckoff']}{ebr_info['site_symmetry']}", 
                        ebr_info['orbital'],
                        called_from_ingest=True
                    )
        return len(inserted_ebrs)

    def ingest_text(self, sg_number, raw_text, prompt_for_branches_immediately=False):
        if not (1 <= sg_number <= 230): raise ValueError(f"SG number {sg_number} out of range 1-230")
        if not raw_text or not raw_text.strip(): raise ValueError("Raw text cannot be empty")
        
        wyckoff_line, orbital_line, notes_line, kpoint_data_parsed = self._parse_raw_text(raw_text) 
        
        wyckoff_entries = wyckoff_line.split()
        orbital_entries = orbital_line.split()
        notes_entries = notes_line.split() if notes_line else []
        
        num_cols = len(wyckoff_entries)

        if not (len(orbital_entries) == num_cols):
            raise ValueError(f"Column count mismatch: Wyckoff({num_cols}), Orbital({len(orbital_entries)})")

        if notes_line and len(notes_entries) != num_cols:
            if len(notes_entries) > num_cols:
                notes_entries = notes_entries[-num_cols:]
            elif len(notes_entries) > 0 and len(notes_entries) < num_cols and not (len(notes_entries) == 1 and num_cols > 1):
                 raise ValueError(f"Notes line has {len(notes_entries)} tokens, expected {num_cols} or 1. Notes: '{notes_line}'")
        
        inserted_ebrs = self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data_parsed)
        
        if prompt_for_branches_immediately:
            print(f"\n--- Checking {len(inserted_ebrs)} imported EBRs for branch input (SG {sg_number}) ---")
            for ebr_info in inserted_ebrs:
                if ebr_info['note'].lower() == 'decomposable': 
                    self._manage_branches_interactively_for_ebr(
                        ebr_info['ebr_id'], 
                        f"{ebr_info['wyckoff']}{ebr_info['site_symmetry']}", 
                        ebr_info['orbital'],
                        called_from_ingest=True
                    )
        return len(inserted_ebrs)

    def _manage_branches_interactively_for_ebr(self, ebr_id, wyckoff_info, orbital_info, called_from_ingest=False):
        """
        Manages decomposition branches for a single specified EBR.
        wyckoff_info should be a string like "1a(2/m)".
        orbital_info should be the orbital label string.
        """
        print(f"\n--- Managing Branches for EBR ID: {ebr_id} ({wyckoff_info} - {orbital_info}) ---")
        
        if called_from_ingest:
            choice = input("This EBR is marked 'decomposable'. Add/edit decomposition branches now? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"ℹ️ Branch input for EBR ID {ebr_id} skipped. Manage later via main menu option if needed.")
                return

        while True:
            existing_branches = self.get_ebr_decomposition_branches(ebr_id)
            if existing_branches:
                print("  Existing Branches:")
                for br_idx, b1, b2, ca in existing_branches:
                    print(f"    Index {br_idx}: Branch1='{b1}', Branch2='{b2}' (Added: {ca})")
            else:
                print("  No existing branches for this EBR.")

            # ADDED (B)ulk add option
            branch_action = input("\n  Branch actions: (A)dd one, (B)ulk add from paste, (D)elete specific, (C)lear all, (R)eturn/Done: ").strip().upper()
            
            if branch_action == 'A': # Add one by one
                try:
                    dec_idx_str = input("    Enter decomposition index (e.g., 1, 2): ").strip()
                    if not dec_idx_str.isdigit():
                        print("❌ Decomposition index must be a number.")
                        continue
                    dec_idx = int(dec_idx_str)
                    
                    b1_str = input(f"    Enter Branch 1 irreps string for index {dec_idx} (comma-separated): ").strip()
                    b2_str = input(f"    Enter Branch 2 irreps string for index {dec_idx} (comma-separated): ").strip()
                    
                    if not b1_str or not b2_str: 
                        print("❌ Branch strings cannot be empty.")
                        continue
                    
                    if self.add_ebr_decomposition_branch(ebr_id, dec_idx, b1_str, b2_str):
                        print(f"✅ Branch index {dec_idx} added/updated for EBR ID {ebr_id}.")
                    # add_ebr_decomposition_branch prints its own error on failure
                except ValueError: 
                    print("❌ Invalid input for decomposition index (must be a number).")
                except Exception as e:
                     print(f"❌ Error during add operation: {e}")

            elif branch_action == 'B': # NEW: Bulk add
                print("\n  Paste branch data below. Format each line as:")
                print("  <index> <Branch1_Irreps_String> <Branch2_Irreps_String>")
                print("  (Separated by multiple spaces or tabs. Branch strings themselves should not contain tabs or multiple consecutive spaces if they are not part of the irrep name).")
                print("  Example: 1   X3X4,S3S4   X3X4,S3S4")
                print("  Enter a blank line (or Ctrl+D/Ctrl+Z+Enter) to finish pasting.")
                
                bulk_lines = []
                while True:
                    try:
                        line = input("  > ")
                        if not line.strip(): # Empty line signifies end of bulk paste
                            break
                        bulk_lines.append(line)
                    except EOFError:
                        break
                
                if not bulk_lines:
                    print("ℹ️ No bulk data pasted.")
                    continue

                added_count = 0
                updated_count = 0
                error_lines = []

                clear_existing_choice = input("  Clear all existing branches for this EBR before bulk adding? (y/n, default n): ").strip().lower()
                if clear_existing_choice == 'y':
                    deleted_count = self.delete_all_ebr_decomposition_branches(ebr_id)
                    print(f"  Cleared {deleted_count} existing branches.")
                    existing_branches = [] # Refresh local view

                for i, line_data in enumerate(bulk_lines):
                    parts = re.split(r'\s+', line_data.strip(), 2) # Split by any whitespace, max 2 splits
                    if len(parts) == 3:
                        idx_str, b1, b2 = parts
                        try:
                            dec_idx = int(idx_str)
                            # Check if this index already exists for 'update' count
                            is_update = any(br[0] == dec_idx for br in existing_branches)

                            if self.add_ebr_decomposition_branch(ebr_id, dec_idx, b1, b2):
                                if is_update and clear_existing_choice != 'y': # only count as update if not cleared
                                    updated_count += 1
                                else:
                                    added_count +=1
                            else:
                                error_lines.append(f"Line {i+1} (DB Error): {line_data}")
                        except ValueError:
                            error_lines.append(f"Line {i+1} (Bad Index): {line_data}")
                    else:
                        error_lines.append(f"Line {i+1} (Bad Format): {line_data}")
                
                print(f"\n  Bulk Add Summary for EBR ID {ebr_id}:")
                print(f"    Successfully added: {added_count} new branches.")
                if updated_count > 0 : print(f"    Successfully updated: {updated_count} existing branches.")
                if error_lines:
                    print(f"    Errors on {len(error_lines)} lines:")
                    for err_line in error_lines:
                        print(f"      {err_line}")

            elif branch_action == 'D': # Delete specific
                try:
                    dec_idx_del_str = input("    Enter decomposition index to delete: ").strip()
                    if not dec_idx_del_str.isdigit():
                        print("❌ Decomposition index must be a number.")
                        continue
                    dec_idx_del = int(dec_idx_del_str)
                    
                    if self.delete_ebr_decomposition_branch(ebr_id, dec_idx_del):
                        print(f"✅ Branch index {dec_idx_del} deleted for EBR ID {ebr_id}.")
                    else: 
                        print(f"ℹ️ Branch index {dec_idx_del} not found or could not be deleted.");
                except ValueError: 
                    print("❌ Invalid input for decomposition index (must be a number).")
                except Exception as e:
                    print(f"❌ Error during delete operation: {e}")
            
            elif branch_action == 'C': # Clear all
                confirm_clear = input(f"⚠️ Are you sure you want to delete ALL branches for EBR ID {ebr_id}? (yes/no): ").strip().lower()
                if confirm_clear == 'yes':
                    count = self.delete_all_ebr_decomposition_branches(ebr_id)
                    print(f"✅ Deleted {count} branches for EBR ID {ebr_id}.")
                else: 
                    print("ℹ️ Clear operation cancelled.")
            
            elif branch_action == 'R': 
                print(f"Finished managing branches for EBR ID {ebr_id}.")
                break
            
            else: 
                print("❌ Invalid branch action. Please choose A, B, D, C, or R.")
    
    
    # ... (get_next_space_group, get_database_status, query_space_group as in previous full code) ...
    def get_next_space_group(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(number) FROM space_groups WHERE number <= 230")
        r = cursor.fetchone()
        if r[0] is None: return 1
        return r[0] + 1 if r[0] < 230 else None

    def get_database_status(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM space_groups WHERE number <= 230")
        sg_c = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ebrs")
        ebr_c = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM irreps")
        irrep_c = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ebr_decomposition_branches")
        branch_c = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(number), MAX(number) FROM space_groups WHERE number <= 230")
        min_sg, max_sg = cursor.fetchone()
        cursor.execute("""WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM nums WHERE n<230) SELECT GROUP_CONCAT(n) FROM nums WHERE n NOT IN (SELECT number FROM space_groups WHERE number <= 230)""")
        missing = cursor.fetchone()[0]
        return {'space_groups': sg_c, 'ebrs': ebr_c, 'irreps': irrep_c, 'branches': branch_c, 'min_sg': min_sg or 0, 'max_sg': max_sg or 0, 'missing_sgs': missing.split(',') if missing else []}

    def query_space_group(self, sg_number):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM space_groups WHERE number = ?", (sg_number,))
        sg_data = cursor.fetchone()
        if not sg_data: return None
        cursor.execute("SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes, created_at, updated_at FROM ebrs WHERE space_group_id = ? ORDER BY id", (sg_data[0],))
        ebrs_rows = cursor.fetchall()
        result = {'space_group': sg_data, 'ebrs': []}
        for ebr_row in ebrs_rows:
            ebr_id = ebr_row[0]
            cursor.execute("SELECT k_point, irrep_label, multiplicity FROM irreps WHERE ebr_id = ? ORDER BY k_point", (ebr_id,))
            irreps = cursor.fetchall()
            branches = self.get_ebr_decomposition_branches(ebr_id)
            result['ebrs'].append({'ebr_data': ebr_row, 'irreps': irreps, 'branches': branches})
        return result


def interactive_mode():
    db = MagneticEBRDatabaseManager()
    print("=== Elementary Band Representation Database Manager ===")
    print("Interactive Mode\n")

    while True:
        status = db.get_database_status()
        next_sg = db.get_next_space_group()
        
        print(f"\n--- Database Status ---")
        print(f"  Space Groups: {status['space_groups']}/230")
        print(f"  Total EBRs: {status['ebrs']}")
        print(f"  Total Irreps: {status['irreps']}")
        print(f"  Total Decomposition Branches: {status['branches']}")
        if status['space_groups'] > 0: print(f"  Range: SG {status['min_sg']} - {status['max_sg']}")
        
        if next_sg is None and status['space_groups'] == 230: print("\n🎉 All 230 SGs processed!")
        
        print(f"\nNext auto-SG: {next_sg if next_sg else 'All processed or start fresh'}")
        # Only show missing SGs if not all are completed
        if status['space_groups'] < 230 and status['missing_sgs']:
            print(f"Missing SGs: {len(status['missing_sgs'])}")
        elif status['space_groups'] == 230:
            print("All SGs accounted for.")

        print("\n--- Options ---")
        print("1. Input raw data for next space group (prompts for branches)")
        print("2. Input raw data for specific space group (prompts for branches)")
        print("3. Input from file for specific space group (prompts for branches)")
        print("4. Manage decomposition branches for an EBR (manual access)") 
        print("5. Show detailed database status")
        print("6. Query space group data")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()

        try:
            if choice == "1":
                if next_sg is None: 
                    print("ℹ️ No 'next' SG to process automatically. Use option 2 for a specific SG or check status.")
                    continue
                sg_to_process = next_sg
                print(f"\nEntering data for SG {sg_to_process}. Paste raw data (Ctrl+D on Unix/macOS, Ctrl+Z then Enter on Windows, or 2 empty lines to end):")
                print("-" * 80)
                raw_lines = []
                empty_line_count = 0
                while True:
                    try: 
                        line = input()
                        raw_lines.append(line)
                    except EOFError: # End of input
                        break 
                    if not line.strip(): # Check for empty line
                        empty_line_count += 1
                    else:
                        empty_line_count = 0 # Reset if line has content
                    if empty_line_count >= 2: # Two consecutive empty lines signal end
                        # Remove the last two empty lines if that's the termination method
                        raw_lines = raw_lines[:-2]
                        break
                raw_data = "\n".join(raw_lines)
                if raw_data.strip():
                    num_ebrs = db.ingest_text(sg_to_process, raw_data.strip(), prompt_for_branches_immediately=True)
                    print(f"✅ Imported {num_ebrs} EBRs for SG {sg_to_process}.")
                else: 
                    print("❌ No data entered.")

            elif choice == "2":
                sg_num_str = input("Enter SG number (1-230): ")
                if not sg_num_str.isdigit(): print("❌ Invalid SG number."); continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230): print("❌ SG out of range."); continue
                print(f"\nEntering data for SG {sg_num}. Paste raw data (Ctrl+D on Unix/macOS, Ctrl+Z then Enter on Windows, or 2 empty lines to end):")
                print("-" * 80)
                raw_lines = []
                empty_line_count = 0
                while True:
                    try: 
                        line = input()
                        raw_lines.append(line)
                    except EOFError: break
                    if not line.strip(): empty_line_count +=1
                    else: empty_line_count = 0
                    if empty_line_count >=2: raw_lines = raw_lines[:-2]; break
                raw_data = "\n".join(raw_lines)
                if raw_data.strip():
                    num_ebrs = db.ingest_text(sg_num, raw_data.strip(), prompt_for_branches_immediately=True)
                    print(f"✅ Imported {num_ebrs} EBRs for SG {sg_num}.")
                else: print("❌ No data entered.")

            elif choice == "3":
                sg_num_str = input("Enter SG number (1-230): ")
                if not sg_num_str.isdigit(): print("❌ Invalid SG number."); continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230): print("❌ SG out of range."); continue
                filepath = input(f"File path for SG {sg_num}: ").strip()
                num_ebrs = db.ingest_file(sg_num, filepath, prompt_for_branches_immediately=True)
                print(f"✅ Imported {num_ebrs} EBRs for SG {sg_num} from '{filepath}'.")
            
            elif choice == "4": 
                sg_num_str = input("Enter SG number (1-230) to manage branches: ")
                if not sg_num_str.isdigit(): print("❌ Invalid SG number."); continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230): print("❌ SG out of range."); continue

                decomposable_ebrs = db.get_decomposable_ebrs(sg_num)
                if not decomposable_ebrs:
                    print(f"ℹ️ No decomposable EBRs found for SG {sg_num}.")
                    continue
                
                print(f"\n--- Decomposable EBRs for SG {sg_num} ---")
                ebr_map = {}
                for i, ebr_data_tuple in enumerate(decomposable_ebrs):
                    display_idx_str = str(i + 1)
                    ebr_map[display_idx_str] = ebr_data_tuple 
                    
                    # MODIFIED DISPLAY LOGIC
                    # ebr_data_tuple is (id, wl, ss, orbital_label, orbital_multiplicity, notes)
                    label = ebr_data_tuple[3]
                    mult = ebr_data_tuple[4]
                    full_orbital_str = f"{label}({mult})" if mult > 1 else label
                    
                    print(f"  {display_idx_str}. EBR ID: {ebr_data_tuple[0]}, Wyckoff: {ebr_data_tuple[1]}{ebr_data_tuple[2]}, Orbital: {full_orbital_str}")
                
                ebr_choice_idx_str = input("Select EBR by number to manage branches: ").strip()
                selected_ebr_tuple = ebr_map.get(ebr_choice_idx_str)

                if not selected_ebr_tuple: 
                    print("❌ Invalid selection.")
                    continue
                
                selected_ebr_id = selected_ebr_tuple[0]
                wyck_info = f"{selected_ebr_tuple[1]}{selected_ebr_tuple[2]}"
                
                # Reconstruct orb_info for the interactive prompt
                orb_label = selected_ebr_tuple[3]
                orb_mult = selected_ebr_tuple[4]
                orb_info = f"{orb_label}({orb_mult})" if orb_mult > 1 else orb_label

                db._manage_branches_interactively_for_ebr(selected_ebr_id, wyck_info, orb_info, called_from_ingest=False)

            elif choice == "5":
                status = db.get_database_status() # Refresh status
                print(f"\nDetailed Status:")
                print(f"  SGs Processed: {status['space_groups']}/230")
                print(f"  Min/Max SG: {status['min_sg']}/{status['max_sg']}")
                print(f"  Total EBRs: {status['ebrs']}, Irreps: {status['irreps']}, Branches: {status['branches']}")
                if status['space_groups'] < 230 and status['missing_sgs']:
                    mc = len(status['missing_sgs'])
                    if mc <= 20:
                         print(f"  Missing SGs: {', '.join(status['missing_sgs'])}")
                    else:
                         print(f"  Missing SGs: {mc} total. First 10: {', '.join(status['missing_sgs'][:10])}")
                elif status['space_groups'] == 230 :
                    print("  ✅ All 230 space groups are present in the database.")

            elif choice == "6":
                sg_num_str = input("Enter SG number to query: ")
                if not sg_num_str.isdigit(): print("❌ Invalid SG number."); continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230): print("❌ SG out of range."); continue
                result = db.query_space_group(sg_num)
                if result:
                    sg_info = result['space_group']
                    print(f"\n--- Query Result for SG {sg_num} (DB ID: {sg_info[0]}, Symbol: {sg_info[2] if sg_info[2] else 'N/A'}) ---")
                    print(f"  Created: {sg_info[3]}, Updated: {sg_info[4]}")
                    print(f"  EBRs Found: {len(result['ebrs'])}")
                    for i, ebr_entry in enumerate(result['ebrs'], 1):
                        ebr_d = ebr_entry['ebr_data'] # (id, wl, ss, ol, notes, cr_at, up_at)
                        print(f"\n  EBR {i}: ID={ebr_d[0]}, Orbital='{ebr_d[3]}' at Wyckoff='{ebr_d[1]}{ebr_d[2]}'")
                        print(f"    Notes: '{ebr_d[4]}', Created: {ebr_d[5]}, Updated: {ebr_d[6]}")
                        if ebr_entry['irreps']:
                            print(f"    Irreps ({len(ebr_entry['irreps'])}):")
                            for kpt, lbl, mult in ebr_entry['irreps']:
                                print(f"      {kpt}: {lbl}{f'({mult})' if mult is not None else ''}")
                        if ebr_entry['branches']:
                            print(f"    Decomposition Branches ({len(ebr_entry['branches'])}):")
                            for dec_idx, b1, b2, ca_br in ebr_entry['branches']: # (dec_idx, b1, b2, created_at)
                                print(f"      Index {dec_idx}: B1='{b1}', B2='{b2}' (Added: {ca_br})")
                else: print(f"ℹ️ No data for SG {sg_num}")
            elif choice == "7": break
            else: print("❌ Invalid choice (1-7).")
        
        except ValueError as e: print(f"❌ Value Error: {e}")
        except sqlite3.Error as e: print(f"❌ Database Error: {e}")
        except Exception as e: print(f"❌ Unexpected Error: {e}")
        print("-" * 30) 
    db.close()
    print("Database connection closed.")

def main():
    parser = argparse.ArgumentParser(
        description="EBR Database Manager with Integrated Branch Input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --interactive
  %(prog)s --sg 12 data_sg12.txt  (branches managed via interactive option 4 later)
  %(prog)s --status
  %(prog)s --query --sg 12"""
    )
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--sg", type=int, help="Space group number (1-230)")
    parser.add_argument("--status", action="store_true", help="Show database status")
    parser.add_argument("--query", action="store_true", help="Query SG data (use with --sg)")
    parser.add_argument("input_file", nargs="?", help="Input file for data import (with --sg)")
    args = parser.parse_args()

    db_manager_cli = None
    try:
        if args.interactive: interactive_mode()
        elif args.status:
            db_manager_cli = MagneticEBRDatabaseManager()
            status = db_manager_cli.get_database_status()
            print("=== Database Status ===")
            pc = (status['space_groups']/230*100) if status['space_groups'] > 0 else 0
            print(f"Space Groups: {status['space_groups']}/230 ({pc:.1f}%)")
            print(f"Total EBRs: {status['ebrs']}, Irreps: {status['irreps']}, Branches: {status['branches']}")
            if status['space_groups'] == 230: print("✅ All 230 SGs completed!")
            elif status['missing_sgs']: print(f"Missing SGs: {len(status['missing_sgs'])}")
        elif args.query and args.sg:
            if not (1 <= args.sg <= 230): print(f"❌ Query SG {args.sg} out of range.", file=sys.stderr); sys.exit(1)
            db_manager_cli = MagneticEBRDatabaseManager()
            res = db_manager_cli.query_space_group(args.sg)
            if res: print(f"Data found for SG {args.sg}. EBRs: {len(res['ebrs'])}. Use interactive query for full details.")
            else: print(f"No data for SG {args.sg}.")
        elif args.sg and args.input_file:
            if not (1 <= args.sg <= 230): print(f"❌ Import SG {args.sg} out of range.", file=sys.stderr); sys.exit(1)
            db_manager_cli = MagneticEBRDatabaseManager()
            num = db_manager_cli.ingest_file(args.sg, args.input_file, prompt_for_branches_immediately=False) 
            print(f"✅ Imported {num} EBRs for SG {args.sg} from '{args.input_file}'. Manage branches via interactive mode if needed.")
        else: parser.print_help()
    except sqlite3.Error as e: print(f"❌ CLI Database Error: {e}", file=sys.stderr); sys.exit(1)
    except ValueError as e: print(f"❌ CLI Value Error: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"❌ CLI Unexpected Error: {e}", file=sys.stderr); sys.exit(1)
    finally:
        if db_manager_cli: db_manager_cli.close()

if __name__ == "__main__":
    main()