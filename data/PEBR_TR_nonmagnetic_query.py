# import sqlite3
# import argparse
# import re
# import sys
# import os
# from pathlib import Path

# # Helper functions
# # def parse_kpoint_cells(irreps_txt):
# #     """
# #     Given a string like "A1+(1)⊕A2+(1)   A3A4(2)",
# #     return ["A1+(1)","A2+(1)"] as a two-tuple for the first token, and "A3A4(2)" for the next.
# #     """
# #     tokens = [tok.strip() for tok in irreps_txt.split() if tok.strip()]
# #     processed = []
# #     for tok in tokens:
# #         if '⊕' in tok:
# #             left, right = [p.strip() for p in tok.split('⊕', 1)]
# #             processed.append((left, right))
# #         else:
# #             processed.append(tok)
# #     return processed

# # physical elementary band representation w/ time reversal from bilbao to simple sql db
# # Usage:
#     # python ebr_database_manager.py --interactive
#     # python ebr_database_manager.py --sg <space_group_number> <input_file>
#     # python ebr_database_manager.py --status
#     # python ebr_database_manager.py --query --sg <space_group_number>

# '''
# CREATE TABLE space_groups (
#   id         INTEGER PRIMARY KEY AUTOINCREMENT,
#   number     INTEGER UNIQUE NOT NULL,
#   symbol     TEXT,
#   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# CREATE TABLE ebrs (
#   id             INTEGER PRIMARY KEY AUTOINCREMENT,
#   space_group_id INTEGER NOT NULL REFERENCES space_groups(id),
#   wyckoff_letter TEXT    NOT NULL,
#   site_symmetry  TEXT    NOT NULL,
#   orbital_label  TEXT    NOT NULL,
#   time_reversal  BOOLEAN NOT NULL DEFAULT 1,
#   single_index   TEXT,
#   double_index   TEXT,
#   notes          TEXT,
#   branch         INTEGER NOT NULL DEFAULT 0,
#   topo_single    INTEGER,  -- (optional) store Z₂ or Z₄ index for single‐valued
#   topo_double    INTEGER,  -- (optional) store Z₂ or Z₄ index for double‐valued
#   branch1_irreps TEXT,     -- for decomposable EBRs
#   branch2_irreps TEXT,     -- for decomposable EBRs
#   created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# CREATE TABLE irreps (
#   id           INTEGER PRIMARY KEY AUTOINCREMENT,
#   ebr_id       INTEGER NOT NULL REFERENCES ebrs(id),
#   k_point      TEXT    NOT NULL,
#   irrep_label  TEXT    NOT NULL,
#   multiplicity INTEGER,
#   created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
# '''

# import sqlite3
# import argparse
# import re
# import sys
# import os
# from pathlib import Path

# # Helper functions
# def parse_kpoint_cells(irreps_txt):
#     """
#     Given a substring like "C1+(1) ⊕ C2+(1)   C3C4(2)",
#     return ["C1+(1) ⊕ C2+(1)", "C3C4(2)"].

#     In other words, treat each whitespace-separated token as a single string
#     (even if it contains "⊕").
#     """
#     return [tok.strip() for tok in irreps_txt.split() if tok.strip()]

# def split_decomposable_column(j, kpoint_data):
#         # decomposable columns in bilbao bandrep nonmagnetic tqc are buttons leading to tables of branch 1 branch 2 data
#         branch1 = []
#         branch2 = []
#         for (kp_label, reps) in kpoint_data:
#             cell = reps[j]
#             if isinstance(cell, tuple):
#                 branch1.append(cell[0].strip())
#                 branch2.append(cell[1].strip())
#             else:
#                 s = cell.strip()
#                 branch1.append(s)
#                 branch2.append(s)
#         return branch1, branch2


# class EBRDatabaseManager:
#     def __init__(self, db_path="pebr_tr_nonmagnetic.db"):
#         self.db_path = db_path
#         self.conn = None
#         self.connect()
#         self.create_tables()
    
#     def connect(self):
#         """Connect to SQLite database."""
#         self.conn = sqlite3.connect(self.db_path)
#         self.conn.execute("PRAGMA foreign_keys = ON")
    
#     def close(self):
#         """Close database connection."""
#         if self.conn:
#             self.conn.close()
    
#     def create_tables(self):
#         """Create database tables if they don't exist."""
#         cursor = self.conn.cursor()
        
#         # Enhanced space_groups table
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS space_groups (
#           id            INTEGER PRIMARY KEY AUTOINCREMENT,
#           number        INTEGER UNIQUE NOT NULL,
#           symbol        TEXT,
#           created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#           updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
        
#         # Enhanced ebrs table
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ebrs (
#           id             INTEGER PRIMARY KEY AUTOINCREMENT,
#           space_group_id INTEGER NOT NULL REFERENCES space_groups(id),
#           wyckoff_letter TEXT    NOT NULL,
#           site_symmetry  TEXT    NOT NULL,
#           orbital_label  TEXT    NOT NULL,
#           time_reversal  BOOLEAN NOT NULL DEFAULT 1,
#           single_index   TEXT,
#           double_index   TEXT,
#           notes          TEXT,
#           branch1_irreps TEXT,
#           branch2_irreps TEXT,
#           created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
        
#         # Enhanced irreps table
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS irreps (
#           id           INTEGER PRIMARY KEY AUTOINCREMENT,
#           ebr_id       INTEGER NOT NULL REFERENCES ebrs(id),
#           k_point      TEXT    NOT NULL,
#           irrep_label  TEXT    NOT NULL,
#           multiplicity INTEGER,
#           created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
        
#         # Create indexes for better performance
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_sg_number ON space_groups(number);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_sg ON ebrs(space_group_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_ebr ON irreps(ebr_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_kpoint ON irreps(k_point);")
        
#         self.conn.commit()
    
#     def get_next_space_group(self):
#         """Get the next space group number to process (1-230)."""
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT MAX(number) FROM space_groups WHERE number <= 230")
#         result = cursor.fetchone()
        
#         if result[0] is None:
#             return 1  # Start with space group 1
#         elif result[0] >= 230:
#             return None  # All space groups completed
#         else:
#             return result[0] + 1
    
#     def get_database_status(self):
#         """Get current database status."""
#         cursor = self.conn.cursor()
        
#         # Count space groups
#         cursor.execute("SELECT COUNT(*) FROM space_groups WHERE number <= 230")
#         sg_count = cursor.fetchone()[0]
        
#         # Count EBRs
#         cursor.execute("SELECT COUNT(*) FROM ebrs")
#         ebr_count = cursor.fetchone()[0]
        
#         # Count irreps
#         cursor.execute("SELECT COUNT(*) FROM irreps")
#         irrep_count = cursor.fetchone()[0]
        
#         # Get range of space groups
#         cursor.execute("SELECT MIN(number), MAX(number) FROM space_groups WHERE number <= 230")
#         min_sg, max_sg = cursor.fetchone()
        
#         # Get missing space groups
#         cursor.execute("""
#             WITH RECURSIVE numbers(n) AS (
#                 SELECT 1
#                 UNION ALL
#                 SELECT n+1 FROM numbers WHERE n < 230
#             )
#             SELECT GROUP_CONCAT(n) FROM numbers 
#             WHERE n NOT IN (SELECT number FROM space_groups WHERE number <= 230)
#         """)
#         missing_sgs = cursor.fetchone()[0]
        
#         return {
#             'space_groups': sg_count,
#             'ebrs': ebr_count,
#             'irreps': irrep_count,
#             'min_sg': min_sg or 0,
#             'max_sg': max_sg or 0,
#             'missing_sgs': missing_sgs.split(',') if missing_sgs else []
#         }
    
#     def parse_wyckoff(self, entry):
#         """Parse Wyckoff position entry like '1a(1)' -> ('1a', '1')."""
#         m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
#         if m:
#             return m.group(1), m.group(2)
#         else:
#             return entry, ""
    
#     def parse_orbital(self, entry):
#         """Parse orbital entry like 'A↑G(1)' -> ('A↑G', 1)."""
#         m = re.match(r"^(.+)\((\d+)\)$", entry)
#         if m:
#             return m.group(1), int(m.group(2))
#         return entry, 1
    
#     def parse_multiplicity(self, entry):
#         """Parse multiplicity from entry like 'R2R2(2)' -> 2."""
#         m = re.search(r"\((\d+)\)$", entry)
#         if m:
#             return int(m.group(1))
#         return None
    
#     def parse_irrep_label(self, entry):
#         """Parse irrep label by removing the multiplicity part."""
#         m = re.match(r"^(.+)\(\d+\)$", entry)
#         return m.group(1) if m else entry
    
#     def validate_input_file(self, filepath):
#         """Validate that input file exists and has expected format."""
#         if not os.path.exists(filepath):
#             return False, f"File '{filepath}' not found."
        
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 content = f.read()
                
#             # Check for required sections
#             required_sections = ["Wyckoff pos.", "Band-Rep.", "Decomposable"]
#             missing = [section for section in required_sections if section not in content]
            
#             if missing:
#                 return False, f"Missing required sections: {', '.join(missing)}"
            
#             return True, "File format appears valid."
            
#         except Exception as e:
#             return False, f"Error reading file: {str(e)}"
    
#     def ingest_file(self, sg_number, filepath):
#         """Ingest data from file for specified space group."""
#         # Validate inputs
#         if not (1 <= sg_number <= 230):
#             raise ValueError(f"Space group number must be between 1 and 230, got {sg_number}")
        
#         is_valid, message = self.validate_input_file(filepath)
#         if not is_valid:
#             raise ValueError(message)
        
#         # Read and parse file
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 raw_lines = [line.rstrip("\n") for line in f if line.strip()]
#         except Exception as e:
#             raise ValueError(f"Error reading file: {str(e)}")
        
#         # Parse file content
#         wyckoff_line, orbital_line, notes_line, kpoint_lines = self._parse_file_content(raw_lines)
        
#         # Process data
#         wyckoff_entries = wyckoff_line.split()
#         orbital_entries = orbital_line.split()
#         notes_entries   = notes_line.split()

#         num_cols = len(wyckoff_entries)

#         # If notes_entries has more tokens than columns, keep only the last `num_cols` tokens
#         if len(notes_entries) > num_cols:
#             notes_entries = notes_entries[-num_cols:]
#         elif len(notes_entries) < num_cols:
#             raise ValueError(
#                 f"Column count mismatch: Wyckoff({num_cols}), "
#                 f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
#             )

#         # Now we can assert consistency
#         if not (len(orbital_entries) == num_cols == len(notes_entries)):
#             raise ValueError(
#                 f"Column count mismatch after trimming: "
#                 f"Wyckoff({len(wyckoff_entries)}), "
#                 f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
#             )
        
#         # Parse k-point data
#         kpoint_data = []
#         for kp_line in kpoint_lines:
#             tokens = kp_line.split()
#             if tokens:
#                 kp_label = tokens[0].rstrip(":")
#                 reps = tokens[1:]
#                 kpoint_data.append((kp_label, reps))
        
#         # Insert into database
#         self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data)
        
#         return num_cols
    
#     def ingest_text(self, sg_number, raw_text):
#         """Ingest data from raw text string for specified space group."""
#         # Validate inputs
#         if not (1 <= sg_number <= 230):
#             raise ValueError(f"Space group number must be between 1 and 230, got {sg_number}")
        
#         if not raw_text or not raw_text.strip():
#             raise ValueError("Raw text cannot be empty")
        
#         # Parse raw text data
#         try:
#             wyckoff_line, orbital_line, notes_line, kpoint_data = self._parse_raw_text(raw_text)
#         except Exception as e:
#             raise ValueError(f"Error parsing raw text: {str(e)}")
        
#         wyckoff_entries = wyckoff_line.split()
#         orbital_entries = orbital_line.split()
#         notes_entries = notes_line.split()
        
#         # Validate column consistency
#         num_cols = len(wyckoff_entries)
        
#         # Apply the same fix for trimming excess notes entries
#         if len(notes_entries) > num_cols:
#             notes_entries = notes_entries[-num_cols:]
#         elif len(notes_entries) < num_cols:
#             raise ValueError(
#                 f"Column count mismatch: Wyckoff({len(wyckoff_entries)}), "
#                 f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
#             )
        
#         if not (len(orbital_entries) == num_cols == len(notes_entries)):
#             raise ValueError(
#                 f"Column count mismatch after trimming: "
#                 f"Wyckoff({len(wyckoff_entries)}), "
#                 f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
#             )
        
#         return self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data)
    
#     def _parse_file_content(self, raw_lines):
#         """
#         Parse file content (from a .txt) to extract Wyckoff, Orbital, notes, and k-points.
#         First, collapse ANY trailing backslash+newline into a single space.
#         """
#         # 1) Re-join into one big string and collapse backslash+newline → space
#         joined = "\n".join(raw_lines)
#         joined = re.sub(r'\\\s*\n\s*', ' ', joined)

#         # 2) Re-split into clean lines (dropping any blank lines)
#         lines = [line.rstrip("\n") for line in joined.splitlines() if line.strip()]

#         # 3) Now apply your existing "find Wyckoff / Band-Rep / Decomposable" logic
#         wyckoff_line = None
#         orbital_line  = None
#         notes_line    = None
#         kpoint_lines  = []

#         i = 0
#         while i < len(lines):
#             line = lines[i]

#             if line.startswith("Wyckoff pos."):
#                 i += 1
#                 # skip any blank lines
#                 while i < len(lines) and not lines[i].strip():
#                     i += 1
#                 if i < len(lines):
#                     wyckoff_line = lines[i]

#             elif line.startswith("Band-Rep."):
#                 i += 1
#                 # skip any blank lines
#                 while i < len(lines) and not lines[i].strip():
#                     i += 1
#                 if i < len(lines):
#                     orbital_line = lines[i]

#             elif line.startswith("Decomposable"):
#                 # skip the "Decomposable" line itself
#                 i += 1
#                 # skip the "Indecomposable" header line
#                 while i < len(lines) and not lines[i].strip():
#                     i += 1
#                 if i < len(lines):
#                     i += 1  # move past the header line

#                 # next non-blank is actual notes
#                 while i < len(lines) and not lines[i].strip():
#                     i += 1
#                 if i < len(lines):
#                     notes_line = lines[i]

#                 # everything after that is k-point data
#                 i += 1
#                 while i < len(lines):
#                     if lines[i].strip():
#                         kpoint_lines.append(lines[i])
#                     i += 1
#                 break

#             i += 1

#         if not all([wyckoff_line, orbital_line, notes_line]):
#             raise ValueError("Could not locate all required sections in file")

#         return wyckoff_line, orbital_line, notes_line, kpoint_lines

#     def _parse_raw_text(self, raw_text):
#         """
#         Parse raw pasted text to extract:
#           • wyckoff_line  (string)
#           • orbital_line  (string)
#           • notes_line    (string)
#           • kpoint_data   (list of (kpt_label, cells_list)),
#                            where each cells_list[j] is a single string
#                            (e.g. "C1+(1) ⊕ C2+(1)").
#         """
#         # 1) Collapse any "\" + newline into a single space:
#         text = re.sub(r'\\\s*\n\s*', ' ', raw_text.strip())

#         # 2) Extract Wyckoff: between "Wyckoff pos." and "Band-Rep."
#         wyckoff_match = re.search(
#             r'Wyckoff pos\.\s*(.+?)\s*Band-Rep\.', text,
#             re.IGNORECASE | re.DOTALL
#         )
#         # 3) Extract Orbital: between "Band-Rep." and "Decomposable"
#         orbital_match = re.search(
#             r'Band-Rep\.\s*(.+?)\s*Decomposable', text,
#             re.IGNORECASE | re.DOTALL
#         )
#         if not wyckoff_match or not orbital_match:
#             raise ValueError("Could not find required sections (Wyckoff pos. or Band-Rep.)")

#         wyckoff_line = re.sub(r'\s+', ' ', wyckoff_match.group(1).strip())
#         orbital_line = re.sub(r'\s+', ' ', orbital_match.group(1).strip())

#         # 4) Extract everything after "Decomposable" (allow optional backslash)
#         decomp_pattern = r'Decomposable\\?\s*(.*?)(?=[A-ZΓ]+:\([^)]+\))'
#         decomp_match = re.search(decomp_pattern, text, re.IGNORECASE | re.DOTALL)
#         if not decomp_match:
#             raise ValueError("Could not find Decomposable section")

#         decomp_section = decomp_match.group(1).strip()
#         lines = [line.strip() for line in decomp_section.split('\n') if line.strip()]

#         expected_cols = len(wyckoff_line.split())
#         notes_line = None
#         for ln in lines:
#             toks = ln.split()
#             if len(toks) == expected_cols:
#                 notes_line = ln
#                 break
#         if not notes_line:
#             # fallback: pick the last "data" line
#             data_lines = [ln for ln in lines
#                           if not ln.lower().startswith('indecomposable') or len(ln.split()) > 1]
#             if data_lines:
#                 notes_line = data_lines[-1]
#             else:
#                 notes_line = lines[-1] if lines else ""
#         if not notes_line:
#             raise ValueError("Could not find notes data line")

#         # 5) Extract k-point block
#         kpoint_start_match = re.search(
#             r'([A-ZΓ]+:\([^)]+\).*)', text,
#             re.IGNORECASE | re.DOTALL
#         )
#         if not kpoint_start_match:
#             raise ValueError("Could not find k-point data")

#         kpoint_text = kpoint_start_match.group(1)
#         kpoint_pattern = r'([A-ZΓ]+):\(([^)]+)\)'
#         matches = list(re.finditer(kpoint_pattern, kpoint_text))

#         kpoint_data = []
#         for idx, m in enumerate(matches):
#             kpt_label = m.group(1)
#             # (coords = m.group(2) not strictly needed here)

#             start_pos = m.end()
#             if idx + 1 < len(matches):
#                 end_pos = matches[idx + 1].start()
#                 irreps_txt = kpoint_text[start_pos:end_pos]
#             else:
#                 irreps_txt = kpoint_text[start_pos:]

#             # Each whitespace-separated token (e.g. "C1+(1) ⊕ C2+(1)")
#             cells = parse_kpoint_cells(irreps_txt)
#             kpoint_data.append((kpt_label, cells))

#         return wyckoff_line, orbital_line, notes_line, kpoint_data
    
#     def _insert_data(self, sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data):
#         """
#         For each column j (0 <= j < len(wyckoff_entries)):
#          - If column j is not decomposable: insert one EBR + its irreps (as before).
#          - If column j is decomposable: insert one EBR but don't populate irreps.
#         """
#         cursor = self.conn.cursor()

#         # 1) Upsert space_group
#         cursor.execute("""
#             INSERT OR REPLACE INTO space_groups(number, updated_at)
#             VALUES (?, CURRENT_TIMESTAMP)
#         """, (sg_number,))
#         self.conn.commit()
#         cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
#         sg_id = cursor.fetchone()[0]

#         # 2) Delete old EBRs (and their irreps) for this SG
#         cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
#         self.conn.commit()

#         num_cols = len(wyckoff_entries)
#         for j in range(num_cols):
#             # a) Parse wyckoff_entries[j]
#             wyck_raw = wyckoff_entries[j]
#             m = re.match(r"^(\d+\w)(\([^)]+\))$", wyck_raw)
#             if m:
#                 wyckoff_letter = m.group(1)
#                 site_symmetry  = m.group(2)
#             else:
#                 wyckoff_letter = wyck_raw
#                 site_symmetry  = ""

#             # b) Parse orbital_entries[j]
#             orb_raw = orbital_entries[j]
#             orb_label, orb_mult = self.parse_orbital(orb_raw)
#             if orb_mult > 1:
#                 single_val = None
#                 double_val = orb_label
#             else:
#                 single_val = orb_label
#                 double_val = None

#             note_j = notes_entries[j].lower()  # should be either "decomposable" or "indecomposable"

#             if note_j == "indecomposable":
#                 # Insert exactly one EBR + its irreps (loop over kpoint_data[j] strings)
#                 cursor.execute("""
#                     INSERT INTO ebrs (
#                       space_group_id,
#                       wyckoff_letter,
#                       site_symmetry,
#                       orbital_label,
#                       time_reversal,
#                       single_index,
#                       double_index,
#                       notes,
#                       branch1_irreps,
#                       branch2_irreps
#                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     sg_id,
#                     wyckoff_letter,
#                     site_symmetry,
#                     orb_label,
#                     1,
#                     single_val,
#                     double_val,
#                     note_j,          # typically "indecomposable"
#                     None,
#                     None
#                 ))
#                 self.conn.commit()
#                 ebr_id = cursor.lastrowid

#                 # Write all irreps exactly as strings, including any "⊕"
#                 for (kp_label, cells) in kpoint_data:
#                     if j < len(cells):  # Safety check
#                         full_str = cells[j]    # e.g. "C1+(1) ⊕ C2+(1)"
#                         mult = self.parse_multiplicity(full_str)
#                         label = self.parse_irrep_label(full_str)
#                         cursor.execute("""
#                             INSERT INTO irreps (
#                               ebr_id,
#                               k_point,
#                               irrep_label,
#                               multiplicity
#                             ) VALUES (?, ?, ?, ?)
#                         """, (ebr_id, kp_label, label, mult))
#                 self.conn.commit()

#             else:  # note_j == "decomposable"
#                 # Insert one EBR row, but set branch1_irreps/branch2_irreps = NULL
#                 cursor.execute("""
#                     INSERT INTO ebrs (
#                       space_group_id,
#                       wyckoff_letter,
#                       site_symmetry,
#                       orbital_label,
#                       time_reversal,
#                       single_index,
#                       double_index,
#                       notes,
#                       branch1_irreps,
#                       branch2_irreps
#                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     sg_id,
#                     wyckoff_letter,
#                     site_symmetry,
#                     orb_label,
#                     1,
#                     single_val,
#                     double_val,
#                     note_j,      # typically "decomposable"
#                     None,
#                     None
#                 ))
#                 self.conn.commit()
#                 ebr_id = cursor.lastrowid

#                 # For decomposable entries, still store the combined irrep representation
#                 for (kp_label, cells) in kpoint_data:
#                     if j < len(cells):  # Safety check
#                         full_str = cells[j]  # still a single string, e.g. "C1+(1) ⊕ C2+(1)"
#                         mult = self.parse_multiplicity(full_str)
#                         label = self.parse_irrep_label(full_str)
#                         cursor.execute("""
#                             INSERT INTO irreps (
#                               ebr_id,
#                               k_point,
#                               irrep_label,
#                               multiplicity
#                             ) VALUES (?, ?, ?, ?)
#                         """, (ebr_id, kp_label, label, mult))
#                 self.conn.commit()
        
#         return num_cols
    
#     def query_space_group(self, sg_number):
#         """Query data for a specific space group."""
#         cursor = self.conn.cursor()
        
#         # Get space group info
#         cursor.execute("SELECT * FROM space_groups WHERE number = ?", (sg_number,))
#         sg_data = cursor.fetchone()
        
#         if not sg_data:
#             return None
        
#         # Get EBRs
#         cursor.execute("""
#             SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes
#             FROM ebrs WHERE space_group_id = ?
#         """, (sg_data[0],))
#         ebrs = cursor.fetchall()
        
#         # Get irreps for each EBR
#         result = {
#             'space_group': sg_data,
#             'ebrs': []
#         }
        
#         for ebr in ebrs:
#             cursor.execute("""
#                 SELECT k_point, irrep_label, multiplicity
#                 FROM irreps WHERE ebr_id = ?
#                 ORDER BY k_point
#             """, (ebr[0],))
#             irreps = cursor.fetchall()
            
#             result['ebrs'].append({
#                 'ebr_data': ebr,
#                 'irreps': irreps
#             })
        
#         return result

# def interactive_mode():
#     """Run interactive mode for sequential data input."""
#     db = EBRDatabaseManager()
    
#     print("=== Elementary Band Representation Database Manager ===")
#     print("Interactive Mode - Sequential Input from Space Group 1 to 230\n")
    
#     while True:
#         # Get current status
#         status = db.get_database_status()
#         next_sg = db.get_next_space_group()
        
#         print(f"Database Status:")
#         print(f"  Space Groups: {status['space_groups']}/230")
#         print(f"  Total EBRs: {status['ebrs']}")
#         print(f"  Total Irreps: {status['irreps']}")
        
#         if status['space_groups'] > 0:
#             print(f"  Range: SG {status['min_sg']} - {status['max_sg']}")
        
#         if next_sg is None:
#             print("\n🎉 All 230 space groups have been processed!")
#             break
        
#         print(f"\nNext space group to process: {next_sg}")
#         print(f"Missing space groups: {len(status['missing_sgs'])}")
        
#         print("\nOptions:")
#         print("1. Input raw data for next space group")
#         print("2. Input raw data for specific space group")
#         print("3. Input from file for specific space group")
#         print("4. Show database status")
#         print("5. Query space group data")
#         print("6. Exit")
        
#         choice = input("\nEnter choice (1-6): ").strip()
        
#         if choice == "1":
#             # Input raw data for next space group
#             print(f"\nEntering data for space group {next_sg}")
#             print("Paste your raw data below (press Enter twice when done):")
#             print("Example format: 'Wyckoff pos.1a(1)1a(1)Band-Rep.A↑G(1)AA↑G(2)Decomposable\\...'")
#             print("-" * 80)
            
#             raw_data = ""
#             empty_lines = 0
#             while True:
#                 try:
#                     line = input()
#                     if not line.strip():
#                         empty_lines += 1
#                         if empty_lines >= 2:
#                             break
#                     else:
#                         empty_lines = 0
#                         raw_data += line + " "
#                 except EOFError:
#                     break
            
#             if raw_data.strip():
#                 try:
#                     num_ebrs = db.ingest_text(next_sg, raw_data.strip())
#                     print(f"✅ Successfully imported {num_ebrs} EBRs for space group {next_sg}")
#                 except Exception as e:
#                     print(f"❌ Error: {str(e)}")
#                     print("Please check your data format and try again.")
#             else:
#                 print("❌ No data entered.")
        
#         elif choice == "2":
#             # Input raw data for specific space group
#             try:
#                 sg_num = int(input("Enter space group number (1-230): "))
#                 print(f"\nEntering data for space group {sg_num}")
#                 print("Paste your raw data below (press Enter twice when done):")
#                 print("-" * 80)
                
#                 raw_data = ""
#                 empty_lines = 0
#                 while True:
#                     try:
#                         line = input()
#                         if not line.strip():
#                             empty_lines += 1
#                             if empty_lines >= 2:
#                                 break
#                         else:
#                             empty_lines = 0
#                             raw_data += line + " "
#                     except EOFError:
#                         break
                
#                 if raw_data.strip():
#                     num_ebrs = db.ingest_text(sg_num, raw_data.strip())
#                     print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
#                 else:
#                     print("❌ No data entered.")
                    
#             except ValueError as e:
#                 print(f"❌ Error: {str(e)}")
#             except Exception as e:
#                 print(f"❌ Error: {str(e)}")
        
#         elif choice == "3":
#             # Input from file for specific space group
#             try:
#                 sg_num = int(input("Enter space group number (1-230): "))
#                 filepath = input(f"Enter file path for space group {sg_num}: ").strip()
                
#                 num_ebrs = db.ingest_file(sg_num, filepath)
#                 print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
#             except ValueError as e:
#                 print(f"❌ Error: {str(e)}")
#             except Exception as e:
#                 print(f"❌ Error: {str(e)}")
        
#         elif choice == "4":
#             # Show detailed status
#             print(f"\nDetailed Status:")
#             if status['missing_sgs']:
#                 missing_count = len(status['missing_sgs'])
#                 if missing_count <= 20:
#                     print(f"Missing space groups: {', '.join(status['missing_sgs'])}")
#                 else:
#                     print(f"Missing space groups: {missing_count} total")
#                     print(f"First 10: {', '.join(status['missing_sgs'][:10])}")
        
#         elif choice == "5":
#             # Query space group
#             try:
#                 sg_num = int(input("Enter space group number to query: "))
#                 result = db.query_space_group(sg_num)
                
#                 if result:
#                     print(f"\nSpace Group {sg_num}:")
#                     print(f"  Created: {result['space_group'][3]}")
#                     print(f"  Updated: {result['space_group'][4]}")
#                     print(f"  Number of EBRs: {len(result['ebrs'])}")
                    
#                     for i, ebr_data in enumerate(result['ebrs'], 1):
#                         ebr = ebr_data['ebr_data']
#                         print(f"    EBR {i}: {ebr[3]} at {ebr[1]} ({ebr[4]})")
#                 else:
#                     print(f"No data found for space group {sg_num}")
#             except ValueError:
#                 print("❌ Please enter a valid number")
        
#         elif choice == "6":
#             break
        
#         else:
#             print("❌ Invalid choice")
        
#         print()  # Add spacing
    
#     db.close()
#     print("Database connection closed.")

# def main():
#     parser = argparse.ArgumentParser(
#         description="Enhanced EBR Database Manager for Bilbao crystal server data",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   %(prog)s --interactive                    # Start interactive mode
#   %(prog)s --sg 10 data_sg10.txt          # Import space group 10
#   %(prog)s --status                        # Show database status
#   %(prog)s --query --sg 10                # Query space group 10 data
#         """
#     )
    
#     parser.add_argument("--interactive", action="store_true", 
#                        help="Start interactive mode for sequential input")
#     parser.add_argument("--sg", type=int, 
#                        help="Space group number (1-230)")
#     parser.add_argument("--status", action="store_true",
#                        help="Show database status")
#     parser.add_argument("--query", action="store_true",
#                        help="Query space group data (use with --sg)")
#     parser.add_argument("input_file", nargs="?",
#                        help="Path to input file (required for data import)")
    
#     args = parser.parse_args()
    
#     if args.interactive:
#         interactive_mode()

#     elif args.status:
#         db = EBRDatabaseManager()
#         status = db.get_database_status()
        
#         print("=== Database Status ===")
#         print(f"Space Groups: {status['space_groups']}/230 ({status['space_groups']/230*100:.1f}%)")
#         print(f"Total EBRs: {status['ebrs']}")
#         print(f"Total Irreps: {status['irreps']}")
        
#         if status['space_groups'] > 0:
#             print(f"Range: SG {status['min_sg']} - {status['max_sg']}")
        
#         missing_count = len(status['missing_sgs'])
#         if missing_count > 0:
#             print(f"Missing: {missing_count} space groups")
#             if missing_count <= 10:
#                 print(f"Missing SGs: {', '.join(status['missing_sgs'])}")
#         else:
#             print("✅ All space groups completed!")
        
#         db.close()
#     elif args.query and args.sg:
#         db = EBRDatabaseManager()
#         result = db.query_space_group(args.sg)
        
#         if result:
#             print(f"=== Space Group {args.sg} ===")
#             sg_data = result['space_group']
#             print(f"Created: {sg_data[3]}")
#             print(f"Updated: {sg_data[4]}")
#             print(f"EBRs: {len(result['ebrs'])}")
            
#             for i, ebr_data in enumerate(result['ebrs'], 1):
#                 ebr = ebr_data['ebr_data']
#                 irreps = ebr_data['irreps']
#                 print(f"\nEBR {i}: {ebr[3]} at {ebr[1]}{ebr[2]} ({ebr[4]})")
#                 print(f"  K-points: {len(irreps)}")
#                 for k_point, irrep_label, mult in irreps:
#                     mult_str = f"({mult})" if mult else ""
#                     print(f"    {k_point}: {irrep_label}{mult_str}")
#         else:
#             print(f"No data found for space group {args.sg}")

#     elif args.sg and args.input_file:
#         db = EBRDatabaseManager()
#         try:
#             num_ebrs = db.ingest_file(args.sg, args.input_file)
#             print(f"✅ Successfully imported {num_ebrs} EBRs for space group {args.sg}")
#         except Exception as e:
#             print(f"❌ Error: {str(e)}", file=sys.stderr)
#             sys.exit(1)
#         finally:
#             db.close()
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()

import sqlite3
import argparse
import re
import sys
import os
from pathlib import Path

# Helper functions
# def parse_kpoint_cells(irreps_txt):
#     """
#     Given a string like "A1+(1)⊕A2+(1)   A3A4(2)",
#     return ["A1+(1)","A2+(1)"] as a two-tuple for the first token, and "A3A4(2)" for the next.
#     """
#     tokens = [tok.strip() for tok in irreps_txt.split() if tok.strip()]
#     processed = []
#     for tok in tokens:
#         if '⊕' in tok:
#             left, right = [p.strip() for p in tok.split('⊕', 1)]
#             processed.append((left, right))
#         else:
#             processed.append(tok)
#     return processed

# physical elementary band representation w/ time reversal from bilbao to simple sql db
# Usage:
    # python ebr_database_manager.py --interactive
    # python ebr_database_manager.py --sg <space_group_number> <input_file>
    # python ebr_database_manager.py --status
    # python ebr_database_manager.py --query --sg <space_group_number>

'''
CREATE TABLE space_groups (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  number     INTEGER UNIQUE NOT NULL,
  symbol     TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ebrs (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  space_group_id INTEGER NOT NULL REFERENCES space_groups(id),
  wyckoff_letter TEXT    NOT NULL,
  site_symmetry  TEXT    NOT NULL,
  orbital_label  TEXT    NOT NULL,
  time_reversal  BOOLEAN NOT NULL DEFAULT 1,
  single_index   TEXT,
  double_index   TEXT,
  notes          TEXT,
  branch         INTEGER NOT NULL DEFAULT 0,
  topo_single    INTEGER,  -- (optional) store Z₂ or Z₄ index for single‐valued
  topo_double    INTEGER,  -- (optional) store Z₂ or Z₄ index for double‐valued
  branch1_irreps TEXT,     -- for decomposable EBRs
  branch2_irreps TEXT,     -- for decomposable EBRs
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE irreps (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ebr_id       INTEGER NOT NULL REFERENCES ebrs(id),
  k_point      TEXT    NOT NULL,
  irrep_label  TEXT    NOT NULL,
  multiplicity INTEGER,
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''

# Helper functions
def parse_kpoint_cells(irreps_txt):
    """
    Given a substring like "C1+(1) ⊕ C2+(1)   C3C4(2)",
    return ["C1+(1) ⊕ C2+(1)", "C3C4(2)"].

    In other words, treat each whitespace-separated token as a single string
    (even if it contains "⊕").
    """
    return [tok.strip() for tok in irreps_txt.split() if tok.strip()]

def split_decomposable_column(j, kpoint_data):
        # decomposable columns in bilbao bandrep nonmagnetic tqc are buttons leading to tables of branch 1 branch 2 data
        branch1 = []
        branch2 = []
        for (kp_label, reps) in kpoint_data:
            cell = reps[j]
            if isinstance(cell, tuple):
                branch1.append(cell[0].strip())
                branch2.append(cell[1].strip())
            else:
                s = cell.strip()
                branch1.append(s)
                branch2.append(s)
        return branch1, branch2


class EBRDatabaseManager:
    def __init__(self, db_path="pebr_tr_nonmagnetic.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Enhanced space_groups table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS space_groups (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          number        INTEGER UNIQUE NOT NULL,
          symbol        TEXT,
          created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Enhanced ebrs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ebrs (
          id             INTEGER PRIMARY KEY AUTOINCREMENT,
          space_group_id INTEGER NOT NULL REFERENCES space_groups(id),
          wyckoff_letter TEXT    NOT NULL,
          site_symmetry  TEXT    NOT NULL,
          orbital_label  TEXT    NOT NULL,
          time_reversal  BOOLEAN NOT NULL DEFAULT 1,
          single_index   TEXT,
          double_index   TEXT,
          notes          TEXT,
          branch1_irreps TEXT,
          branch2_irreps TEXT,
          created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Enhanced irreps table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS irreps (
          id           INTEGER PRIMARY KEY AUTOINCREMENT,
          ebr_id       INTEGER NOT NULL REFERENCES ebrs(id),
          k_point      TEXT    NOT NULL,
          irrep_label  TEXT    NOT NULL,
          multiplicity INTEGER,
          created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sg_number ON space_groups(number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_sg ON ebrs(space_group_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_ebr ON irreps(ebr_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_kpoint ON irreps(k_point);")
        # cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_parent ON ebrs(parent_ebr_id);") # FIXED: Removed this line
        
        self.conn.commit()
    
    def get_next_space_group(self):
        """Get the next space group number to process (1-230)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(number) FROM space_groups WHERE number <= 230")
        result = cursor.fetchone()
        
        if result[0] is None:
            return 1  # Start with space group 1
        elif result[0] >= 230:
            return None  # All space groups completed
        else:
            return result[0] + 1
    
    def get_database_status(self):
        """Get current database status."""
        cursor = self.conn.cursor()
        
        # Count space groups
        cursor.execute("SELECT COUNT(*) FROM space_groups WHERE number <= 230")
        sg_count = cursor.fetchone()[0]
        
        # Count EBRs
        cursor.execute("SELECT COUNT(*) FROM ebrs")
        ebr_count = cursor.fetchone()[0]
        
        # Count irreps
        cursor.execute("SELECT COUNT(*) FROM irreps")
        irrep_count = cursor.fetchone()[0]
        
        # Get range of space groups
        cursor.execute("SELECT MIN(number), MAX(number) FROM space_groups WHERE number <= 230")
        min_sg, max_sg = cursor.fetchone()
        
        # Get missing space groups
        cursor.execute("""
            WITH RECURSIVE numbers(n) AS (
                SELECT 1
                UNION ALL
                SELECT n+1 FROM numbers WHERE n < 230
            )
            SELECT GROUP_CONCAT(n) FROM numbers 
            WHERE n NOT IN (SELECT number FROM space_groups WHERE number <= 230)
        """)
        missing_sgs = cursor.fetchone()[0]
        
        return {
            'space_groups': sg_count,
            'ebrs': ebr_count,
            'irreps': irrep_count,
            'min_sg': min_sg or 0,
            'max_sg': max_sg or 0,
            'missing_sgs': missing_sgs.split(',') if missing_sgs else []
        }
    
    def parse_wyckoff(self, entry):
        """Parse Wyckoff position entry like '1a(1)' -> ('1a', '1')."""
        m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
        if m:
            return m.group(1), m.group(2)
        else:
            return entry, ""
    
    def parse_orbital(self, entry):
        """Parse orbital entry like 'A↑G(1)' -> ('A↑G', 1)."""
        m = re.match(r"^(.+)\((\d+)\)$", entry)
        if m:
            return m.group(1), int(m.group(2))
        return entry, 1
    
    def parse_multiplicity(self, entry):
        """Parse multiplicity from entry like 'R2R2(2)' -> 2."""
        m = re.search(r"\((\d+)\)$", entry)
        if m:
            return int(m.group(1))
        return None
    
    def parse_irrep_label(self, entry):
        """Parse irrep label by removing the multiplicity part."""
        m = re.match(r"^(.+)\(\d+\)$", entry)
        return m.group(1) if m else entry
    
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
    
    def ingest_file(self, sg_number, filepath):
        """Ingest data from file for specified space group."""
        # Validate inputs
        if not (1 <= sg_number <= 230):
            raise ValueError(f"Space group number must be between 1 and 230, got {sg_number}")
        
        is_valid, message = self.validate_input_file(filepath)
        if not is_valid:
            raise ValueError(message)
        
        # Read and parse file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_lines = [line.rstrip("\n") for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        
        # Parse file content
        wyckoff_line, orbital_line, notes_line, kpoint_lines = self._parse_file_content(raw_lines)
        
        # Process data
        wyckoff_entries = wyckoff_line.split()
        orbital_entries = orbital_line.split()
        notes_entries   = notes_line.split()

        num_cols = len(wyckoff_entries)

        # If notes_entries has more tokens than columns, keep only the last `num_cols` tokens
        if len(notes_entries) > num_cols:
            notes_entries = notes_entries[-num_cols:]
        elif len(notes_entries) < num_cols:
            # Check if notes_entries is shorter due to a leading "Indecomposable" or similar line
            # This can happen if the notes line itself is "Indecomposable Indecomposable ..."
            # and a preceding line was just "Indecomposable"
            if len(notes_entries) > 0 and notes_entries[0].lower() in ["indecomposable", "decomposable"] and num_cols % len(notes_entries) == 0 :
                 # Heuristic: if the first token is a status and it evenly divides the expected columns,
                 # it might be a repeated status. This is a guess and might need refinement.
                 pass # Allow this case for now, it will be handled by the general logic or fail later if truly inconsistent.
            else:
                raise ValueError(
                    f"Column count mismatch: Wyckoff({num_cols}), "
                    f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
                )


        # Now we can assert consistency, ensuring notes_entries has been adjusted or was already correct
        if not (len(orbital_entries) == num_cols and (len(notes_entries) == num_cols or len(notes_entries) == 0 and num_cols > 0) ):
             # Allow notes_entries to be empty if num_cols > 0, assuming it means all are e.g. "indecomposable" by default (though current logic requires it)
            if not (len(notes_entries) == 0 and num_cols > 0) : # Only raise if not the "empty notes" case
                raise ValueError(
                    f"Column count mismatch after potential trimming: "
                    f"Wyckoff({len(wyckoff_entries)}), "
                    f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
                )
        
        # Parse k-point data
        kpoint_data = []
        for kp_line in kpoint_lines:
            tokens = kp_line.split()
            if tokens:
                kp_label = tokens[0].rstrip(":")
                reps = tokens[1:] # These are the cell strings for this k-point row
                kpoint_data.append((kp_label, reps))
        
        # Insert into database
        self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data)
        
        return num_cols
    
    def ingest_text(self, sg_number, raw_text):
        """Ingest data from raw text string for specified space group."""
        # Validate inputs
        if not (1 <= sg_number <= 230):
            raise ValueError(f"Space group number must be between 1 and 230, got {sg_number}")
        
        if not raw_text or not raw_text.strip():
            raise ValueError("Raw text cannot be empty")
        
        # Parse raw text data
        try:
            wyckoff_line, orbital_line, notes_line, kpoint_data = self._parse_raw_text(raw_text)
        except Exception as e:
            raise ValueError(f"Error parsing raw text: {str(e)}")
        
        wyckoff_entries = wyckoff_line.split()
        orbital_entries = orbital_line.split()
        notes_entries = notes_line.split()
        
        # Validate column consistency
        num_cols = len(wyckoff_entries)
        
        # Apply the same fix for trimming excess notes entries
        if len(notes_entries) > num_cols:
            notes_entries = notes_entries[-num_cols:]
        elif len(notes_entries) < num_cols:
             # Allow notes_entries to be empty if num_cols > 0
            if not (len(notes_entries) == 0 and num_cols > 0):
                raise ValueError(
                    f"Column count mismatch: Wyckoff({len(wyckoff_entries)}), "
                    f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
                )
        
        if not (len(orbital_entries) == num_cols and (len(notes_entries) == num_cols or len(notes_entries) == 0 and num_cols > 0)):
            if not (len(notes_entries) == 0 and num_cols > 0):
                raise ValueError(
                    f"Column count mismatch after potential trimming: "
                    f"Wyckoff({len(wyckoff_entries)}), "
                    f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
                )
        
        # If notes_entries became empty but should exist, this will be caught by _insert_data needing notes_entries[j]
        # Or we can pre-fill if necessary, assuming a default like "indecomposable" if notes are missing
        if num_cols > 0 and len(notes_entries) == 0:
            # This case implies notes might have been missing or unparsed.
            # Depending on requirements, either raise error or provide default.
            # For now, _insert_data will fail if it tries to access notes_entries[j] and it's empty.
            # Consider raising a more specific error here or defaulting notes.
            # print(f"Warning: Notes section resulted in empty notes_entries for {num_cols} columns. Processing will likely use defaults or fail if notes are mandatory per column.")
            pass


        return self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data)
    
    def _parse_file_content(self, raw_lines):
        """
        Parse file content (from a .txt) to extract Wyckoff, Orbital, notes, and k-points.
        First, collapse ANY trailing backslash+newline into a single space.
        """
        # 1) Re-join into one big string and collapse backslash+newline → space
        joined = "\n".join(raw_lines)
        joined = re.sub(r'\\\s*\n\s*', ' ', joined)

        # 2) Re-split into clean lines (dropping any blank lines)
        lines = [line.rstrip("\n") for line in joined.splitlines() if line.strip()]

        # 3) Now apply your existing "find Wyckoff / Band-Rep / Decomposable" logic
        wyckoff_line = None
        orbital_line  = None
        notes_line    = None
        kpoint_lines  = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("Wyckoff pos."):
                i += 1
                # skip any blank lines
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    wyckoff_line = lines[i]

            elif line.startswith("Band-Rep."):
                i += 1
                # skip any blank lines
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    orbital_line = lines[i]

            elif line.startswith("Decomposable"):
                # This line is "Decomposable" or "Decomposable\"
                i += 1 
                
                # Skip any blank lines immediately after "Decomposable"
                while i < len(lines) and not lines[i].strip():
                    i += 1
                
                # The next line could be a sub-header like "Indecomposable"
                # or the actual notes line. We need to be careful.
                # If the line after "Decomposable" has few words (e.g. "Indecomposable" itself)
                # and the line after that has many words (matching column count),
                # then the first is a header.
                if i < len(lines):
                    potential_header_line = lines[i]
                    if i + 1 < len(lines):
                        potential_notes_line = lines[i+1]
                        # Heuristic: if potential_header has 1-2 words and potential_notes has more, assume header
                        if len(potential_header_line.split()) <= 2 and len(potential_notes_line.split()) > len(potential_header_line.split()):
                             if any(kw in potential_header_line.lower() for kw in ["decomposable", "indecomposable"]):
                                i += 1 # Skip this assumed header line
                    
                # The current line (lines[i]) should now be the actual notes line
                while i < len(lines) and not lines[i].strip(): # Skip any further blank lines
                    i+=1
                if i < len(lines):
                    notes_line = lines[i]

                # everything after that is k-point data
                i += 1
                while i < len(lines):
                    if lines[i].strip():
                        kpoint_lines.append(lines[i])
                    i += 1
                break # Found all sections for decomposable part

            i += 1

        if not all([wyckoff_line, orbital_line]): # Notes line can be optional / tricky
            missing_parts = []
            if not wyckoff_line: missing_parts.append("Wyckoff pos.")
            if not orbital_line: missing_parts.append("Band-Rep.")
            # Do not error if notes_line is missing, handle in ingest logic
            # if not notes_line: missing_parts.append("Notes under Decomposable")
            if missing_parts:
                 raise ValueError(f"Could not locate required sections in file: {', '.join(missing_parts)}")

        if not notes_line: # If notes_line is still None after parsing attempts
            # This means the structure might be different, or notes are truly absent.
            # For robustness, we can try to find k-point lines directly if notes are missing.
            # Or, assume a default for notes (e.g., all "indecomposable") if that's a valid interpretation.
            # For now, we'll allow notes_line to be None and let downstream logic handle it.
            # print("Warning: Notes line under 'Decomposable' section not found or parsed.")
            pass


        return wyckoff_line, orbital_line, notes_line if notes_line else "", kpoint_lines

    def _parse_raw_text(self, raw_text):
        """
        Parse raw pasted text to extract:
          • wyckoff_line  (string)
          • orbital_line  (string)
          • notes_line    (string)
          • kpoint_data   (list of (kpt_label, cells_list)),
                           where each cells_list[j] is a single string
                           (e.g. "C1+(1) ⊕ C2+(1)").
        """
        # 1) Collapse any "\" + newline into a single space:
        text = re.sub(r'\\\s*\n\s*', ' ', raw_text.strip())

        # 2) Extract Wyckoff: between "Wyckoff pos." and "Band-Rep."
        wyckoff_match = re.search(
            r'Wyckoff pos\.(.*?)(?=Band-Rep\.)', text, # Ensure Band-Rep. is a lookahead
            re.IGNORECASE | re.DOTALL
        )
        # 3) Extract Orbital: between "Band-Rep." and "Decomposable"
        orbital_match = re.search(
            r'Band-Rep\.(.*?)(?=Decomposable)', text, # Ensure Decomposable is a lookahead
            re.IGNORECASE | re.DOTALL
        )
        if not wyckoff_match :
            raise ValueError("Could not find required section: Wyckoff pos.")
        if not orbital_match:
             raise ValueError("Could not find required section: Band-Rep. (ensure 'Decomposable' follows)")


        wyckoff_line = re.sub(r'\s+', ' ', wyckoff_match.group(1).strip())
        orbital_line = re.sub(r'\s+', ' ', orbital_match.group(1).strip())
        
        expected_cols = len(wyckoff_line.split())

        # 4) Extract Decomposable section and notes
        # The "Decomposable" keyword itself, optionally followed by '\' and then whitespace.
        # Then capture content (non-greedily) up to the k-point data.
        decomp_keyword_pattern = r'Decomposable\\?\s*'
        # Try to find the start of k-point data first to delimit the end of the decomposable section
        kpoint_section_start_match = re.search(r'([A-ZΓ]+:\s*\([^)]+\))', text, re.IGNORECASE)
        
        notes_line = ""
        if kpoint_section_start_match:
            kpoint_start_index = kpoint_section_start_match.start()
            # Search for Decomposable section before the k-points
            decomposable_intro_match = re.search(decomp_keyword_pattern, text[:kpoint_start_index], re.IGNORECASE | re.DOTALL)
            if decomposable_intro_match:
                # Content between "Decomposable..." and k-points
                potential_notes_section = text[decomposable_intro_match.end():kpoint_start_index].strip()
                # Split this section into lines to find the actual notes line
                decomp_lines = [line.strip() for line in potential_notes_section.split('\n') if line.strip()]
                
                # Try to find the notes line: it should have expected_cols tokens
                # It might be preceded by a header line like "Indecomposable"
                found_notes_line = None
                if len(decomp_lines) > 0:
                    if len(decomp_lines) == 1: # Only one line, assume it's the notes
                        if len(decomp_lines[0].split()) == expected_cols or expected_cols == 0: # check if it matches expected columns or if no columns expected
                            found_notes_line = decomp_lines[0]
                    elif len(decomp_lines) > 1: # Multiple lines, e.g. header then notes
                        # Prefer the last line if it matches column count
                        if len(decomp_lines[-1].split()) == expected_cols:
                            found_notes_line = decomp_lines[-1]
                        else: # Fallback: check other lines, e.g. if header was skipped
                            for ln_idx_rev in range(len(decomp_lines) -1, -1, -1):
                                if len(decomp_lines[ln_idx_rev].split()) == expected_cols:
                                    found_notes_line = decomp_lines[ln_idx_rev]
                                    break
                notes_line = found_notes_line if found_notes_line else ""
                if not notes_line and decomp_lines: # If no line matches expected_cols, and there was content
                     # Heuristic: take the last non-empty line in the section as notes if it's not clearly a header
                    if not (len(decomp_lines[-1].split()) == 1 and decomp_lines[-1].lower() in ["indecomposable", "decomposable"]):
                        notes_line = decomp_lines[-1]
                    # print(f"Warning: Notes line in Decomposable section parsed heuristically: '{notes_line}'")

            else: # Decomposable keyword not found before k-points
                # This case should ideally not happen if "Decomposable" is a required section marker
                # print("Warning: 'Decomposable' keyword not found before k-point data.")
                pass # notes_line remains ""
        else: # K-point data not found, cannot reliably delimit Decomposable section end
            raise ValueError("Could not find k-point data, which is needed to delimit the Decomposable section.")

        if not notes_line and expected_cols > 0 :
            # This implies notes were not successfully parsed.
            # Depending on strictness, could raise error or assign a default (e.g., all "indecomposable")
            # For now, let it be empty; _insert_data might fail or use defaults.
            # print(f"Warning: Notes line for {expected_cols} columns could not be definitively parsed from Decomposable section. Defaulting to empty notes.")
            pass


        # 5) Extract k-point block (re-use kpoint_section_start_match if available)
        if not kpoint_section_start_match: # Should have been found above
             raise ValueError("Could not find k-point data (logic error).")

        kpoint_text_block = text[kpoint_section_start_match.start():]
        
        # Pattern for individual k-point lines: Label:(coords) Irrep1 Irrep2 ...
        # We need to split the kpoint_text_block into lines, then parse each.
        kpoint_raw_lines = [line.strip() for line in kpoint_text_block.split('\n') if line.strip()]

        kpoint_data = []
        for kp_raw_line in kpoint_raw_lines:
            # Match KPointLabel:(coords)
            match_label_coords = re.match(r'([A-ZΓ]+):\s*\(([^)]+)\)\s*', kp_raw_line)
            if match_label_coords:
                kpt_label = match_label_coords.group(1)
                # The rest of the line after "Label:(coords) " is the irreps string
                irreps_txt_for_kpoint = kp_raw_line[match_label_coords.end():].strip()
                cells = parse_kpoint_cells(irreps_txt_for_kpoint) # This splits the rest of the line by whitespace
                
                # Validate that the number of irrep cells matches expected_cols (if expected_cols > 0)
                if expected_cols > 0 and len(cells) != expected_cols:
                    raise ValueError(
                        f"K-point line for {kpt_label} has {len(cells)} irrep entries, "
                        f"but expected {expected_cols} columns based on Wyckoff positions. Line: '{kp_raw_line}'"
                    )
                kpoint_data.append((kpt_label, cells))
            # else: this line is not a valid k-point starting line, skip or error
        
        if not kpoint_data and expected_cols > 0 : # if there are columns, there should be k-points
            raise ValueError("No k-point data lines were successfully parsed.")

        return wyckoff_line, orbital_line, notes_line, kpoint_data
    
    def _insert_data(self, sg_number, wyckoff_entries, orbital_entries, notes_entries_list_or_str, kpoint_data):
        """
        For each column j (0 <= j < len(wyckoff_entries)):
         - If column j is not decomposable: insert one EBR + its irreps (as before).
         - If column j is decomposable: insert one EBR but don't populate irreps.
        """
        cursor = self.conn.cursor()

        # 1) Upsert space_group
        cursor.execute("""
            INSERT OR REPLACE INTO space_groups(number, updated_at)
            VALUES (?, CURRENT_TIMESTAMP)
        """, (sg_number,))
        self.conn.commit()
        cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
        sg_id = cursor.fetchone()[0]

        # 2) Delete old EBRs (and their irreps) for this SG
        # Fetch existing EBR IDs for this space group
        cursor.execute("SELECT id FROM ebrs WHERE space_group_id = ?", (sg_id,))
        existing_ebr_ids = [row[0] for row in cursor.fetchall()]

        if existing_ebr_ids:
            # Delete irreps associated with these EBRs
            placeholders = ','.join('?' for _ in existing_ebr_ids)
            cursor.execute(f"DELETE FROM irreps WHERE ebr_id IN ({placeholders})", existing_ebr_ids)
            # Delete the EBRs themselves
            cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
            self.conn.commit()


        num_cols = len(wyckoff_entries)
        
        # Ensure notes_entries is a list, even if it was a single string that got split
        if isinstance(notes_entries_list_or_str, str):
            notes_entries = notes_entries_list_or_str.split()
        else: # Should already be a list from parsing logic
            notes_entries = notes_entries_list_or_str

        # If notes_entries is still not the right length, try to use defaults
        if num_cols > 0 and len(notes_entries) != num_cols:
            # print(f"Warning: Notes entries count ({len(notes_entries)}) mismatch with column count ({num_cols}).")
            # If notes are missing, assume "indecomposable" for all (or could raise error)
            if len(notes_entries) == 0 :
                # print(f"Assuming 'indecomposable' for all {num_cols} columns due to missing notes.")
                notes_entries = ["indecomposable"] * num_cols
            elif len(notes_entries) == 1 and num_cols > 1: # e.g. notes_line was just "Indecomposable"
                # print(f"Notes line was '{notes_entries[0]}', applying to all {num_cols} columns.")
                notes_entries = [notes_entries[0]] * num_cols
            else: # Unhandled mismatch
                 raise ValueError(
                    f"Critical notes count mismatch: Wyckoff columns ({num_cols}), "
                    f"Notes entries ({len(notes_entries)}). Content: '{notes_entries_list_or_str}'"
                )


        for j in range(num_cols):
            # a) Parse wyckoff_entries[j]
            wyck_raw = wyckoff_entries[j]
            m = re.match(r"^(\d+\w)(\([^)]+\))$", wyck_raw)
            if m:
                wyckoff_letter = m.group(1)
                site_symmetry  = m.group(2)
            else:
                wyckoff_letter = wyck_raw
                site_symmetry  = ""

            # b) Parse orbital_entries[j]
            orb_raw = orbital_entries[j]
            orb_label, orb_mult = self.parse_orbital(orb_raw)
            if orb_mult > 1:
                single_val = None
                double_val = orb_label
            else:
                single_val = orb_label
                double_val = None

            # Ensure notes_entries[j] is available
            if j >= len(notes_entries):
                # This should not happen if previous logic correctly populates notes_entries
                # Default to "indecomposable" if something went wrong
                note_j_status = "indecomposable" 
                # print(f"Warning: Missing note for column {j}, defaulting to 'indecomposable'.")
            else:
                note_j_status = notes_entries[j].lower()


            cursor.execute("""
                INSERT INTO ebrs (
                  space_group_id,
                  wyckoff_letter,
                  site_symmetry,
                  orbital_label,
                  time_reversal,
                  single_index,
                  double_index,
                  notes,
                  branch1_irreps,
                  branch2_irreps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sg_id,
                wyckoff_letter,
                site_symmetry,
                orb_label,
                1, # Assuming time_reversal is true, based on table default and typical use
                single_val,
                double_val,
                note_j_status, # "indecomposable" or "decomposable"
                None, # branch1_irreps not handled by this parser
                None  # branch2_irreps not handled by this parser
            ))
            self.conn.commit() # Commit after each EBR to get lastrowid correctly
            ebr_id = cursor.lastrowid

            # Insert irreps for this EBR column j
            for (kp_label, cells_for_this_kpoint) in kpoint_data:
                if j < len(cells_for_this_kpoint):
                    full_irrep_str = cells_for_this_kpoint[j] # e.g., "C1+(1) ⊕ C2+(1)" or "A1+(1)"
                    mult = self.parse_multiplicity(full_irrep_str)
                    label_no_mult = self.parse_irrep_label(full_irrep_str)
                    
                    cursor.execute("""
                        INSERT INTO irreps (
                          ebr_id,
                          k_point,
                          irrep_label,
                          multiplicity
                        ) VALUES (?, ?, ?, ?)
                    """, (ebr_id, kp_label, label_no_mult, mult))
                else:
                    # This indicates a mismatch between number of wyckoff columns and k-point irrep columns
                    raise ValueError(
                        f"Mismatch in irrep data for k-point {kp_label}. "
                        f"Expected data for column {j}, but only {len(cells_for_this_kpoint)} cells found. "
                        f"Line was: {cells_for_this_kpoint}"
                    )
            self.conn.commit() # Commit irreps for the current EBR
        
        return num_cols
    
    def query_space_group(self, sg_number):
        """Query data for a specific space group."""
        cursor = self.conn.cursor()
        
        # Get space group info
        cursor.execute("SELECT * FROM space_groups WHERE number = ?", (sg_number,))
        sg_data = cursor.fetchone()
        
        if not sg_data:
            return None
        
        # Get EBRs
        cursor.execute("""
            SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes
            FROM ebrs WHERE space_group_id = ?
        """, (sg_data[0],)) # sg_data[0] is the id
        ebrs = cursor.fetchall()
        
        # Get irreps for each EBR
        result = {
            'space_group': sg_data,
            'ebrs': []
        }
        
        for ebr_row in ebrs: # ebr_row is (id, wyckoff_letter, ...)
            ebr_id = ebr_row[0]
            cursor.execute("""
                SELECT k_point, irrep_label, multiplicity
                FROM irreps WHERE ebr_id = ?
                ORDER BY k_point
            """, (ebr_id,))
            irreps = cursor.fetchall()
            
            result['ebrs'].append({
                'ebr_data': ebr_row,
                'irreps': irreps
            })
        
        return result

def interactive_mode():
    """Run interactive mode for sequential data input."""
    db = EBRDatabaseManager()
    
    print("=== Elementary Band Representation Database Manager ===")
    print("Interactive Mode - Sequential Input from Space Group 1 to 230\n")
    
    while True:
        # Get current status
        status = db.get_database_status()
        next_sg = db.get_next_space_group()
        
        print(f"Database Status:")
        print(f"  Space Groups: {status['space_groups']}/230")
        print(f"  Total EBRs: {status['ebrs']}")
        print(f"  Total Irreps: {status['irreps']}")
        
        if status['space_groups'] > 0:
            print(f"  Range: SG {status['min_sg']} - {status['max_sg']}")
        
        if next_sg is None and status['space_groups'] == 230:
            print("\n🎉 All 230 space groups have been processed!")
            break
        elif next_sg is None and status['space_groups'] < 230:
             print(f"\n⚠️ All processed up to {status['max_sg']}, but not all 230 SGs are in DB.")
        
        print(f"\nNext space group to process: {next_sg if next_sg else 'All processed or starting fresh'}")
        print(f"Missing space groups: {len(status['missing_sgs'])}")
        
        print("\nOptions:")
        print("1. Input raw data for next space group (if available)")
        print("2. Input raw data for specific space group")
        print("3. Input from file for specific space group")
        print("4. Show database status")
        print("5. Query space group data")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            if next_sg is None:
                print("ℹ️ No 'next' space group to process automatically. Choose option 2 for specific SG.")
                continue
            # Input raw data for next space group
            sg_to_process = next_sg
            print(f"\nEntering data for space group {sg_to_process}")
            print("Paste your raw data below (Ctrl+D or an empty line then Enter, or two empty lines if term behaves oddly, to finish):")
            print("Example format: 'Wyckoff pos.\\n...\\nBand-Rep.\\n...\\nDecomposable\\n...'")
            print("-" * 80)
            
            raw_data_lines = []
            empty_line_count = 0
            while True:
                try:
                    line = input()
                    if not line.strip():
                        empty_line_count += 1
                        if empty_line_count >= 2: # Two consecutive empty lines to finish
                            break
                    else:
                        empty_line_count = 0 # Reset if line has content
                    raw_data_lines.append(line)
                except EOFError: # Ctrl+D
                    break
            
            raw_data = "\n".join(raw_data_lines) # FIXED: Join with newline

            if raw_data.strip():
                try:
                    num_ebrs = db.ingest_text(sg_to_process, raw_data.strip())
                    print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_to_process}")
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    print("Please check your data format and try again.")
            else:
                print("❌ No data entered.")
        
        elif choice == "2":
            # Input raw data for specific space group
            try:
                sg_num_str = input("Enter space group number (1-230): ")
                if not sg_num_str.isdigit():
                    print("❌ Invalid space group number.")
                    continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230):
                    print("❌ Space group number must be between 1 and 230.")
                    continue

                print(f"\nEntering data for space group {sg_num}")
                print("Paste your raw data below (Ctrl+D or an empty line then Enter, or two empty lines if term behaves oddly, to finish):")
                print("-" * 80)
                
                raw_data_lines = []
                empty_line_count = 0
                while True:
                    try:
                        line = input()
                        if not line.strip():
                            empty_line_count += 1
                            if empty_line_count >= 2: # Two consecutive empty lines to finish
                                break
                        else:
                            empty_line_count = 0
                        raw_data_lines.append(line)
                    except EOFError:
                        break
                
                raw_data = "\n".join(raw_data_lines) # FIXED: Join with newline

                if raw_data.strip():
                    num_ebrs = db.ingest_text(sg_num, raw_data.strip())
                    print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
                else:
                    print("❌ No data entered.")
                    
            except ValueError as e: # Handles int conversion error for sg_num as well
                print(f"❌ Error: {str(e)}")
            except Exception as e: # Catch other potential errors from ingest_text
                print(f"❌ Error processing data: {str(e)}")
        
        elif choice == "3":
            # Input from file for specific space group
            try:
                sg_num_str = input("Enter space group number (1-230): ")
                if not sg_num_str.isdigit():
                    print("❌ Invalid space group number.")
                    continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230):
                    print("❌ Space group number must be between 1 and 230.")
                    continue

                filepath = input(f"Enter file path for space group {sg_num}: ").strip()
                
                num_ebrs = db.ingest_file(sg_num, filepath)
                print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
            except ValueError as e:
                print(f"❌ Error: {str(e)}")
            except Exception as e:
                print(f"❌ Error processing file: {str(e)}")
        
        elif choice == "4":
            # Show detailed status (already shown at the top, but can repeat)
            print(f"\nDetailed Status:")
            print(f"  Space Groups Processed: {status['space_groups']}/230")
            print(f"  Min Processed SG: {status['min_sg']}, Max Processed SG: {status['max_sg']}")
            if status['missing_sgs']:
                missing_count = len(status['missing_sgs'])
                if missing_count <= 20:
                    print(f"  Missing space groups: {', '.join(status['missing_sgs'])}")
                else:
                    print(f"  Missing space groups: {missing_count} total")
                    print(f"  First 10 missing: {', '.join(status['missing_sgs'][:10])}")
            else:
                print("  ✅ All 230 space groups are present in the database.")

        
        elif choice == "5":
            # Query space group
            try:
                sg_num_str = input("Enter space group number to query: ")
                if not sg_num_str.isdigit():
                    print("❌ Invalid space group number.")
                    continue
                sg_num = int(sg_num_str)
                if not (1 <= sg_num <= 230): # Also validate query input
                    print("❌ Space group number must be between 1 and 230.")
                    continue

                result = db.query_space_group(sg_num)
                
                if result:
                    print(f"\n--- Space Group {sg_num} ---")
                    sg_db_id, sg_db_number, sg_db_symbol, sg_db_created, sg_db_updated = result['space_group']
                    print(f"  DB ID: {sg_db_id}, Number: {sg_db_number}, Symbol: {sg_db_symbol if sg_db_symbol else 'N/A'}")
                    print(f"  Created: {sg_db_created}")
                    print(f"  Updated: {sg_db_updated}")
                    print(f"  Number of EBRs: {len(result['ebrs'])}")
                    
                    for i, ebr_entry in enumerate(result['ebrs'], 1):
                        # ebr_data is (id, wyckoff_letter, site_symmetry, orbital_label, notes)
                        ebr_info = ebr_entry['ebr_data']
                        print(f"    EBR {i}: ID={ebr_info[0]}, {ebr_info[3]} at {ebr_info[1]}{ebr_info[2]} (Notes: {ebr_info[4]})")
                        if ebr_entry['irreps']:
                            for ir_kpt, ir_label, ir_mult in ebr_entry['irreps']:
                                mult_str = f"({ir_mult})" if ir_mult is not None else ""
                                print(f"      {ir_kpt}: {ir_label}{mult_str}")
                        else:
                            print("      No irreps found for this EBR.")
                else:
                    print(f"No data found for space group {sg_num}")
            except ValueError: # Handles int conversion for sg_num
                print("❌ Please enter a valid number for the space group.")
            except Exception as e:
                print(f"❌ Error during query: {str(e)}")

        
        elif choice == "6":
            break
        
        else:
            print("❌ Invalid choice, please enter a number between 1 and 6.")
        
        print("-" * 30) # Add spacing for clarity
    
    db.close()
    print("Database connection closed.")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced EBR Database Manager for Bilbao crystal server data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                    # Start interactive mode
  %(prog)s --sg 10 data_sg10.txt          # Import space group 10
  %(prog)s --status                        # Show database status
  %(prog)s --query --sg 10                # Query space group 10 data
        """
    )
    
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive mode for sequential input")
    parser.add_argument("--sg", type=int, 
                       help="Space group number (1-230)")
    parser.add_argument("--status", action="store_true",
                       help="Show database status")
    parser.add_argument("--query", action="store_true",
                       help="Query space group data (use with --sg)")
    parser.add_argument("input_file", nargs="?",
                       help="Path to input file (required for data import)")
    
    args = parser.parse_args()
    
    db_manager = None # Ensure db_manager is defined for potential closing in finally
    try:
        if args.interactive:
            interactive_mode() # interactive_mode manages its own DB connection

        elif args.status:
            db_manager = EBRDatabaseManager()
            status = db_manager.get_database_status()
            
            print("=== Database Status ===")
            percent_complete = (status['space_groups']/230*100) if status['space_groups'] > 0 else 0
            print(f"Space Groups: {status['space_groups']}/230 ({percent_complete:.1f}%)")
            print(f"Total EBRs: {status['ebrs']}")
            print(f"Total Irreps: {status['irreps']}")
            
            if status['space_groups'] > 0:
                print(f"Range: SG {status['min_sg']} - {status['max_sg']}")
            
            missing_count = len(status['missing_sgs'])
            if missing_count > 0 and status['space_groups'] < 230 : # only show missing if not all are done
                print(f"Missing: {missing_count} space groups")
                if missing_count <= 10:
                    print(f"Missing SGs: {', '.join(status['missing_sgs'])}")
            elif status['space_groups'] == 230:
                print("✅ All 230 space groups completed!")
            
        elif args.query and args.sg:
            if not (1 <= args.sg <= 230):
                 print(f"❌ Error: Space group number for query must be between 1 and 230, got {args.sg}", file=sys.stderr)
                 sys.exit(1)
            db_manager = EBRDatabaseManager()
            result = db_manager.query_space_group(args.sg)
            
            if result:
                print(f"=== Space Group {args.sg} ===")
                sg_db_id, sg_db_number, sg_db_symbol, sg_db_created, sg_db_updated = result['space_group']
                print(f"  DB ID: {sg_db_id}, Number: {sg_db_number}, Symbol: {sg_db_symbol if sg_db_symbol else 'N/A'}")
                print(f"  Created: {sg_db_created}")
                print(f"  Updated: {sg_db_updated}")
                print(f"  EBRs Found: {len(result['ebrs'])}")
                
                for i, ebr_entry in enumerate(result['ebrs'], 1):
                    ebr_info = ebr_entry['ebr_data']
                    irreps = ebr_entry['irreps']
                    print(f"\n  EBR {i}: ID={ebr_info[0]}, Orbital='{ebr_info[3]}' at Wyckoff='{ebr_info[1]}{ebr_info[2]}' (Notes: '{ebr_info[4]}')")
                    if irreps:
                        print(f"    K-points ({len(irreps)}):")
                        for k_point, irrep_label, mult in irreps:
                            mult_str = f"({mult})" if mult is not None else ""
                            print(f"      {k_point}: {irrep_label}{mult_str}")
                    else:
                        print("    No irreps found for this EBR.")

            else:
                print(f"No data found for space group {args.sg}")

        elif args.sg and args.input_file:
            if not (1 <= args.sg <= 230):
                 print(f"❌ Error: Space group number for import must be between 1 and 230, got {args.sg}", file=sys.stderr)
                 sys.exit(1)
            db_manager = EBRDatabaseManager()
            try:
                num_ebrs = db_manager.ingest_file(args.sg, args.input_file)
                print(f"✅ Successfully imported {num_ebrs} EBRs for space group {args.sg} from file '{args.input_file}'")
            except Exception as e:
                print(f"❌ Error ingesting file for SG {args.sg}: {str(e)}", file=sys.stderr)
                sys.exit(1)
        else:
            parser.print_help()
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # Catch other ValueErrors not caught by specific operations
        print(f"❌ Value error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # General exception handler
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if db_manager:
            db_manager.close()


if __name__ == "__main__":
    main()