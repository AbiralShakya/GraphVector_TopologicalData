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

# import sqlite3
# import argparse
# import re
# import sys
# import os
# from pathlib import Path

# # Helper functions (parse_kpoint_cells, split_decomposable_column remain the same)
# def parse_kpoint_cells(irreps_txt):
#     return [tok.strip() for tok in irreps_txt.split() if tok.strip()]

# class EBRDatabaseManager:
#     def __init__(self, db_path="pebr_tr_nonmagnetic.db"):
#         self.db_path = db_path
#         self.conn = None
#         self.connect()
#         self.create_tables()
    
#     def connect(self):
#         self.conn = sqlite3.connect(self.db_path)
#         self.conn.execute("PRAGMA foreign_keys = ON")
    
#     def close(self):
#         if self.conn:
#             self.conn.close()
    
#     def create_tables(self): # Included for context, schema is important
#         cursor = self.conn.cursor()
        
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS space_groups (
#           id            INTEGER PRIMARY KEY AUTOINCREMENT,
#           number        INTEGER UNIQUE NOT NULL,
#           symbol        TEXT,
#           created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#           updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
        
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
#           created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#           updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
#         );
#         """)
        
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ebr_decomposition_branches (
#           id                  INTEGER PRIMARY KEY AUTOINCREMENT,
#           ebr_id              INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
#           decomposition_index INTEGER NOT NULL,
#           branch1_irreps      TEXT NOT NULL,
#           branch2_irreps      TEXT NOT NULL,
#           created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#           UNIQUE(ebr_id, decomposition_index)
#         );
#         """)

#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS irreps (
#           id           INTEGER PRIMARY KEY AUTOINCREMENT,
#           ebr_id       INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
#           k_point      TEXT    NOT NULL,
#           irrep_label  TEXT    NOT NULL,
#           multiplicity INTEGER,
#           created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
        
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_sg_number ON space_groups(number);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_sg ON ebrs(space_group_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_ebr ON irreps(ebr_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_kpoint ON irreps(k_point);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_decomp_ebr_id ON ebr_decomposition_branches(ebr_id);")

#         cursor.execute("""
#         CREATE TRIGGER IF NOT EXISTS update_space_groups_updated_at
#         AFTER UPDATE ON space_groups FOR EACH ROW
#         BEGIN UPDATE space_groups SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
#         """)
#         cursor.execute("""
#         CREATE TRIGGER IF NOT EXISTS update_ebrs_updated_at
#         AFTER UPDATE ON ebrs FOR EACH ROW
#         BEGIN UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
#         """)
#         self.conn.commit()

#     def parse_wyckoff(self, entry):
#         m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
#         return (m.group(1), m.group(2)) if m else (entry, "")

#     def parse_orbital(self, entry):
#         m = re.match(r"^(.+)\((\d+)\)$", entry)
#         return (m.group(1), int(m.group(2))) if m else (entry, 1)

#     def parse_multiplicity(self, entry):
#         m = re.search(r"\((\d+)\)$", entry)
#         return int(m.group(1)) if m else None

#     def parse_irrep_label(self, entry):
#         m = re.match(r"^(.+)\(\d+\)$", entry)
#         return m.group(1) if m else entry

#     def _parse_file_content(self, raw_lines): # Assumed from previous, ensure it's correct
#         joined = "\n".join(raw_lines)
#         joined = re.sub(r'\\\s*\n\s*', ' ', joined)
#         lines = [line.rstrip("\n") for line in joined.splitlines() if line.strip()]
#         wyckoff_line, orbital_line, notes_line, kpoint_lines = None, None, None, []
#         i = 0
#         while i < len(lines):
#             line = lines[i]
#             if line.startswith("Wyckoff pos."):
#                 i += 1
#                 while i < len(lines) and not lines[i].strip(): i += 1
#                 if i < len(lines): wyckoff_line = lines[i]
#             elif line.startswith("Band-Rep."):
#                 i += 1
#                 while i < len(lines) and not lines[i].strip(): i += 1
#                 if i < len(lines): orbital_line = lines[i]
#             elif line.startswith("Decomposable"):
#                 i += 1 
#                 while i < len(lines) and not lines[i].strip(): i += 1
#                 if i < len(lines):
#                     potential_header_line = lines[i]
#                     if i + 1 < len(lines):
#                         potential_notes_line = lines[i+1]
#                         if len(potential_header_line.split()) <= 2 and len(potential_notes_line.split()) > len(potential_header_line.split()):
#                              if any(kw in potential_header_line.lower() for kw in ["decomposable", "indecomposable"]):
#                                 i += 1 
#                 while i < len(lines) and not lines[i].strip(): i+=1
#                 if i < len(lines): notes_line = lines[i]
#                 i += 1
#                 while i < len(lines):
#                     if lines[i].strip(): kpoint_lines.append(lines[i])
#                     i += 1
#                 break 
#             i += 1
#         if not all([wyckoff_line, orbital_line]):
#             missing = [p for p, v in [("Wyckoff pos.", wyckoff_line), ("Band-Rep.", orbital_line)] if not v]
#             raise ValueError(f"Could not locate required sections in file: {', '.join(missing)}")
#         return wyckoff_line, orbital_line, notes_line if notes_line else "", kpoint_lines

#     def _parse_raw_text(self, raw_text): # Assumed from previous, ensure it's correct
#         text = re.sub(r'\\\s*\n\s*', ' ', raw_text.strip())
#         wyckoff_match = re.search(r'Wyckoff pos\.(.*?)(?=Band-Rep\.)', text, re.IGNORECASE | re.DOTALL)
#         orbital_match = re.search(r'Band-Rep\.(.*?)(?=Decomposable)', text, re.IGNORECASE | re.DOTALL)
#         if not wyckoff_match: raise ValueError("Could not find: Wyckoff pos.")
#         if not orbital_match: raise ValueError("Could not find: Band-Rep. (ensure 'Decomposable' follows)")

#         wyckoff_line = re.sub(r'\s+', ' ', wyckoff_match.group(1).strip())
#         orbital_line = re.sub(r'\s+', ' ', orbital_match.group(1).strip())
#         expected_cols = len(wyckoff_line.split())
#         notes_line = ""
#         kpoint_section_start_match = re.search(r'([A-ZΓ]+:\s*\([^)]+\))', text, re.IGNORECASE)
        
#         if kpoint_section_start_match:
#             kpoint_start_index = kpoint_section_start_match.start()
#             decomp_keyword_pattern = r'Decomposable\\?\s*'
#             decomposable_intro_match = re.search(decomp_keyword_pattern, text[:kpoint_start_index], re.IGNORECASE | re.DOTALL)
#             if decomposable_intro_match:
#                 potential_notes_section = text[decomposable_intro_match.end():kpoint_start_index].strip()
#                 decomp_lines = [ln.strip() for ln in potential_notes_section.split('\n') if ln.strip()]
#                 found_notes = None
#                 if decomp_lines:
#                     if len(decomp_lines) == 1 and (len(decomp_lines[0].split()) == expected_cols or expected_cols == 0):
#                         found_notes = decomp_lines[0]
#                     elif len(decomp_lines) > 1:
#                         if len(decomp_lines[-1].split()) == expected_cols: found_notes = decomp_lines[-1]
#                         else:
#                             for ln_idx_rev in range(len(decomp_lines) - 1, -1, -1):
#                                 if len(decomp_lines[ln_idx_rev].split()) == expected_cols:
#                                     found_notes = decomp_lines[ln_idx_rev]; break
#                 notes_line = found_notes if found_notes else ""
#                 if not notes_line and decomp_lines and not (len(decomp_lines[-1].split()) == 1 and decomp_lines[-1].lower() in ["indecomposable", "decomposable"]):
#                     notes_line = decomp_lines[-1]
#         else: raise ValueError("K-point data not found (needed to delimit Decomposable section).")

#         if not kpoint_section_start_match: raise ValueError("K-point data not found (logic error).")
#         kpoint_text_block = text[kpoint_section_start_match.start():]
#         kpoint_raw_lines = [ln.strip() for ln in kpoint_text_block.split('\n') if ln.strip()]
#         kpoint_data = []
#         for kp_raw_line in kpoint_raw_lines:
#             match_label_coords = re.match(r'([A-ZΓ]+):\s*\(([^)]+)\)\s*', kp_raw_line)
#             if match_label_coords:
#                 kpt_label = match_label_coords.group(1)
#                 irreps_txt_for_kpoint = kp_raw_line[match_label_coords.end():].strip()
#                 cells = parse_kpoint_cells(irreps_txt_for_kpoint)
#                 if expected_cols > 0 and len(cells) != expected_cols:
#                     raise ValueError(f"K-point {kpt_label}: {len(cells)} irreps, expected {expected_cols}. Line: '{kp_raw_line}'")
#                 kpoint_data.append((kpt_label, cells))
#         if not kpoint_data and expected_cols > 0: raise ValueError("No k-point data lines parsed.")
#         return wyckoff_line, orbital_line, notes_line, kpoint_data
    

#     def _manage_branches_interactively_for_ebr(self, ebr_id, wyckoff_info, orbital_info, called_from_ingest=False):
#         """
#         Manages decomposition branches for a single specified EBR.
#         wyckoff_info should be a string like "1a(2/m)".
#         orbital_info should be the orbital label string.
#         """
#         print(f"\n--- Managing Branches for EBR ID: {ebr_id} ({wyckoff_info} - {orbital_info}) ---")
        
#         if called_from_ingest:
#             # Ensure it's actually marked decomposable before asking (already checked by caller)
#             choice = input("This EBR is marked 'decomposable'. Add/edit decomposition branches now? (y/n): ").strip().lower()
#             if choice != 'y':
#                 print(f"ℹ️ Branch input for EBR ID {ebr_id} skipped. Manage later via main menu option 4 if needed.")
#                 return

#         while True:
#             existing_branches = self.get_ebr_decomposition_branches(ebr_id)
#             if existing_branches:
#                 print("  Existing Branches:")
#                 for br_idx, b1, b2, ca in existing_branches:
#                     print(f"    Index {br_idx}: Branch1='{b1}', Branch2='{b2}' (Added: {ca})")
#             else:
#                 print("  No existing branches for this EBR.")

#             branch_action = input("\n  Branch actions: (A)dd new, (D)elete specific, (C)lear all for this EBR, (R)eturn/Done: ").strip().upper()
            
#             if branch_action == 'A':
#                 try:
#                     dec_idx_str = input("    Enter decomposition index (e.g., 1, 2): ").strip()
#                     if not dec_idx_str.isdigit():
#                         print("❌ Decomposition index must be a number.")
#                         continue
#                     dec_idx = int(dec_idx_str)
                    
#                     b1_str = input(f"    Enter Branch 1 irreps string for index {dec_idx} (comma-separated): ").strip()
#                     b2_str = input(f"    Enter Branch 2 irreps string for index {dec_idx} (comma-separated): ").strip()
                    
#                     if not b1_str or not b2_str: 
#                         print("❌ Branch strings cannot be empty.")
#                         continue
                    
#                     if self.add_ebr_decomposition_branch(ebr_id, dec_idx, b1_str, b2_str):
#                         print(f"✅ Branch index {dec_idx} added/updated for EBR ID {ebr_id}.")
#                     else: 
#                         # add_ebr_decomposition_branch prints its own error
#                         pass 
#                 except ValueError: 
#                     print("❌ Invalid input for decomposition index (must be a number).")
#                 except Exception as e:
#                      print(f"❌ Error during add operation: {e}")

#             elif branch_action == 'D':
#                 try:
#                     dec_idx_del_str = input("    Enter decomposition index to delete: ").strip()
#                     if not dec_idx_del_str.isdigit():
#                         print("❌ Decomposition index must be a number.")
#                         continue
#                     dec_idx_del = int(dec_idx_del_str)
                    
#                     if self.delete_ebr_decomposition_branch(ebr_id, dec_idx_del):
#                         print(f"✅ Branch index {dec_idx_del} deleted for EBR ID {ebr_id}.")
#                     else: 
#                         print(f"ℹ️ Branch index {dec_idx_del} not found or could not be deleted.");
#                 except ValueError: 
#                     print("❌ Invalid input for decomposition index (must be a number).")
#                 except Exception as e:
#                     print(f"❌ Error during delete operation: {e}")
            
#             elif branch_action == 'C':
#                 confirm_clear = input(f"⚠️ Are you sure you want to delete ALL branches for EBR ID {ebr_id}? (yes/no): ").strip().lower()
#                 if confirm_clear == 'yes':
#                     count = self.delete_all_ebr_decomposition_branches(ebr_id)
#                     print(f"✅ Deleted {count} branches for EBR ID {ebr_id}.")
#                 else: 
#                     print("ℹ️ Clear operation cancelled.")
            
#             elif branch_action == 'R': 
#                 print(f"Finished managing branches for EBR ID {ebr_id}.")
#                 break
            
#             else: 
#                 print("❌ Invalid branch action. Please choose A, D, C, or R.")
    
    
#     def ingest_file(self, sg_number, filepath, prompt_for_branches_immediately=False):
#         if not (1 <= sg_number <= 230): raise ValueError(f"SG number {sg_number} out of range 1-230")
#         is_valid, msg = self.validate_input_file(filepath)
#         if not is_valid: raise ValueError(msg)
        
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f: 
#                 # Filter out empty lines before passing to _parse_file_content
#                 raw_lines = [line.rstrip("\n") for line in f if line.strip()]
#         except Exception as e:
#             raise ValueError(f"Error reading file '{filepath}': {str(e)}")

#         if not raw_lines:
#             raise ValueError(f"Input file '{filepath}' is empty or contains only whitespace.")

#         wyckoff_line, orbital_line, notes_line, kpoint_lines_parsed = self._parse_file_content(raw_lines)
        
#         wyckoff_entries = wyckoff_line.split()
#         orbital_entries = orbital_line.split()
#         # notes_line can be empty if not found, handle it
#         notes_entries = notes_line.split() if notes_line else [] 
        
#         num_cols = len(wyckoff_entries)

#         # Column consistency checks
#         if not (len(orbital_entries) == num_cols):
#             raise ValueError(f"Column count mismatch: Wyckoff({num_cols}), Orbital({len(orbital_entries)})")
        
#         # Notes entries consistency (allow notes_entries to be empty if notes_line was empty)
#         if notes_line and len(notes_entries) != num_cols:
#             if len(notes_entries) > num_cols:
#                 notes_entries = notes_entries[-num_cols:]
#             # If notes_entries is shorter but not empty, it's an issue unless it's a single token meant for all
#             elif len(notes_entries) > 0 and len(notes_entries) < num_cols and not (len(notes_entries) == 1 and num_cols > 1):
#                  raise ValueError(f"Notes line has {len(notes_entries)} tokens, expected {num_cols} or 1 (if applying to all). Notes: '{notes_line}'")

#         kpoint_data_final = []
#         for kp_line_str in kpoint_lines_parsed:
#             tokens = kp_line_str.split()
#             if tokens: 
#                 # tokens[1:] should be a list of cell strings for this k-point row
#                 kpoint_data_final.append((tokens[0].rstrip(":"), tokens[1:]))
        
#         inserted_ebrs = self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data_final)
        
#         if prompt_for_branches_immediately:
#             print(f"\n--- Checking {len(inserted_ebrs)} imported EBRs for branch input (SG {sg_number}) ---")
#             for ebr_info in inserted_ebrs:
#                 # Trigger if note is "decomposable" (add other conditions like specific placeholders if needed)
#                 if ebr_info['note'].lower() == 'decomposable': 
#                     self._manage_branches_interactively_for_ebr(
#                         ebr_info['ebr_id'], 
#                         f"{ebr_info['wyckoff']}{ebr_info['site_symmetry']}", 
#                         ebr_info['orbital'],
#                         called_from_ingest=True
#                     )
#         return len(inserted_ebrs)

#     def ingest_text(self, sg_number, raw_text, prompt_for_branches_immediately=False):
#         if not (1 <= sg_number <= 230): raise ValueError(f"SG number {sg_number} out of range 1-230")
#         if not raw_text or not raw_text.strip(): raise ValueError("Raw text cannot be empty")
        
#         wyckoff_line, orbital_line, notes_line, kpoint_data_parsed = self._parse_raw_text(raw_text) 
        
#         wyckoff_entries = wyckoff_line.split()
#         orbital_entries = orbital_line.split()
#         notes_entries = notes_line.split() if notes_line else []
        
#         num_cols = len(wyckoff_entries)

#         if not (len(orbital_entries) == num_cols):
#             raise ValueError(f"Column count mismatch: Wyckoff({num_cols}), Orbital({len(orbital_entries)})")

#         if notes_line and len(notes_entries) != num_cols:
#             if len(notes_entries) > num_cols:
#                 notes_entries = notes_entries[-num_cols:]
#             elif len(notes_entries) > 0 and len(notes_entries) < num_cols and not (len(notes_entries) == 1 and num_cols > 1):
#                  raise ValueError(f"Notes line has {len(notes_entries)} tokens, expected {num_cols} or 1. Notes: '{notes_line}'")
        
#         inserted_ebrs = self._insert_data(sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data_parsed)
        
#         if prompt_for_branches_immediately:
#             print(f"\n--- Checking {len(inserted_ebrs)} imported EBRs for branch input (SG {sg_number}) ---")
#             for ebr_info in inserted_ebrs:
#                 if ebr_info['note'].lower() == 'decomposable': 
#                     self._manage_branches_interactively_for_ebr(
#                         ebr_info['ebr_id'], 
#                         f"{ebr_info['wyckoff']}{ebr_info['site_symmetry']}", 
#                         ebr_info['orbital'],
#                         called_from_ingest=True
#                     )
#         return len(inserted_ebrs)

#     def _insert_data(self, sg_number, wyckoff_entries, orbital_entries, notes_entries_list, kpoint_data_list_of_tuples):
#         """
#         Inserts EBRs and their irreps.
#         Returns a list of dictionaries, each with info about an inserted EBR.
#         """
#         cursor = self.conn.cursor()
#         # Ensure space group exists and get its ID
#         cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
#         sg_row = cursor.fetchone()
#         if sg_row:
#             sg_id = sg_row[0]
#             # Update its timestamp if we are about to add/replace EBRs
#             cursor.execute("UPDATE space_groups SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (sg_id,))
#         else:
#             # Insert new space group, symbol can be updated later if parsed
#             cursor.execute("INSERT INTO space_groups(number, symbol) VALUES (?, ?)", (sg_number, None))
#             sg_id = cursor.lastrowid # Get ID of newly inserted space group
        
#         # Delete old EBRs for this space group. ON DELETE CASCADE handles irreps and branches.
#         cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
#         self.conn.commit()

#         num_cols = len(wyckoff_entries)
        
#         # Normalize notes_final list
#         notes_final = notes_entries_list
#         if num_cols > 0 and len(notes_final) != num_cols:
#             if not notes_final: # If notes_entries_list was empty
#                 notes_final = ["indecomposable"] * num_cols # Default all to indecomposable
#             elif len(notes_final) == 1 and num_cols > 1: # If only one note token was provided for multiple columns
#                 notes_final = [notes_final[0]] * num_cols # Apply that single note to all
#             else: # Unhandled mismatch
#                  raise ValueError(
#                     f"Critical notes count mismatch: Wyckoff columns ({num_cols}), "
#                     f"Notes entries ({len(notes_entries_list)} after processing). Cannot normalize."
#                 )

#         inserted_ebr_info_list = [] 

#         for j in range(num_cols):
#             wyck_letter, site_sym = self.parse_wyckoff(wyckoff_entries[j])
#             orb_label, orb_mult = self.parse_orbital(orbital_entries[j])
#             single_val, double_val = (None, orb_label) if orb_mult > 1 else (orb_label, None)
            
#             # Ensure note_j_status is available
#             note_j_status = "indecomposable" # Default
#             if j < len(notes_final):
#                 note_j_status = notes_final[j].lower()
#             # else: it remains 'indecomposable' if notes_final was shorter (should be handled by normalization above)

#             cursor.execute("""
#                 INSERT INTO ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, 
#                                   time_reversal, single_index, double_index, notes) 
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#             """, (sg_id, wyck_letter, site_sym, orb_label, 1, single_val, double_val, note_j_status))
#             ebr_id = cursor.lastrowid # Get ID of newly inserted EBR
            
#             inserted_ebr_info_list.append({
#                 'ebr_id': ebr_id, 
#                 'note': note_j_status, 
#                 'wyckoff': wyck_letter, 
#                 'site_symmetry': site_sym, # Added for more complete info
#                 'orbital': orb_label
#             })
            
#             for (kp_label, cells_for_kpoint) in kpoint_data_list_of_tuples:
#                 if j < len(cells_for_kpoint):
#                     full_irrep_str = cells_for_kpoint[j]
#                     mult = self.parse_multiplicity(full_irrep_str)
#                     label_no_mult = self.parse_irrep_label(full_irrep_str)
#                     cursor.execute("""
#                         INSERT INTO irreps (ebr_id, k_point, irrep_label, multiplicity) 
#                         VALUES (?, ?, ?, ?)
#                     """, (ebr_id, kp_label, label_no_mult, mult))
#                 else: 
#                     raise ValueError(
#                         f"Mismatch in irrep data for k-point {kp_label}. "
#                         f"Expected data for column {j+1}, but only {len(cells_for_kpoint)} cells found. "
#                         f"Cells: {cells_for_kpoint}"
#                     )
#         self.conn.commit()
#         return inserted_ebr_info_list

#     # --- Methods for managing decomposition branches ---
#     def get_decomposable_ebrs(self, sg_id_or_num):
#         """Lists decomposable EBRs for a given space group ID or number."""
#         cursor = self.conn.cursor()
#         sg_id = sg_id_or_num
#         if isinstance(sg_id_or_num, int) and sg_id_or_num <= 230: # assume it's sg number
#             cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_id_or_num,))
#             sg_row = cursor.fetchone()
#             if not sg_row: return []
#             sg_id = sg_row[0]
        
#         cursor.execute("""
#             SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes
#             FROM ebrs 
#             WHERE space_group_id = ? AND lower(notes) = 'decomposable'
#             ORDER BY id
#         """, (sg_id,))
#         return cursor.fetchall()

#     def add_ebr_decomposition_branch(self, ebr_id, decomposition_index, branch1_str, branch2_str):
#         cursor = self.conn.cursor()
#         try:
#             cursor.execute("INSERT OR REPLACE INTO ebr_decomposition_branches (ebr_id, decomposition_index, branch1_irreps, branch2_irreps) VALUES (?, ?, ?, ?)", 
#                            (ebr_id, decomposition_index, branch1_str, branch2_str))
#             self.conn.commit()
#             # Trigger should handle updated_at on ebrs if we decide to link it that way,
#             # but for now, this action doesn't directly modify the ebrs row itself.
#             # If an updated_at on ebrs is desired when a branch is added, it needs explicit update.
#             # For now, let's assume updated_at on ebrs is for its direct fields.
#             # cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)) 
#             # self.conn.commit()
#             return True
#         except sqlite3.Error as e: print(f"Error adding branch: {e}"); return False

#     def get_ebr_decomposition_branches(self, ebr_id):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT decomposition_index, branch1_irreps, branch2_irreps, created_at FROM ebr_decomposition_branches WHERE ebr_id = ? ORDER BY decomposition_index", (ebr_id,))
#         return cursor.fetchall()

#     def delete_ebr_decomposition_branch(self, ebr_id, decomposition_index):
#         cursor = self.conn.cursor()
#         cursor.execute("DELETE FROM ebr_decomposition_branches WHERE ebr_id = ? AND decomposition_index = ?", (ebr_id, decomposition_index))
#         rc = cursor.rowcount
#         self.conn.commit()
#         # if rc > 0: cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)); self.conn.commit()
#         return rc > 0

#     def delete_all_ebr_decomposition_branches(self, ebr_id):
#         cursor = self.conn.cursor()
#         cursor.execute("DELETE FROM ebr_decomposition_branches WHERE ebr_id = ?", (ebr_id,))
#         rc = cursor.rowcount
#         self.conn.commit()
#         # if rc > 0: cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)); self.conn.commit()
#         return rc > 0
        
#     # --- Status and Query methods ---
#     def get_next_space_group(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT MAX(number) FROM space_groups WHERE number <= 230")
#         r = cursor.fetchone()
#         if r[0] is None: return 1
#         return r[0] + 1 if r[0] < 230 else None

#     def get_database_status(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM space_groups WHERE number <= 230")
#         sg_c = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM ebrs")
#         ebr_c = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM irreps")
#         irrep_c = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM ebr_decomposition_branches")
#         branch_c = cursor.fetchone()[0]
#         cursor.execute("SELECT MIN(number), MAX(number) FROM space_groups WHERE number <= 230")
#         min_sg, max_sg = cursor.fetchone()
#         cursor.execute("""
#             WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM nums WHERE n<230)
#             SELECT GROUP_CONCAT(n) FROM nums WHERE n NOT IN (SELECT number FROM space_groups WHERE number <= 230)
#         """)
#         missing = cursor.fetchone()[0]
#         return {
#             'space_groups': sg_c, 'ebrs': ebr_c, 'irreps': irrep_c, 
#             'branches': branch_c, 'min_sg': min_sg or 0, 'max_sg': max_sg or 0,
#             'missing_sgs': missing.split(',') if missing else []
#         }

#     def query_space_group(self, sg_number):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM space_groups WHERE number = ?", (sg_number,))
#         sg_data = cursor.fetchone()
#         if not sg_data: return None
        
#         cursor.execute("""
#             SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes, created_at, updated_at
#             FROM ebrs WHERE space_group_id = ? ORDER BY id
#         """, (sg_data[0],))
#         ebrs_rows = cursor.fetchall()
        
#         result = {'space_group': sg_data, 'ebrs': []}
#         for ebr_row in ebrs_rows:
#             ebr_id = ebr_row[0]
#             cursor.execute("SELECT k_point, irrep_label, multiplicity FROM irreps WHERE ebr_id = ? ORDER BY k_point", (ebr_id,))
#             irreps = cursor.fetchall()
            
#             # Fetch decomposition branches
#             branches = self.get_ebr_decomposition_branches(ebr_id)
            
#             result['ebrs'].append({'ebr_data': ebr_row, 'irreps': irreps, 'branches': branches})
#         return result


import sqlite3
import argparse
import re
import sys
import os
from pathlib import Path

def parse_kpoint_cells(irreps_txt):
    """
    Parses a string containing multiple k-point irrep cells into a list of individual cell strings.
    Handles cells like "A1(2)", "2 Γ3Γ4(2)", and "D3D3(2) ⊕ D4D4(2)" as single tokens.
    """
    if not irreps_txt:
        return []

    # Define a pattern for a single irrep unit, e.g., "A1+(1)", "AgAg↑G(4)", "D1+D2+(2)"
    # It's a name (letters, numbers, +, -, arrows) followed by (digits)
    irrep_unit_pattern = r"[A-ZΓa-z0-9+\-↑↓]+?\([^\)]+\)" # Non-greedy name to handle internal +,-

    # Define patterns for different types of cells, ordered by complexity to ensure correct matching
    # 1. Oplus-combined form: IRREP_UNIT space* OPLUS space* IRREP_UNIT
    oplus_form_pattern = rf"({irrep_unit_pattern}\s*⊕\s*{irrep_unit_pattern})"
    # 2. Numbered form: NUMBER space+ IRREP_UNIT
    numbered_form_pattern = rf"(\d+\s+{irrep_unit_pattern})"
    # 3. Simple irrep unit form (must be last as it's a sub-pattern of others)
    simple_form_pattern = rf"({irrep_unit_pattern})"

    # Combine patterns for re.match in a loop. Order matters: try most complex first.
    # This regex will try to match one full cell at the beginning of the string.
    cell_parser_regex = re.compile(
        f"(?:{oplus_form_pattern}|{numbered_form_pattern}|{simple_form_pattern})",
        re.UNICODE
    )

    cells = []
    current_text = irreps_txt.strip()
    while current_text:
        match = cell_parser_regex.match(current_text)
        if match:
            token = match.group(0).strip() # group(0) is the whole match
            cells.append(token)
            current_text = current_text[match.end():].strip() # Move to the next part of the string
        else:
            # If no pattern matches, it implies either end of string or an unparsable segment.
            # This could happen if there's unexpected text or malformed irreps.
            # For robustness, one might try a simple split for the remainder or log an error.
            # However, if the regex is comprehensive, this 'else' block should ideally not be hit
            # if the input is well-formed according to the defined patterns.
            if current_text: # If there's text left that didn't match
                # Fallback: Take the next non-whitespace block. This might misinterpret complex cases.
                # Consider raising an error here if strict parsing is required.
                # print(f"Warning: Unrecognized pattern in k-point cells near: '{current_text[:30]}'. Attempting fallback split.", file=sys.stderr)
                next_chunk = current_text.split(maxsplit=1)
                if next_chunk:
                    cells.append(next_chunk[0])
                    current_text = next_chunk[1].strip() if len(next_chunk) > 1 else ""
                else: # Should not happen if current_text is non-empty
                    break
            else: # current_text is empty
                break
    return cells

class EBRDatabaseManager:
    # ... (constructor, connect, close, create_tables, parsing helpers, validation, 
    #      _parse_file_content, _parse_raw_text, branch management methods like 
    #      get_decomposable_ebrs, add_ebr_decomposition_branch, etc. are assumed from previous full code) ...

    def __init__(self, db_path="pebr_tr_nonmagnetic.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def create_tables(self): # Included for context, schema is important
        cursor = self.conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS space_groups (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          number        INTEGER UNIQUE NOT NULL,
          symbol        TEXT,
          created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
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
          created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ebr_decomposition_branches (
          id                  INTEGER PRIMARY KEY AUTOINCREMENT,
          ebr_id              INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
          decomposition_index INTEGER NOT NULL,
          branch1_irreps      TEXT NOT NULL,
          branch2_irreps      TEXT NOT NULL,
          created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(ebr_id, decomposition_index)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS irreps (
          id           INTEGER PRIMARY KEY AUTOINCREMENT,
          ebr_id       INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
          k_point      TEXT    NOT NULL,
          irrep_label  TEXT    NOT NULL,
          multiplicity INTEGER,
          created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sg_number ON space_groups(number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_sg ON ebrs(space_group_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_ebr ON irreps(ebr_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_irrep_kpoint ON irreps(k_point);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ebr_decomp_ebr_id ON ebr_decomposition_branches(ebr_id);")

        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_space_groups_updated_at
        AFTER UPDATE ON space_groups FOR EACH ROW
        BEGIN UPDATE space_groups SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
        """)
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_ebrs_updated_at
        AFTER UPDATE ON ebrs FOR EACH ROW
        BEGIN UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id; END;
        """)
        self.conn.commit()

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
        cursor.execute("SELECT id, wyckoff_letter, site_symmetry, orbital_label, notes FROM ebrs WHERE space_group_id = ? AND lower(notes) = 'decomposable' ORDER BY id", (sg_id,))
        return cursor.fetchall()

    def add_ebr_decomposition_branch(self, ebr_id, decomposition_index, branch1_str, branch2_str):
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT OR REPLACE INTO ebr_decomposition_branches (ebr_id, decomposition_index, branch1_irreps, branch2_irreps) VALUES (?, ?, ?, ?)", 
                           (ebr_id, decomposition_index, branch1_str, branch2_str))
            self.conn.commit()
            # Trigger should handle updated_at on ebrs if we decide to link it that way,
            # but for now, this action doesn't directly modify the ebrs row itself.
            # If an updated_at on ebrs is desired when a branch is added, it needs explicit update.
            # For now, let's assume updated_at on ebrs is for its direct fields.
            # cursor.execute("UPDATE ebrs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (ebr_id,)) 
            # self.conn.commit()
            return True
        except sqlite3.Error as e: print(f"Error adding branch: {e}"); return False

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

    def _insert_data(self, sg_number, wyckoff_entries, orbital_entries, notes_entries_list, kpoint_data_list_of_tuples):
        """
        Inserts EBRs and their irreps.
        Returns a list of dictionaries, each with info about an inserted EBR.
        """
        cursor = self.conn.cursor()
        # Ensure space group exists and get its ID
        cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
        sg_row = cursor.fetchone()
        if sg_row:
            sg_id = sg_row[0]
            # Update its timestamp if we are about to add/replace EBRs
            cursor.execute("UPDATE space_groups SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (sg_id,))
        else:
            # Insert new space group, symbol can be updated later if parsed
            cursor.execute("INSERT INTO space_groups(number, symbol) VALUES (?, ?)", (sg_number, None))
            sg_id = cursor.lastrowid # Get ID of newly inserted space group
        
        # Delete old EBRs for this space group. ON DELETE CASCADE handles irreps and branches.
        cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
        self.conn.commit()

        num_cols = len(wyckoff_entries)
        
        # Normalize notes_final list
        notes_final = notes_entries_list
        if num_cols > 0 and len(notes_final) != num_cols:
            if not notes_final: # If notes_entries_list was empty
                notes_final = ["indecomposable"] * num_cols # Default all to indecomposable
            elif len(notes_final) == 1 and num_cols > 1: # If only one note token was provided for multiple columns
                notes_final = [notes_final[0]] * num_cols # Apply that single note to all
            else: # Unhandled mismatch
                 raise ValueError(
                    f"Critical notes count mismatch: Wyckoff columns ({num_cols}), "
                    f"Notes entries ({len(notes_entries_list)} after processing). Cannot normalize."
                )

        inserted_ebr_info_list = [] 

        for j in range(num_cols):
            wyck_letter, site_sym = self.parse_wyckoff(wyckoff_entries[j])
            orb_label, orb_mult = self.parse_orbital(orbital_entries[j])
            single_val, double_val = (None, orb_label) if orb_mult > 1 else (orb_label, None)
            
            # Ensure note_j_status is available
            note_j_status = "indecomposable" # Default
            if j < len(notes_final):
                note_j_status = notes_final[j].lower()
            # else: it remains 'indecomposable' if notes_final was shorter (should be handled by normalization above)

            cursor.execute("""
                INSERT INTO ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, 
                                  time_reversal, single_index, double_index, notes) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (sg_id, wyck_letter, site_sym, orb_label, 1, single_val, double_val, note_j_status))
            ebr_id = cursor.lastrowid # Get ID of newly inserted EBR
            
            inserted_ebr_info_list.append({
                'ebr_id': ebr_id, 
                'note': note_j_status, 
                'wyckoff': wyck_letter, 
                'site_symmetry': site_sym, # Added for more complete info
                'orbital': orb_label
            })
            
            for (kp_label, cells_for_kpoint) in kpoint_data_list_of_tuples:
                if j < len(cells_for_kpoint):
                    full_irrep_str = cells_for_kpoint[j]
                    mult = self.parse_multiplicity(full_irrep_str)
                    label_no_mult = self.parse_irrep_label(full_irrep_str)
                    cursor.execute("""
                        INSERT INTO irreps (ebr_id, k_point, irrep_label, multiplicity) 
                        VALUES (?, ?, ?, ?)
                    """, (ebr_id, kp_label, label_no_mult, mult))
                else: 
                    raise ValueError(
                        f"Mismatch in irrep data for k-point {kp_label}. "
                        f"Expected data for column {j+1}, but only {len(cells_for_kpoint)} cells found. "
                        f"Cells: {cells_for_kpoint}"
                    )
        self.conn.commit()
        return inserted_ebr_info_list

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
            # Ensure it's actually marked decomposable before asking (already checked by caller)
            choice = input("This EBR is marked 'decomposable'. Add/edit decomposition branches now? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"ℹ️ Branch input for EBR ID {ebr_id} skipped. Manage later via main menu option 4 if needed.")
                return

        while True:
            existing_branches = self.get_ebr_decomposition_branches(ebr_id)
            if existing_branches:
                print("  Existing Branches:")
                for br_idx, b1, b2, ca in existing_branches:
                    print(f"    Index {br_idx}: Branch1='{b1}', Branch2='{b2}' (Added: {ca})")
            else:
                print("  No existing branches for this EBR.")

            branch_action = input("\n  Branch actions: (A)dd new, (D)elete specific, (C)lear all for this EBR, (R)eturn/Done: ").strip().upper()
            
            if branch_action == 'A':
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
                    else: 
                        # add_ebr_decomposition_branch prints its own error
                        pass 
                except ValueError: 
                    print("❌ Invalid input for decomposition index (must be a number).")
                except Exception as e:
                     print(f"❌ Error during add operation: {e}")

            elif branch_action == 'D':
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
            
            elif branch_action == 'C':
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
                print("❌ Invalid branch action. Please choose A, D, C, or R.")
    
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
    db = EBRDatabaseManager()
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
                ebr_map = {} # Maps display index (string) to the full EBR data tuple
                for i, ebr_data_tuple in enumerate(decomposable_ebrs):
                    display_idx_str = str(i + 1)
                    ebr_map[display_idx_str] = ebr_data_tuple 
                    # ebr_data_tuple is (id, wyckoff_letter, site_symmetry, orbital_label, notes)
                    print(f"  {display_idx_str}. EBR ID: {ebr_data_tuple[0]}, Wyckoff: {ebr_data_tuple[1]}{ebr_data_tuple[2]}, Orbital: {ebr_data_tuple[3]}")
                
                ebr_choice_idx_str = input("Select EBR by number to manage branches: ").strip()
                selected_ebr_tuple = ebr_map.get(ebr_choice_idx_str)

                if not selected_ebr_tuple: 
                    print("❌ Invalid selection.")
                    continue
                
                selected_ebr_id = selected_ebr_tuple[0]
                # Construct wyckoff_info and orbital_info from the tuple
                wyck_info = f"{selected_ebr_tuple[1]}{selected_ebr_tuple[2]}" # wyckoff_letter + site_symmetry
                orb_info = selected_ebr_tuple[3] # orbital_label

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
            db_manager_cli = EBRDatabaseManager()
            status = db_manager_cli.get_database_status()
            print("=== Database Status ===")
            pc = (status['space_groups']/230*100) if status['space_groups'] > 0 else 0
            print(f"Space Groups: {status['space_groups']}/230 ({pc:.1f}%)")
            print(f"Total EBRs: {status['ebrs']}, Irreps: {status['irreps']}, Branches: {status['branches']}")
            if status['space_groups'] == 230: print("✅ All 230 SGs completed!")
            elif status['missing_sgs']: print(f"Missing SGs: {len(status['missing_sgs'])}")
        elif args.query and args.sg:
            if not (1 <= args.sg <= 230): print(f"❌ Query SG {args.sg} out of range.", file=sys.stderr); sys.exit(1)
            db_manager_cli = EBRDatabaseManager()
            res = db_manager_cli.query_space_group(args.sg)
            if res: print(f"Data found for SG {args.sg}. EBRs: {len(res['ebrs'])}. Use interactive query for full details.")
            else: print(f"No data for SG {args.sg}.")
        elif args.sg and args.input_file:
            if not (1 <= args.sg <= 230): print(f"❌ Import SG {args.sg} out of range.", file=sys.stderr); sys.exit(1)
            db_manager_cli = EBRDatabaseManager()
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