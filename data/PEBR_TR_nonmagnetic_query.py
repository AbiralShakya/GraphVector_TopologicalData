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

import sqlite3
import argparse
import re
import sys
import os
from pathlib import Path

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
    def __init__(self, db_path="tqc.db"):
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
            raise ValueError(
                f"Column count mismatch: Wyckoff({num_cols}), "
                f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
            )

        # Now we can assert consistency
        if not (len(orbital_entries) == num_cols == len(notes_entries)):
            raise ValueError(
                f"Column count mismatch after trimming: "
                f"Wyckoff({len(wyckoff_entries)}), "
                f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
            )
        
        # Parse k-point data
        kpoint_data = []
        for kp_line in kpoint_lines:
            tokens = kp_line.split()
            if tokens:
                kp_label = tokens[0].rstrip(":")
                reps = tokens[1:]
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
            raise ValueError(
                f"Column count mismatch: Wyckoff({len(wyckoff_entries)}), "
                f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
            )
        
        if not (len(orbital_entries) == num_cols == len(notes_entries)):
            raise ValueError(
                f"Column count mismatch after trimming: "
                f"Wyckoff({len(wyckoff_entries)}), "
                f"Orbital({len(orbital_entries)}), Notes({len(notes_entries)})"
            )
        
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
                # skip the "Decomposable" line itself
                i += 1
                # skip the "Indecomposable" header line
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    i += 1  # move past the header line

                # next non-blank is actual notes
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    notes_line = lines[i]

                # everything after that is k-point data
                i += 1
                while i < len(lines):
                    if lines[i].strip():
                        kpoint_lines.append(lines[i])
                    i += 1
                break

            i += 1

        if not all([wyckoff_line, orbital_line, notes_line]):
            raise ValueError("Could not locate all required sections in file")

        return wyckoff_line, orbital_line, notes_line, kpoint_lines

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
            r'Wyckoff pos\.\s*(.+?)\s*Band-Rep\.', text,
            re.IGNORECASE | re.DOTALL
        )
        # 3) Extract Orbital: between "Band-Rep." and "Decomposable"
        orbital_match = re.search(
            r'Band-Rep\.\s*(.+?)\s*Decomposable', text,
            re.IGNORECASE | re.DOTALL
        )
        if not wyckoff_match or not orbital_match:
            raise ValueError("Could not find required sections (Wyckoff pos. or Band-Rep.)")

        wyckoff_line = re.sub(r'\s+', ' ', wyckoff_match.group(1).strip())
        orbital_line = re.sub(r'\s+', ' ', orbital_match.group(1).strip())

        # 4) Extract everything after "Decomposable" (allow optional backslash)
        decomp_pattern = r'Decomposable\\?\s*(.*?)(?=[A-ZΓ]+:\([^)]+\))'
        decomp_match = re.search(decomp_pattern, text, re.IGNORECASE | re.DOTALL)
        if not decomp_match:
            raise ValueError("Could not find Decomposable section")

        decomp_section = decomp_match.group(1).strip()
        lines = [line.strip() for line in decomp_section.split('\n') if line.strip()]

        expected_cols = len(wyckoff_line.split())
        notes_line = None
        for ln in lines:
            toks = ln.split()
            if len(toks) == expected_cols:
                notes_line = ln
                break
        if not notes_line:
            # fallback: pick the last "data" line
            data_lines = [ln for ln in lines
                          if not ln.lower().startswith('indecomposable') or len(ln.split()) > 1]
            if data_lines:
                notes_line = data_lines[-1]
            else:
                notes_line = lines[-1] if lines else ""
        if not notes_line:
            raise ValueError("Could not find notes data line")

        # 5) Extract k-point block
        kpoint_start_match = re.search(
            r'([A-ZΓ]+:\([^)]+\).*)', text,
            re.IGNORECASE | re.DOTALL
        )
        if not kpoint_start_match:
            raise ValueError("Could not find k-point data")

        kpoint_text = kpoint_start_match.group(1)
        kpoint_pattern = r'([A-ZΓ]+):\(([^)]+)\)'
        matches = list(re.finditer(kpoint_pattern, kpoint_text))

        kpoint_data = []
        for idx, m in enumerate(matches):
            kpt_label = m.group(1)
            # (coords = m.group(2) not strictly needed here)

            start_pos = m.end()
            if idx + 1 < len(matches):
                end_pos = matches[idx + 1].start()
                irreps_txt = kpoint_text[start_pos:end_pos]
            else:
                irreps_txt = kpoint_text[start_pos:]

            # Each whitespace-separated token (e.g. "C1+(1) ⊕ C2+(1)")
            cells = parse_kpoint_cells(irreps_txt)
            kpoint_data.append((kpt_label, cells))

        return wyckoff_line, orbital_line, notes_line, kpoint_data
    
    def _insert_data(self, sg_number, wyckoff_entries, orbital_entries, notes_entries, kpoint_data):
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
        cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
        self.conn.commit()

        num_cols = len(wyckoff_entries)
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

            note_j = notes_entries[j].lower()  # should be either "decomposable" or "indecomposable"

            if note_j == "indecomposable":
                # Insert exactly one EBR + its irreps (loop over kpoint_data[j] strings)
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
                    1,
                    single_val,
                    double_val,
                    note_j,          # typically "indecomposable"
                    None,
                    None
                ))
                self.conn.commit()
                ebr_id = cursor.lastrowid

                # Write all irreps exactly as strings, including any "⊕"
                for (kp_label, cells) in kpoint_data:
                    if j < len(cells):  # Safety check
                        full_str = cells[j]    # e.g. "C1+(1) ⊕ C2+(1)"
                        mult = self.parse_multiplicity(full_str)
                        label = self.parse_irrep_label(full_str)
                        cursor.execute("""
                            INSERT INTO irreps (
                              ebr_id,
                              k_point,
                              irrep_label,
                              multiplicity
                            ) VALUES (?, ?, ?, ?)
                        """, (ebr_id, kp_label, label, mult))
                self.conn.commit()

            else:  # note_j == "decomposable"
                # Insert one EBR row, but set branch1_irreps/branch2_irreps = NULL
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
                    1,
                    single_val,
                    double_val,
                    note_j,      # typically "decomposable"
                    None,
                    None
                ))
                self.conn.commit()
                ebr_id = cursor.lastrowid

                # For decomposable entries, still store the combined irrep representation
                for (kp_label, cells) in kpoint_data:
                    if j < len(cells):  # Safety check
                        full_str = cells[j]  # still a single string, e.g. "C1+(1) ⊕ C2+(1)"
                        mult = self.parse_multiplicity(full_str)
                        label = self.parse_irrep_label(full_str)
                        cursor.execute("""
                            INSERT INTO irreps (
                              ebr_id,
                              k_point,
                              irrep_label,
                              multiplicity
                            ) VALUES (?, ?, ?, ?)
                        """, (ebr_id, kp_label, label, mult))
                self.conn.commit()
        
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
        """, (sg_data[0],))
        ebrs = cursor.fetchall()
        
        # Get irreps for each EBR
        result = {
            'space_group': sg_data,
            'ebrs': []
        }
        
        for ebr in ebrs:
            cursor.execute("""
                SELECT k_point, irrep_label, multiplicity
                FROM irreps WHERE ebr_id = ?
                ORDER BY k_point
            """, (ebr[0],))
            irreps = cursor.fetchall()
            
            result['ebrs'].append({
                'ebr_data': ebr,
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
        
        if next_sg is None:
            print("\n🎉 All 230 space groups have been processed!")
            break
        
        print(f"\nNext space group to process: {next_sg}")
        print(f"Missing space groups: {len(status['missing_sgs'])}")
        
        print("\nOptions:")
        print("1. Input raw data for next space group")
        print("2. Input raw data for specific space group")
        print("3. Input from file for specific space group")
        print("4. Show database status")
        print("5. Query space group data")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            # Input raw data for next space group
            print(f"\nEntering data for space group {next_sg}")
            print("Paste your raw data below (press Enter twice when done):")
            print("Example format: 'Wyckoff pos.1a(1)1a(1)Band-Rep.A↑G(1)AA↑G(2)Decomposable\\...'")
            print("-" * 80)
            
            raw_data = ""
            empty_lines = 0
            while True:
                try:
                    line = input()
                    if not line.strip():
                        empty_lines += 1
                        if empty_lines >= 2:
                            break
                    else:
                        empty_lines = 0
                        raw_data += line + " "
                except EOFError:
                    break
            
            if raw_data.strip():
                try:
                    num_ebrs = db.ingest_text(next_sg, raw_data.strip())
                    print(f"✅ Successfully imported {num_ebrs} EBRs for space group {next_sg}")
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    print("Please check your data format and try again.")
            else:
                print("❌ No data entered.")
        
        elif choice == "2":
            # Input raw data for specific space group
            try:
                sg_num = int(input("Enter space group number (1-230): "))
                print(f"\nEntering data for space group {sg_num}")
                print("Paste your raw data below (press Enter twice when done):")
                print("-" * 80)
                
                raw_data = ""
                empty_lines = 0
                while True:
                    try:
                        line = input()
                        if not line.strip():
                            empty_lines += 1
                            if empty_lines >= 2:
                                break
                        else:
                            empty_lines = 0
                            raw_data += line + " "
                    except EOFError:
                        break
                
                if raw_data.strip():
                    num_ebrs = db.ingest_text(sg_num, raw_data.strip())
                    print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
                else:
                    print("❌ No data entered.")
                    
            except ValueError as e:
                print(f"❌ Error: {str(e)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == "3":
            # Input from file for specific space group
            try:
                sg_num = int(input("Enter space group number (1-230): "))
                filepath = input(f"Enter file path for space group {sg_num}: ").strip()
                
                num_ebrs = db.ingest_file(sg_num, filepath)
                print(f"✅ Successfully imported {num_ebrs} EBRs for space group {sg_num}")
            except ValueError as e:
                print(f"❌ Error: {str(e)}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == "4":
            # Show detailed status
            print(f"\nDetailed Status:")
            if status['missing_sgs']:
                missing_count = len(status['missing_sgs'])
                if missing_count <= 20:
                    print(f"Missing space groups: {', '.join(status['missing_sgs'])}")
                else:
                    print(f"Missing space groups: {missing_count} total")
                    print(f"First 10: {', '.join(status['missing_sgs'][:10])}")
        
        elif choice == "5":
            # Query space group
            try:
                sg_num = int(input("Enter space group number to query: "))
                result = db.query_space_group(sg_num)
                
                if result:
                    print(f"\nSpace Group {sg_num}:")
                    print(f"  Created: {result['space_group'][3]}")
                    print(f"  Updated: {result['space_group'][4]}")
                    print(f"  Number of EBRs: {len(result['ebrs'])}")
                    
                    for i, ebr_data in enumerate(result['ebrs'], 1):
                        ebr = ebr_data['ebr_data']
                        print(f"    EBR {i}: {ebr[3]} at {ebr[1]} ({ebr[4]})")
                else:
                    print(f"No data found for space group {sg_num}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "6":
            break
        
        else:
            print("❌ Invalid choice")
        
        print()  # Add spacing
    
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
    
    if args.interactive:
        interactive_mode()

    elif args.status:
        db = EBRDatabaseManager()
        status = db.get_database_status()
        
        print("=== Database Status ===")
        print(f"Space Groups: {status['space_groups']}/230 ({status['space_groups']/230*100:.1f}%)")
        print(f"Total EBRs: {status['ebrs']}")
        print(f"Total Irreps: {status['irreps']}")
        
        if status['space_groups'] > 0:
            print(f"Range: SG {status['min_sg']} - {status['max_sg']}")
        
        missing_count = len(status['missing_sgs'])
        if missing_count > 0:
            print(f"Missing: {missing_count} space groups")
            if missing_count <= 10:
                print(f"Missing SGs: {', '.join(status['missing_sgs'])}")
        else:
            print("✅ All space groups completed!")
        
        db.close()
    elif args.query and args.sg:
        db = EBRDatabaseManager()
        result = db.query_space_group(args.sg)
        
        if result:
            print(f"=== Space Group {args.sg} ===")
            sg_data = result['space_group']
            print(f"Created: {sg_data[3]}")
            print(f"Updated: {sg_data[4]}")
            print(f"EBRs: {len(result['ebrs'])}")
            
            for i, ebr_data in enumerate(result['ebrs'], 1):
                ebr = ebr_data['ebr_data']
                irreps = ebr_data['irreps']
                print(f"\nEBR {i}: {ebr[3]} at {ebr[1]}{ebr[2]} ({ebr[4]})")
                print(f"  K-points: {len(irreps)}")
                for k_point, irrep_label, mult in irreps:
                    mult_str = f"({mult})" if mult else ""
                    print(f"    {k_point}: {irrep_label}{mult_str}")
        else:
            print(f"No data found for space group {args.sg}")

    elif args.sg and args.input_file:
        db = EBRDatabaseManager()
        try:
            num_ebrs = db.ingest_file(args.sg, args.input_file)
            print(f"✅ Successfully imported {num_ebrs} EBRs for space group {args.sg}")
        except Exception as e:
            print(f"❌ Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
        finally:
            db.close()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()