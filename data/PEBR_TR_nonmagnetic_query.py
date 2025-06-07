import sqlite3
import re

class EBRDatabaseManager:
    """
    Manages the SQLite database for NON-MAGNETIC band representations
    with a robust, flexible schema.
    """
    def __init__(self, db_path="pebr_tr_nonmagnetic_final.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        print(f"✅ Non-Magnetic DB Manager connected to '{db_path}'.")
        self.create_tables()

    def close(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Creates all necessary tables with the new flexible schema."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS space_groups (
          id            INTEGER PRIMARY KEY,
          number        INTEGER UNIQUE NOT NULL,
          created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ebrs (
          id                   INTEGER PRIMARY KEY,
          space_group_id       INTEGER NOT NULL REFERENCES space_groups(id) ON DELETE CASCADE,
          wyckoff_letter       TEXT NOT NULL,
          site_symmetry        TEXT NOT NULL,
          orbital_label        TEXT NOT NULL,
          orbital_multiplicity INTEGER NOT NULL,
          notes                TEXT,
          created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # FIX: Drop the old rigid branches table and add the new flexible one
        cursor.execute("DROP TABLE IF EXISTS ebr_decomposition_branches;")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ebr_decomposition_items (
          id                  INTEGER PRIMARY KEY,
          space_group_id      INTEGER NOT NULL REFERENCES space_groups(id) ON DELETE CASCADE,
          ebr_id              INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
          decomposition_index INTEGER NOT NULL,
          branch_index        INTEGER NOT NULL,
          irreps_string       TEXT    NOT NULL
        );
        """)
        
        # FIX: Add table for EBR-list decompositions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ebr_decomposition_ebrs (
            id                  INTEGER PRIMARY KEY,
            space_group_id      INTEGER NOT NULL REFERENCES space_groups(id) ON DELETE CASCADE,
            parent_ebr_id       INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
            decomposes_into     TEXT NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS irreps (
          id             INTEGER PRIMARY KEY,
          space_group_id INTEGER NOT NULL REFERENCES space_groups(id) ON DELETE CASCADE,
          ebr_id         INTEGER NOT NULL REFERENCES ebrs(id) ON DELETE CASCADE,
          k_point        TEXT    NOT NULL,
          irrep_label    TEXT    NOT NULL,
          multiplicity   INTEGER
        );
        """)
        self.conn.commit()

    # --- Parsing Helper Methods ---
    def parse_wyckoff(self, entry):
        m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
        return (m.group(1), m.group(2)) if m else (entry, "")

    def parse_orbital(self, entry):
        m = re.match(r"^(.+?)\((\d+)\)$", entry)
        return (m.group(1), int(m.group(2))) if m else (entry, 1)

    def parse_multiplicity(self, entry):
        m = re.search(r"\((\d+)\)$", entry)
        return int(m.group(1)) if m else None

    def parse_irrep_label(self, entry):
        m = re.match(r"^(.+)\(\d+\)$", entry)
        return m.group(1) if m else entry
    
    # --- New Database Interaction Methods ---
    def get_or_create_space_group_id(self, sg_number: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            cursor.execute("INSERT INTO space_groups(number) VALUES (?)", (sg_number,))
            self.conn.commit()
            return cursor.lastrowid

    def delete_ebrs_for_sg(self, space_group_id: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (space_group_id,))
        self.conn.commit()
        
    def log_processed_space_group(self, sg_number: int):
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO space_groups(number) VALUES (?)", (sg_number,))
        self.conn.commit()

    def insert_single_ebr(self, space_group_id: int, ebr_data: dict) -> int:
        cursor = self.conn.cursor()
        wyck_letter, site_sym = self.parse_wyckoff(ebr_data.get("wyckoff", ""))
        orb_label, orb_mult = self.parse_orbital(ebr_data.get("bandrep", ""))
        note = ebr_data.get("decomposability", {}).get("type", "indecomposable")

        cursor.execute("""
            INSERT INTO ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (space_group_id, wyck_letter, site_sym, orb_label, orb_mult, note))
        ebr_id = cursor.lastrowid

        for k_point, irrep_str in ebr_data.get("kpoints", {}).items():
            mult = self.parse_multiplicity(irrep_str)
            label_no_mult = self.parse_irrep_label(irrep_str)
            cursor.execute("""
                INSERT INTO irreps (space_group_id, ebr_id, k_point, irrep_label, multiplicity)
                VALUES (?, ?, ?, ?, ?)
            """, (space_group_id, ebr_id, k_point, label_no_mult, mult))
        
        self.conn.commit()
        return ebr_id
    
    def add_decomposition_item(self, sg_id, ebr_id, decomp_idx, branch_idx, irrep_string):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ebr_decomposition_items (space_group_id, ebr_id, decomposition_index, branch_index, irreps_string)
            VALUES (?, ?, ?, ?, ?)
        """, (sg_id, ebr_id, decomp_idx, branch_idx, irrep_string))
        self.conn.commit()

    def add_ebr_list_decomposition(self, sg_id, parent_ebr_id, ebr_list):
        cursor = self.conn.cursor()
        for ebr_string in ebr_list:
            cursor.execute("""
                INSERT INTO ebr_decomposition_ebrs (space_group_id, parent_ebr_id, decomposes_into)
                VALUES (?, ?, ?)
            """, (sg_id, parent_ebr_id, ebr_string))
        self.conn.commit()

