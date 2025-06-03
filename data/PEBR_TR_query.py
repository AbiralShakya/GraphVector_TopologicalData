# physical elementary band representation w/ time reversal from bilbao to simple sql db

#!/usr/bin/env python3
"""
ingest_tqc.py

Usage:
    python ingest_tqc.py --sg <space_group_number> <input_file>

Example:
    python ingest_tqc.py --sg 10 raw_sg10.txt

This script reads a raw TQC/EBR data file (in a specific whitespace/tab-delimited format),
parses it, and populates a local SQLite database (tqc.db) with three tables:

  1) space_groups
  2) ebrs
  3) irreps

The expected input format (example for SG=10) is:

    Wyckoff pos.
    1a(1)   1a(1)
    Band-Rep.
    A↑G(1)  AA↑G(2)
    Decomposable\
    Indecomposable   Indecomposable
    R:(1/2,1/2,1/2)     R1(1)    R2R2(2)
    T:(0,1/2,1/2)       T1(1)    T2T2(2)
    U:(1/2,0,1/2)       U1(1)    U2U2(2)
    V:(1/2,1/2,0)       V1(1)    V2V2(2)
    X:(1/2,0,0)         X1(1)    X2X2(2)
    Y:(0,1/2,0)         Y1(1)    Y2Y2(2)
    Z:(0,0,1/2)         Z1(1)    Z2Z2(2)
    Γ:(0,0,0)           Γ1(1)    Γ2Γ2(2)

Notes on the format:
- After the literal line "Wyckoff pos." comes a line of space/tab‐delimited entries, e.g. "1a(1)  1a(1)".
- After "Band-Rep." comes a line of orbitals with multiplicities, e.g. "A↑G(1)   AA↑G(2)".
- After a line starting with "Decomposable" comes the line of "Indecomposable" or "Decomposable" flags.
- All lines after that are k-point lines of the form "K:(coords)   rep_j  rep_j  …", split on whitespace.
"""

import sqlite3
import argparse
import re
import sys

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS space_groups (
      id      INTEGER PRIMARY KEY AUTOINCREMENT,
      number  INTEGER UNIQUE NOT NULL,
      symbol  TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ebrs (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      space_group_id INTEGER NOT NULL REFERENCES space_groups(id),
      wyckoff_letter TEXT    NOT NULL,
      site_symmetry  TEXT    NOT NULL,
      orbital_label  TEXT    NOT NULL,
      time_reversal  BOOLEAN NOT NULL DEFAULT 0,
      single_index   TEXT,
      double_index   TEXT,
      notes          TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS irreps (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      ebr_id       INTEGER NOT NULL REFERENCES ebrs(id),
      k_point      TEXT    NOT NULL,
      irrep_label  TEXT    NOT NULL,
      multiplicity INTEGER
    );
    """)
    conn.commit()

def parse_wyckoff(entry):
    """
    Given a string like "1a(1)", return ("1a", "1").
    If no "(...)", return (entry, "").
    """
    m = re.match(r"^(\d+\w)\(([^)]+)\)$", entry)
    if m:
        return m.group(1), m.group(2)
    else:
        return entry, ""

def parse_orbital(entry):
    """
    Given a string like "A↑G(1)", return "A↑G".
    If no "(...)", return entry.
    """
    m = re.match(r"^(.+)\(\d+\)$", entry)
    return m.group(1) if m else entry

def parse_multiplicity(entry):
    """
    Given a string like "R2R2(2)" or "R1(1)", return integer 2 or 1.
    If no "(n)", return None.
    """
    m = re.search(r"\((\d+)\)$", entry)
    if m:
        return int(m.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Ingest raw TQC/EBR data into SQLite 'tqc.db'.")
    parser.add_argument("--sg", type=int, required=True, help="Space group number (1-230)")
    parser.add_argument("input_file", help="Path to the raw TQC data file")
    args = parser.parse_args()

    # 1) Read all lines from input, skipping truly empty lines
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_lines = [line.rstrip("\n") for line in f if line.strip()]
    except FileNotFoundError:
        sys.stderr.write(f"Error: Cannot open '{args.input_file}'.\n")
        sys.exit(1)

    # 2) Locate and parse the labeled rows:
    #    - "Wyckoff pos." → next line = wyckoff_line
    #    - "Band-Rep."    → next line = orbital_line
    #    - line starting with "Decomposable" → next line = notes_line
    #    - everything after notes_line is k-point rows
    wyckoff_line = None
    orbital_line  = None
    notes_line    = None
    kpoint_lines  = []

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith("Wyckoff pos."):
            i += 1
            while i < len(raw_lines) and not raw_lines[i].strip():
                i += 1
            if i < len(raw_lines):
                wyckoff_line = raw_lines[i]

        elif line.startswith("Band-Rep."):
            i += 1
            while i < len(raw_lines) and not raw_lines[i].strip():
                i += 1
            if i < len(raw_lines):
                orbital_line = raw_lines[i]

        elif line.startswith("Decomposable"):
            i += 1
            while i < len(raw_lines) and not raw_lines[i].strip():
                i += 1
            if i < len(raw_lines):
                notes_line = raw_lines[i]
            # Now all remaining non-empty lines are k-point entries
            i += 1
            while i < len(raw_lines):
                if raw_lines[i].strip():
                    kpoint_lines.append(raw_lines[i])
                i += 1
            break  # we’ve collected all k-point lines

        i += 1

    if not (wyckoff_line and orbital_line and notes_line):
        sys.stderr.write("Error: Could not locate all required sections (Wyckoff pos, Band-Rep, Decomposable).\n")
        sys.exit(1)

    # Split each data line on whitespace
    wyckoff_entries = wyckoff_line.split()
    orbital_entries  = orbital_line.split()
    notes_entries    = notes_line.split()

    # Parse k-point rows into (k_point, [rep_for_col_0, rep_for_col_1, ...])
    kpoint_data = []
    for kp_line in kpoint_lines:
        tokens = kp_line.split()
        if not tokens:
            continue
        kp_label = tokens[0].rstrip(":")
        reps     = tokens[1:]
        kpoint_data.append((kp_label, reps))

    num_cols = len(wyckoff_entries)
    if not (len(orbital_entries) == num_cols == len(notes_entries)):
        sys.stderr.write(
            f"Error: Column count mismatch:\n"
            f"  Wyckoff entries: {len(wyckoff_entries)}\n"
            f"  Orbital entries: {len(orbital_entries)}\n"
            f"  Notes entries:   {len(notes_entries)}\n"
        )
        sys.exit(1)

    # 3) Open (or create) SQLite database "tqc.db" and create tables if needed
    conn = sqlite3.connect("tqc.db")
    create_tables(conn)
    cursor = conn.cursor()

    # 4) Insert or ignore the space group
    cursor.execute("INSERT OR IGNORE INTO space_groups(number) VALUES (?)", (args.sg,))
    conn.commit()
    cursor.execute("SELECT id FROM space_groups WHERE number = ?", (args.sg,))
    sg_id = cursor.fetchone()[0]

    # 5) For each column j, insert an EBR and its irreps
    for j in range(num_cols):
        # Parse Wyckoff letter & site symmetry
        wyck_raw = wyckoff_entries[j]
        wyck_letter, site_sym = parse_wyckoff(wyck_raw)

        # Parse orbital (drop the "(n)" multiplicity suffix)
        orb_raw = orbital_entries[j]
        orb_label = parse_orbital(orb_raw)

        # Notes (e.g. "Indecomposable" or "Decomposable")
        note = notes_entries[j]

        # Insert into ebrs (time_reversal = False, single_index/double_index left NULL)
        cursor.execute("""
            INSERT INTO ebrs (
              space_group_id, wyckoff_letter, site_symmetry, orbital_label, notes
            ) VALUES (?, ?, ?, ?, ?)
        """, (sg_id, wyck_letter, site_sym, orb_label, note))
        conn.commit()
        ebr_id = cursor.lastrowid

        # Insert each k-point → irrep for this column
        for kp_label, reps in kpoint_data:
            if j < len(reps):
                irrep_str = reps[j]
                multiplicity = parse_multiplicity(irrep_str)
                cursor.execute("""
                    INSERT INTO irreps (ebr_id, k_point, irrep_label, multiplicity)
                    VALUES (?, ?, ?, ?)
                """, (ebr_id, kp_label, irrep_str, multiplicity))
        conn.commit()

    print(f"Successfully ingested {num_cols} EBR(s) into space group {args.sg} in 'tqc.db'.")
    conn.close()

if __name__ == "__main__":
    main()
