# visualization for tqc nonmagnetic, output of PEBR_TR_nonmagnetic_query.py file

import sqlite3

def print_space_groups(conn):
    cur = conn.execute("SELECT id, number, created_at FROM space_groups ORDER BY number;")
    print("\n>>> space_groups:")
    for row in cur:
        print(f"  id={row[0]}, number={row[1]}, created_at={row[2]}")

def print_ebrs_for_sg(conn, sg_number):
    query = """
    SELECT
      e.id,
      e.wyckoff_letter,
      e.site_symmetry,
      e.orbital_label,
      e.notes
    FROM ebrs e
    JOIN space_groups sg      ON e.space_group_id = sg.id
    WHERE sg.number = ?
    ORDER BY e.id;
    """
    cur = conn.execute(query, (sg_number,))
    print(f"\n>>> EBRs for space_group={sg_number}:")
    for ebr_id, wyck, symm, orb, notes in cur:
        print(f"  ebr_id={ebr_id}: {wyck}  {symm}  {orb}  notes={notes}")

def print_irreps_for_ebr(conn, ebr_id):
    query = """
    SELECT k_point, irrep_label, multiplicity
    FROM irreps
    WHERE ebr_id = ?
    ORDER BY k_point;
    """
    cur = conn.execute(query, (ebr_id,))
    print(f"\n>>> Irreps for ebr_id={ebr_id}:")
    for kpt, irrep, mult in cur:
        print(f"  {kpt}: {irrep}  (mult={mult})")

def dump_all_irreps_for_sg(conn, sg_number):
    query = """
    SELECT
      e.id            AS ebr_id,
      e.wyckoff_letter,
      e.site_symmetry,
      e.orbital_label,
      i.k_point,
      i.irrep_label,
      i.multiplicity
    FROM space_groups sg
    JOIN ebrs e   ON e.space_group_id = sg.id
    JOIN irreps i ON i.ebr_id = e.id
    WHERE sg.number = ?
    ORDER BY e.id, i.k_point;
    """
    cur = conn.execute(query, (sg_number,))
    print(f"\n>>> All irreps for space_group={sg_number}:")
    last_ebr = None
    for ebr_id, wyck, symm, orb, kpt, irrep, mult in cur:
        if ebr_id != last_ebr:
            print(f"\nEBR {ebr_id}: {wyck}  {symm}  {orb}")
            last_ebr = ebr_id
        print(f"   {kpt}: {irrep}  (mult={mult})")

if __name__ == "__main__":
    conn = sqlite3.connect("tqc.db")

    print_space_groups(conn)

    print_ebrs_for_sg(conn, 2)

    print_irreps_for_ebr(conn, 17)

    dump_all_irreps_for_sg(conn, 2)

    conn.close()