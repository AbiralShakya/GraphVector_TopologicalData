import argparse
import random
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PARSER = "html.parser"
from EBR_magnetic_query import MagneticEBRDatabaseManager

BASE_URL = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/"

import re

class BNSNumberProvider:
    """
    Parses the magnetic space group file and provides the next BNS number
    to be scraped.
    """
    def __init__(self, filepath: str):
        """
        Initializes the provider by parsing the specified file.
        """
        self.all_bns_numbers = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    # Find lines containing "BNS:" and extract the number
                    match = re.search(r"BNS:\s*([\d\.]+)", line)
                    if match:
                        self.all_bns_numbers.append(match.group(1))
            print(f"✅ BNS Provider initialized with {len(self.all_bns_numbers)} magnetic space groups.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Magnetic space group file not found at: {filepath}")

    def get_next_bns_number(self, db_manager):
        """
        Compares all BNS numbers against those already in the database
        and returns the next one that needs to be processed.
        """
        scraped_bns_numbers = db_manager.get_all_scraped_bns_numbers()
        for bns_num in self.all_bns_numbers:
            if bns_num not in scraped_bns_numbers:
                return bns_num
        return None # All space groups have been scraped

class MagneticBCSRequestsScraper:
    def __init__(self, db: MagneticEBRDatabaseManager):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/108.0.0.0 Safari/537.36"
            )
        })

    def fetch_main(self, bns_number: str) -> str:
        """
        Fetches the main page for a given Magnetic (BNS) space group using the
        correct, minimal payload.
        """
        url = urljoin(BASE_URL, "mbandrep.pl")
        try:
            # A GET request first is still good practice to establish a session
            self.session.get(url, timeout=30)
        
            # This is the correct, minimal payload for the magnetic program.
            payload = {
                "super": bns_number,
                "elementary": "Elementary",
            }

            r_post = self.session.post(url, data=payload, timeout=60)
            r_post.raise_for_status()
            return r_post.text
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch main page for BNS {bns_number}: {e}") from e

    def parse_main(self, html: str, sg_number: int) -> list[dict]:
        """
        Parses the main EBR table using robust column counting and ensures
        all data objects have a consistent structure.
        """
        soup = BeautifulSoup(html, PARSER)
        # Using frame="box" is a more direct way to find the main table
        main_table = soup.find("table", frame="box")

        if not main_table or not main_table.find(string=lambda s: s and "Wyckoff pos" in s):
            debug_filename = f"debug_sg_{sg_number}.html"
            with open(debug_filename, "w", encoding="utf-8") as f:
                f.write(html)
            raise RuntimeError(
                f"No main table found for SG {sg_number}. "
                f"HTML response saved to '{debug_filename}' for inspection."
            )

        all_rows = main_table.find_all("tr")
        header_data = {}
        k_vector_data = []

        # Separate header rows from k-vector rows
        for row in all_rows:
            cells = row.find_all(['td', 'th'])
            if not cells: continue
            first_cell_text = cells[0].get_text(strip=True)
            if ":" in first_cell_text:
                k_vector_data.append(cells)
            else:
                # Clean up the key for easier access
                header_key = first_cell_text.lower().replace('.', '').replace('/', '')
                if header_key: header_data[header_key] = cells

        # Use your robust method to count the total number of columns
        max_cols = 0
        for key in header_data:
            max_cols = max(max_cols, len(header_data[key]))
        for row in k_vector_data:
            max_cols = max(max_cols, len(row))
        
        num_ebr_columns = max_cols - 1
        if num_ebr_columns <= 0:
            return []
            
        ebr_data_list = []
        
        # Pass 1: Initialize EBR objects from the header rows
        for i in range(1, num_ebr_columns + 1):  # i is the 1-based column index
            ebr_obj = {"kpoints": {}}
            
            wyckoff_cells = header_data.get("wyckoff pos", [])
            bandrep_cells = header_data.get("band-rep", [])
            decomp_cells = header_data.get("decomposableindecomposable", [])

            # --- START: THE FIX ---
            # Ensure 'wyckoff' and 'bandrep' keys are always created,
            # even if their value is an empty string.
            if i < len(wyckoff_cells):
                ebr_obj["wyckoff"] = wyckoff_cells[i].get_text(strip=True)
            else:
                ebr_obj["wyckoff"] = ""  # Add key with empty string

            if i < len(bandrep_cells):
                ebr_obj["bandrep"] = bandrep_cells[i].get_text(strip=True)
            else:
                ebr_obj["bandrep"] = ""  # Add key with empty string
            # --- END: THE FIX ---

            if i < len(decomp_cells):
                form = decomp_cells[i].find("form")
                ebr_obj["decomposability"] = {"type": "decomposable", "payload": {inp.get("name"): inp.get("value") for inp in form.find_all("input") if inp.get("name")}} if form else {"type": "indecomposable"}
            else:
                ebr_obj["decomposability"] = {"type": "indecomposable"}
                
            ebr_data_list.append(ebr_obj)

        # Pass 2: Populate k-point data
        for k_row_cells in k_vector_data:
            k_label = k_row_cells[0].get_text(strip=True).split(":")[0]
            for i in range(1, len(k_row_cells)):
                ebr_index = i - 1
                if ebr_index < len(ebr_data_list):
                    ebr_data_list[ebr_index]["kpoints"][k_label] = k_row_cells[i].get_text(strip=True)

        return ebr_data_list


    def fetch_decomposition(self, form_payload: dict) -> dict | None:
        """
        [UPDATED] Fetches a decomposition page and intelligently determines if it's a
        branch table or an EBR list.
        """
        url = urljoin(BASE_URL, "bandrepdesc.pl") # This might need to be mbandrepdesc.pl
        r = self.session.post(url, data=form_payload, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, PARSER)

        # First, look for a branch table (our original logic)
        branch_table = None
        for table in soup.find_all("table"):
            if table.find(['td', 'th'], string=lambda s: s and "branch" in s.lower()):
                branch_table = table
                break
        
        if branch_table:
            branches = []
            for row in branch_table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")[1:]]
                if cells: branches.append(cells)
            return {"type": "branches", "data": branches}

        # If no branch table, look for an EBR list.
        # Based on your image, these are just listed. We find them by looking for the ↑G symbol.
        ebr_list = [tag.get_text(strip=True) for tag in soup.find_all(string=lambda s: s and "↑G" in s)]
        if ebr_list:
            return {"type": "ebr_list", "data": ebr_list}

        return None # Failed to parse either type

    def process_space_group(self, bns_number: str) -> bool:
        """
        [REWRITTEN] Processes a single space group using the new column-aware architecture.
        """
        print(f"\n--- Processing Magnetic SG (BNS) {bns_number} ---")
        try:
            html = self.fetch_main(bns_number)
            # `ebr_data_list` is a list of objects, one for each column.
            ebr_data_list = self.parse_main(html, bns_number)
        except Exception as e:
            print(f"❌ BNS {bns_number}: fetch/parse failed: {e}")
            return False

        if not ebr_data_list:
            print(f"ℹ️ No EBR data found for BNS {bns_number}.")
            return True

        # Now, iterate through each EBR we found and process it
        for ebr_data in ebr_data_list:
            try:
                # Step 1: Insert the main EBR data and get its new ID
                ebr_id = self.db.insert_single_ebr(bns_number, ebr_data)
                print(f"✅ Inserted EBR '{ebr_data['bandrep']}' with ID {ebr_id} for BNS {bns_number}.")

                # Step 2: Handle decompositions if they exist
                if ebr_data.get("decomposability", {}).get("type") == "decomposable":
                    payload = ebr_data["decomposability"]["payload"]
                    print(f"   ↳ Fetching decomposition for EBR {ebr_id}...")
                    decomp_result = self.fetch_decomposition(payload)

                    if not decomp_result:
                        print(f"   ↳❌ WARNING: Failed to parse decomposition page for EBR {ebr_id}.")
                        continue
                    
                    # Check which type of decomposition we got and call the right DB function
                    if decomp_result["type"] == "branches":
                        print(f"   ↳ Ingesting {len(decomp_result['data'])} branch set(s)...")
                        for decomp_idx, branch_list in enumerate(decomp_result["data"], 1):
                            for branch_idx, irrep_string in enumerate(branch_list, 1):
                                self.db.add_decomposition_item(bns_number, ebr_id, decomp_idx, branch_idx, irrep_string)

                    elif decomp_result["type"] == "ebr_list":
                        print(f"   ↳ Ingesting {len(decomp_result['data'])} EBR list decomposition...")
                        self.db.add_ebr_list_decomposition(bns_number, ebr_id, decomp_result["data"])

            except Exception as e:
                print(f"❌ An error occurred processing an EBR for BNS {bns_number}: {e}")
                import traceback
                traceback.print_exc()

        return True
    
    def insert_single_ebr(self, bns_number, ebr_data):
        """Inserts one EBR and its associated k-point irreps, returning the new ebr_id."""
        from PEBR_TR_nonmagnetic_query import EBRDatabaseManager
        parser_helpers = EBRDatabaseManager()

        # Ensure the space group exists
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO magnetic_space_groups(bns_number) VALUES (?)", (bns_number,))

        # Insert the main EBR record
        wyck_letter, site_sym = parser_helpers.parse_wyckoff(ebr_data["wyckoff"])
        orb_label, orb_mult = parser_helpers.parse_orbital(ebr_data["bandrep"])
        note = ebr_data.get("decomposability", {}).get("type", "indecomposable")

        cursor.execute("""
            INSERT INTO magnetic_ebrs (bns_number, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (bns_number, wyck_letter, site_sym, orb_label, orb_mult, note))
        ebr_id = cursor.lastrowid

        # Insert all the k-point irreps for this EBR
        for k_point, irrep_str in ebr_data["kpoints"].items():
            mult = parser_helpers.parse_multiplicity(irrep_str)
            label_no_mult = parser_helpers.parse_irrep_label(irrep_str)
            cursor.execute("""
                INSERT INTO magnetic_irreps (bns_number, ebr_id, k_point, irrep_label, multiplicity)
                VALUES (?, ?, ?, ?, ?)
            """, (bns_number, ebr_id, k_point, label_no_mult, mult))
        
        self.conn.commit()
        parser_helpers.close()
        return ebr_id

    def add_ebr_list_decomposition(self, bns_number, parent_ebr_id, ebr_list):
        """Adds rows for an EBR that decomposes into other EBRs."""
        cursor = self.conn.cursor()
        for ebr_string in ebr_list:
            cursor.execute("""
                INSERT INTO magnetic_ebr_decomposition_ebrs (bns_number, parent_ebr_id, decomposes_into)
                VALUES (?, ?, ?)
            """, (bns_number, parent_ebr_id, ebr_string))
        self.conn.commit()

    def run(self, bns_provider: BNSNumberProvider):
        failed_bns = []
        while True:
            next_bns = bns_provider.get_next_bns_number(self.db)
            if next_bns is None:
                print("\n🎉 All magnetic space groups have been processed!")
                break
            
            if not self.process_space_group(next_bns):
                failed_bns.append(next_bns)

            time.sleep(random.uniform(1.0, 2.0))

        print(f"\nDone. Failed BNS Numbers: {failed_bns}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bilbao Crystallographic Server Scraper")
    p.add_argument("--magnetic", action="store_true", help="Scrape magnetic band co-reps.")
    args = p.parse_args()
    print("--- Starting MAGNETIC Band Co-Representation Scraper ---")
    db = MagneticEBRDatabaseManager('ebr_magnetic.db')
    provider = BNSNumberProvider('/Users/abiralshakya/Documents/Research/GraphVectorTopological/magnetic_table_bns.txt')
    scraper = MagneticBCSRequestsScraper(db)
    scraper.run(provider)
    db.close()