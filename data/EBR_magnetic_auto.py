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

    # In your MagneticBCSRequestsScraper class, replace parse_main with this:

    # In your MagneticBCSRequestsScraper class

    def parse_main(self, html: str, bns_number: str) -> list[dict]:
        """
        [FINAL CORRECTED VERSION] Parses the main EBR table using a robust
        column counting method to ensure all data is captured.
        """
        soup = BeautifulSoup(html, "html.parser")
        main_table = None
        for table in soup.find_all("table"):
            if table.find(string=lambda s: s and "Wyckoff pos" in s):
                main_table = table
                break

        if not main_table:
            debug_filename = f"debug_bns_{bns_number}.html"
            with open(debug_filename, "w", encoding="utf-8") as f:
                f.write(html)
            raise RuntimeError(
                f"No main table found for BNS {bns_number}. "
                f"HTML response saved to '{debug_filename}' for inspection."
            )

        all_rows = main_table.find_all("tr")
        if not all_rows: return []

        header_rows = {}
        k_vector_rows_start_index = -1

        for i, tr in enumerate(all_rows):
            cells = tr.find_all(['td', 'th'])
            if not cells: continue
            
            text = cells[0].get_text(strip=True).lower()
            if "wyckoff pos" in text:
                header_rows["wyckoff"] = cells
            elif "band-rep" in text:
                header_rows["bandrep"] = cells
            elif "decomposable" in text:
                header_rows["decomposability"] = cells
            elif ":" in cells[0].get_text(strip=True):
                k_vector_rows_start_index = i
                break
                
        if not all(k in header_rows for k in ["wyckoff", "bandrep"]):
            raise ValueError(f"Could not find all header rows for BNS {bns_number}")

        # FIXED: Calculate the number of EBR columns by finding the maximum number of 
        # actual data columns (excluding the first column which contains labels)
        # Also check k-point rows to ensure we don't miss any columns
        max_cols_from_headers = max(
            len(header_rows.get("wyckoff", [])),
            len(header_rows.get("bandrep", [])),
            len(header_rows.get("decomposability", []))
        )
        
        # Check k-point rows to get the actual number of columns
        max_cols_from_kpoints = 0
        if k_vector_rows_start_index != -1:
            for i in range(k_vector_rows_start_index, len(all_rows)):
                k_vector_row_cells = all_rows[i].find_all(['td', 'th'])
                if k_vector_row_cells:
                    max_cols_from_kpoints = max(max_cols_from_kpoints, len(k_vector_row_cells))
        
        # Take the maximum from both header and k-point rows, then subtract 1 for the label column
        num_ebr_columns = max(max_cols_from_headers, max_cols_from_kpoints) - 1
        
        print(f"Debug: Found {num_ebr_columns} EBR columns for BNS {bns_number}")
        print(f"Debug: Header cols = {max_cols_from_headers}, K-point cols = {max_cols_from_kpoints}")

        if num_ebr_columns <= 0:
            return []
            
        ebr_data_list = []

        # Pass 1: Initialize EBR objects from the header rows
        for i in range(1, num_ebr_columns + 1): # i is 1-based column index
            ebr_obj = {"kpoints": {}}
            
            if i < len(header_rows["wyckoff"]):
                ebr_obj["wyckoff"] = header_rows["wyckoff"][i].get_text(strip=True)
            else:
                ebr_obj["wyckoff"] = ""  # Handle missing data gracefully
                
            if i < len(header_rows["bandrep"]):
                ebr_obj["bandrep"] = header_rows["bandrep"][i].get_text(strip=True)
            else:
                ebr_obj["bandrep"] = ""  # Handle missing data gracefully
            
            decomposability_cells = header_rows.get("decomposability", [])
            if i < len(decomposability_cells):
                form = decomposability_cells[i].find("form")
                if form:
                    ebr_obj["decomposability"] = {
                        "type": "decomposable",
                        "payload": {inp.get("name"): inp.get("value") for inp in form.find_all("input") if inp.get("name")}
                    }
                else:
                    ebr_obj["decomposability"] = {"type": "indecomposable"}
            else:
                ebr_obj["decomposability"] = {"type": "indecomposable"}

            ebr_data_list.append(ebr_obj)

        # Pass 2: Populate k-point data for each EBR object
        if k_vector_rows_start_index != -1:
            for i in range(k_vector_rows_start_index, len(all_rows)):
                k_vector_row_cells = all_rows[i].find_all(['td', 'th'])
                if not k_vector_row_cells: continue
                
                k_point_label_full = k_vector_row_cells[0].get_text(strip=True)
                if ":" not in k_point_label_full: continue
                
                k_point_label = k_point_label_full.split(":")[0]

                # FIXED: Make sure we process ALL available columns
                for j in range(min(num_ebr_columns, len(k_vector_row_cells) - 1)): # j is 0-based list index
                    column_index_in_html = j + 1
                    if column_index_in_html < len(k_vector_row_cells):
                        irrep_str = k_vector_row_cells[column_index_in_html].get_text(strip=True)
                        if j < len(ebr_data_list):  # Safety check
                            ebr_data_list[j]["kpoints"][k_point_label] = irrep_str

        # Debug output to verify we're capturing all EBRs
        print(f"Debug: Parsed {len(ebr_data_list)} EBRs for BNS {bns_number}")
        for idx, ebr in enumerate(ebr_data_list):
            print(f"  EBR {idx+1}: Wyckoff='{ebr.get('wyckoff', '')}', Band-rep='{ebr.get('bandrep', '')}', K-points={len(ebr.get('kpoints', {}))}")

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