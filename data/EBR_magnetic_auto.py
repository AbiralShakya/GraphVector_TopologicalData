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
        """Fetches the main page for a given Magnetic (BNS) space group."""
        url = urljoin(BASE_URL, "mbandrep.pl")
        try:
            self.session.get(url, timeout=30)
        
            payload = {
                "super": bns_number,
                "elementary": "Elementary", 
            }
            r_post = self.session.post(url, data=payload, timeout=60)
            r_post.raise_for_status()
            return r_post.text
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch main page for BNS {bns_number}: {e}") from e

    def parse_main(self, html: str, bns_number: str) -> dict:
        """Parse the main EBR table."""
        soup = BeautifulSoup(html, PARSER)
        main_table = None
        for table in soup.find_all("table"):
            if table.find("td", string=lambda s: s and "Wyckoff pos" in s):
                main_table = table
                break

        if not main_table:
            raise RuntimeError(f"No main table found for SG {bns_number}")

        data = {
            "wyckoff": [], "bandrep": [], "kpoints": {}, "decomp_forms": []
        }

        for tr in main_table.find_all("tr"):
            cells = tr.find_all(['td', 'th'])
            if not cells: continue

            first_cell_text = cells[0].get_text(strip=True)

            if first_cell_text.lower().startswith("wyckoff"):
                data["wyckoff"] = [td.get_text(strip=True) for td in cells[1:]]
            elif first_cell_text.lower().startswith("band-rep"):
                data["bandrep"] = [td.get_text(strip=True) for td in cells[1:]]
            elif ":" in first_cell_text:
                label = first_cell_text.split(":")[0]
                data["kpoints"][label] = [td.get_text(strip=True) for td in cells[1:]]
            elif "Decomposable" in first_cell_text or "Indecomposable" in first_cell_text:
                for td in cells[1:]:
                    form = td.find("form", action=lambda a: a and "bandrepdesc.pl" in a)
                    if form:
                        # Extract the hidden input values needed for the POST request
                        form_payload = {
                            inp.get("name"): inp.get("value")
                            for inp in form.find_all("input")
                            if inp.get("name")
                        }
                        data["decomp_forms"].append(form_payload)
                    else:
                        data["decomp_forms"].append(None)
        return data

    def fetch_decomposition(self, form_payload: dict) -> list[list[str]] | None:
        """Fetch a bandrepdesc.pl page using its form data and parse its branches."""
        url = urljoin(BASE_URL, "bandrepdesc.pl")
        r = self.session.post(url, data=form_payload, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, PARSER)

        # Robustly find the decomposition table.
        branch_table = None
        for table in soup.find_all("table"):
            # Check for header cells containing "branch"
            if table.find(['td', 'th'], string=lambda s: s and "branch" in s.lower()):
                branch_table = table
                break

        if not branch_table:
            return None

        branches = []
        # Start from the second row to skip the header
        for row in branch_table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")[1:]]
            if cells: # Only add if there are actual branch cells
                branches.append(cells)
        return branches

    def process_space_group(self, bns_number: int) -> bool:
        """Scrape SG, ingest main EBRs, then any decompositions."""
        print(f"\n--- Processing Magnetic SG (BNS) {bns_number} ---")

        try:
            html = self.fetch_main(bns_number)
            data = self.parse_main(html, bns_number)
        except Exception as e:
            print(f"❌ SG {bns_number}: fetch/parse failed: {e}")
            return False

        # Fetch decompositions for columns that had a form
        decomps = []
        for form_payload in data["decomp_forms"]:
            if form_payload:
                try:
                    # Pass the payload instead of just a URL
                    decomps.append(self.fetch_decomposition(form_payload))
                except Exception as e:
                    print(f"   ↳ branch fetch failed for SG {bns_number}: {e}")
                    decomps.append(None)
            else:
                decomps.append(None)
        data["decompositions"] = decomps

        # Ingest main EBR data
        notes = ["decomposable" if form else "indecomposable" for form in data["decomp_forms"]]
        kpoint_list = list(data["kpoints"].items())
        try:
            inserted, sg_id = self.db._insert_data(
                bns_number, data["wyckoff"], data["bandrep"], notes, kpoint_list
            )
            print(f"✅ BNS {bns_number}: inserted {len(inserted)} EBRs")
        except Exception as e:
            print(f"❌ BNS {bns_number}: DB insertion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Ingest the decomposition branches
        for idx, info in enumerate(inserted):
            if info.get("note") == "decomposable":
                # 'branches' is a list of lists, e.g., [[b1, b2], [b1, b2, b3]]
                decompositions = data["decompositions"][idx]
                if not decompositions:
                    continue
                
                print(f"   ↳ Ingesting {len(decompositions)} decomposition set(s) for EBR {info['ebr_id']}...")

                # 'decomposition_index' corresponds to the row number on the decomposition page (1, 2, ...)
                for decomp_idx, branch_list in enumerate(decompositions, 1):
                    
                    # 'branch_list' is the list of actual branch strings for this one decomposition
                    # e.g., ['M5M6...', 'M5M6...'] or ['branch1', 'branch2', 'branch3']
                    print(f"     ↳ Set {decomp_idx} has {len(branch_list)} branches.")

                    # 'branch_index' corresponds to the column number (branch 1, branch 2, ...)
                    for branch_idx, irrep_string in enumerate(branch_list, 1):
                        try:
                            # Call the new, flexible insertion function for each branch
                            self.db.add_decomposition_item(
                                sg_id, info["ebr_id"], decomp_idx, branch_idx, irrep_string
                            )
                        except Exception as e:
                            print(f"   ↳❌ failed to add item for decomp {decomp_idx}, branch {branch_idx}: {e}")
        return True

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