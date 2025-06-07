import argparse
import random
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PARSER = "html.parser"
from PEBR_TR_nonmagnetic_query import EBRDatabaseManager

BASE_URL = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/"


class BCSRequestsScraper:
    def __init__(self, db: EBRDatabaseManager):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/108.0.0.0 Safari/537.36"
            )
        })

    def fetch_main(self, sg: int) -> str:
        """GET the form page, then POST the SG number, return HTML."""
        url = urljoin(BASE_URL, "bandrep.pl")
        payload = {
            "super": str(sg), "elementaryTR": "Elementary TR", "nomaximal": "yes",
            "elementary": "", "wyck": "", "wyckTR": "", "wyckoff": ""
        }
        r = self.session.post(url, data=payload, timeout=60)
        r.raise_for_status()
        return r.text

    def parse_main(self, html: str, sg: int) -> dict:
        """Parse the main EBR table."""
        soup = BeautifulSoup(html, PARSER)

        # --- START: FIX 1 ---
        # Robustly find the table by checking all tables on the page.
        # This avoids the ambiguous find() call that caused the TypeError.
        main_table = None
        for table in soup.find_all("table"):
            if table.find("td", string=lambda s: s and "Wyckoff pos" in s):
                main_table = table
                break

        if not main_table:
            raise RuntimeError(f"No main table found for SG {sg}")

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

    def process_space_group(self, sg: int) -> bool:
        """Scrape SG, ingest main EBRs, then any decompositions."""
        try:
            html = self.fetch_main(sg)
            data = self.parse_main(html, sg)
        except Exception as e:
            print(f"❌ SG {sg}: fetch/parse failed: {e}")
            return False

        # Fetch decompositions for columns that had a form
        decomps = []
        for form_payload in data["decomp_forms"]:
            if form_payload:
                try:
                    # Pass the payload instead of just a URL
                    decomps.append(self.fetch_decomposition(form_payload))
                except Exception as e:
                    print(f"   ↳ branch fetch failed for SG {sg}: {e}")
                    decomps.append(None)
            else:
                decomps.append(None)
        data["decompositions"] = decomps

        # Ingest main EBR data
        notes = ["decomposable" if form else "indecomposable" for form in data["decomp_forms"]]
        kpoint_list = list(data["kpoints"].items())
        try:
            inserted, sg_id = self.db._insert_data(
                sg, data["wyckoff"], data["bandrep"], notes, kpoint_list
            )
            print(f"✅ SG {sg}: inserted {len(inserted)} EBRs")
        except Exception as e:
            print(f"❌ SG {sg}: DB insertion failed: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for DB errors
            return False

        # Ingest the decomposition branches
        for idx, info in enumerate(inserted):
            if info.get("note") == "decomposable":
                branches = data["decompositions"][idx]
                if not branches:
                    print(f"   ↳ WARNING: EBR {info['ebr_id']} marked decomposable but no branches found/scraped.")
                    continue
                print(f"   ↳ Ingesting {len(branches)} branches for EBR {info['ebr_id']}...")
                for bi, branch_row in enumerate(branches, 1):
                    try:
                        # Assuming all decompositions have exactly two branches
                        self.db.add_ebr_decomposition_branch(
                            sg_id, info["ebr_id"], bi, branch_row[0], branch_row[1]
                        )
                    except Exception as e:
                        print(f"   ↳❌ failed to add branch {bi} for EBR {info['ebr_id']}: {e}")
        return True

    def run(self, start: int, end: int):
        failed = []
        for sg in range(start, end + 1):
            print(f"\n--- Processing SG {sg} ---")
            if not self.process_space_group(sg):
                failed.append(sg)
            time.sleep(random.uniform(1.0, 2.5)) # Be polite to the server
        print(f"\nDone. Failed SGs: {failed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fast BandRep scraper using Requests and BeautifulSoup"
    )
    p.add_argument("--start", type=int, required=True, help="first SG to scrape")
    p.add_argument("--end",   type=int, required=True, help="last SG to scrape")
    args = p.parse_args()

    db = EBRDatabaseManager()
    scraper = BCSRequestsScraper(db)
    scraper.run(args.start, args.end)
    db.close()