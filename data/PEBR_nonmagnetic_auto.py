import argparse
import random
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from PEBR_TR_nonmagnetic_query import EBRDatabaseManager

PARSER = "html.parser"
BASE_URL = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/"

class BCSRequestsScraper:
    def __init__(self, db: EBRDatabaseManager):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        })

    def fetch_main(self, sg_number: int) -> str:
        """Fetches the main page for a non-magnetic space group."""
        url = urljoin(BASE_URL, "bandrep.pl")
        payload = {
            "super": str(sg_number),
            "elementaryTR": "Elementary TR",
            "nomaximal": "yes",
        }
        r = self.session.post(url, data=payload, timeout=60)
        r.raise_for_status()
        return r.text

    def parse_main(self, html: str, sg_number: int) -> list[dict]:
        """
        Parses the main EBR table with enhanced debugging.
        """
        soup = BeautifulSoup(html, PARSER)
        
        # Look for the main table with band representation data
        main_table = None
        tables = soup.find_all("table")
        print(f"  DEBUG: Found {len(tables)} total tables")
        
        for i, table in enumerate(tables):
            # Look for table with frame="box" first
            if table.get("frame") == "box":
                main_table = table
                print(f"  DEBUG: Using table {i} with frame='box'")
                break
        
        if not main_table:
            # Look for any table containing Wyckoff
            for i, table in enumerate(tables):
                if table.find(string=lambda s: s and "Wyckoff" in s):
                    main_table = table
                    print(f"  DEBUG: Using table {i} containing 'Wyckoff'")
                    break
        
        if not main_table:
            print(f"  DEBUG: No suitable table found. Saving HTML for inspection...")
            debug_filename = f"debug_sg_{sg_number}_no_table.html"
            with open(debug_filename, "w", encoding="utf-8") as f: 
                f.write(html)
            print(f"  DEBUG: HTML saved to '{debug_filename}'")
            return []

        all_rows = main_table.find_all("tr")
        print(f"  DEBUG: Found {len(all_rows)} rows in main table")

        # Check if this is just a header-only table (empty result)
        non_empty_rows = []
        for row in all_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) > 1:  # More than just a header cell
                non_empty_rows.append(row)
        
        if not non_empty_rows:
            print(f"  DEBUG: All rows have only 1 cell - this is an empty table")
            return []
        
        # Enhanced row analysis
        header_data = {}
        k_vector_data = []

        for i, row in enumerate(all_rows):
            cells = row.find_all(['td', 'th'])
            if not cells: 
                continue
            
            first_cell_text = cells[0].get_text(strip=True)
            print(f"  DEBUG: Row {i}: '{first_cell_text}' ({len(cells)} cells)")
            
            # Print first few cells for debugging
            if len(cells) > 1:
                cell_preview = [cell.get_text(strip=True)[:20] + "..." if len(cell.get_text(strip=True)) > 20 else cell.get_text(strip=True) for cell in cells[1:4]]
                print(f"    Cell preview: {cell_preview}")
            
            if len(cells) == 1:
                # This is likely a header row with only a label
                continue
                
            if ":" in first_cell_text:
                k_vector_data.append(cells)
                print(f"    -> K-vector row")
            else:
                # Header row with actual data
                key = first_cell_text.lower().replace('.', '').replace('/', '').replace('-', '').replace(' ', '').replace('\\', '')
                if key: 
                    header_data[key] = cells
                    print(f"    -> Header row: '{key}' with {len(cells)} cells")

        print(f"  DEBUG: Header keys found: {list(header_data.keys())}")
        print(f"  DEBUG: K-vector rows: {len(k_vector_data)}")

        if not header_data:
            print(f"  DEBUG: No header data found - saving HTML for inspection")
            debug_filename = f"debug_sg_{sg_number}_no_headers.html"
            with open(debug_filename, "w", encoding="utf-8") as f: 
                f.write(html)
            print(f"  DEBUG: HTML saved to '{debug_filename}'")
            return []

        # Enhanced header analysis
        wyckoff_cells = header_data.get("wyckoffpos", header_data.get("wyckoff", []))
        bandrep_cells = header_data.get("bandrep", header_data.get("bandreps", header_data.get("bandrepresentation", header_data.get("bandrepresentations", []))))
        
        print(f"  DEBUG: Wyckoff cells found: {len(wyckoff_cells)}")
        print(f"  DEBUG: BandRep cells found: {len(bandrep_cells)}")
        
        if not wyckoff_cells and not bandrep_cells:
            print(f"  DEBUG: No Wyckoff or BandRep cells found. Available headers:")
            for key, cells in header_data.items():
                print(f"    '{key}': {len(cells)} cells")
            
            # Try alternate header names
            alternate_keys = ['wyckoffpositions', 'wyckoffposition', 'bandrepresentation', 'bandrep']
            for alt_key in alternate_keys:
                if alt_key in header_data:
                    print(f"  DEBUG: Found alternate key '{alt_key}'")
                    if 'wyckoff' in alt_key:
                        wyckoff_cells = header_data[alt_key]
                    elif 'band' in alt_key:
                        bandrep_cells = header_data[alt_key]
        
        # Detailed content analysis
        max_header_cols = max(len(wyckoff_cells), len(bandrep_cells), 1)
        print(f"  DEBUG: Max header columns: {max_header_cols}")
        
        # Count actual EBR columns with content
        valid_columns = []
        for i in range(1, max_header_cols):  # Start from 1 to skip the label column
            wyckoff_text = wyckoff_cells[i].get_text(strip=True) if i < len(wyckoff_cells) else ""
            bandrep_text = bandrep_cells[i].get_text(strip=True) if i < len(bandrep_cells) else ""
            
            print(f"  DEBUG: Column {i}: Wyckoff='{wyckoff_text}', BandRep='{bandrep_text}'")
            
            # Only count columns that have actual content
            if wyckoff_text or bandrep_text:
                valid_columns.append(i)
        
        print(f"  DEBUG: Valid column indices: {valid_columns}")
        
        if not valid_columns:
            print(f"  DEBUG: No valid EBR columns found - saving HTML for inspection")
            debug_filename = f"debug_sg_{sg_number}_no_valid_columns.html"
            with open(debug_filename, "w", encoding="utf-8") as f: 
                f.write(html)
            print(f"  DEBUG: HTML saved to '{debug_filename}'")
            return []
            
        ebr_data_list = []
        decomp_cells = header_data.get("decomposableindecomposable", header_data.get("decomposable", []))
        
        # Create EBR objects only for valid columns
        for col_idx in valid_columns:
            wyckoff_text = wyckoff_cells[col_idx].get_text(strip=True) if col_idx < len(wyckoff_cells) else ""
            bandrep_text = bandrep_cells[col_idx].get_text(strip=True) if col_idx < len(bandrep_cells) else ""
            
            ebr_obj = {
                "kpoints": {},
                "wyckoff": wyckoff_text,
                "bandrep": bandrep_text
            }
            
            print(f"    Creating EBR {len(ebr_data_list) + 1}: Wyckoff='{wyckoff_text}', BandRep='{bandrep_text}'")

            if col_idx < len(decomp_cells):
                form = decomp_cells[col_idx].find("form")
                ebr_obj["decomposability"] = {"type": "decomposable", "payload": {inp.get("name"): inp.get("value") for inp in form.find_all("input") if inp.get("name")}} if form else {"type": "indecomposable"}
            else:
                ebr_obj["decomposability"] = {"type": "indecomposable"}
                
            ebr_data_list.append(ebr_obj)

        print(f"  DEBUG: Created {len(ebr_data_list)} EBR objects")
        print(f"  DEBUG: Processing {len(k_vector_data)} k-vector rows")
        
        # Populate k-point data
        for k_row_cells in k_vector_data:
            k_label = k_row_cells[0].get_text(strip=True).split(":")[0]
            print(f"    K-point '{k_label}': {len(k_row_cells)} cells")
            
            # Map valid columns to EBR indices
            for ebr_idx, col_idx in enumerate(valid_columns):
                if col_idx < len(k_row_cells) and ebr_idx < len(ebr_data_list):
                    irrep_text = k_row_cells[col_idx].get_text(strip=True)
                    ebr_data_list[ebr_idx]["kpoints"][k_label] = irrep_text
                    print(f"      EBR {ebr_idx + 1} (col {col_idx}): '{irrep_text}'")

        print(f"  DEBUG: Final EBR count: {len(ebr_data_list)}")
        
        # Final validation
        if not ebr_data_list:
            print(f"  DEBUG: No EBRs created - saving HTML for inspection")
            debug_filename = f"debug_sg_{sg_number}_no_ebrs_created.html"
            with open(debug_filename, "w", encoding="utf-8") as f: 
                f.write(html)
            print(f"  DEBUG: HTML saved to '{debug_filename}'")
        
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

        return None 

    def process_space_group(self, sg_number: int) -> bool:
        """Processes one space group with the robust architecture."""
        print(f"\n--- Processing SG {sg_number} ---")
        try:
            html = self.fetch_main(sg_number)
            ebr_data_list = self.parse_main(html, sg_number)
        except RuntimeError as e:
            if "No main table found" in str(e):
                print(f"✅ SG {sg_number}: Valid 'No Data' page found. Skipping.")
                self.db.log_processed_space_group(sg_number)
                return True
            print(f"❌ SG {sg_number}: An unexpected runtime error occurred: {e}")
            return False
        except Exception as e:
            print(f"❌ SG {sg_number}: fetch/parse failed: {e}")
            return False

        if not ebr_data_list:
            print(f"⚠️  SG {sg_number}: Page had a table, but parsing found no EBR data.")
            # Save debug file to investigate
            debug_filename = f"debug_sg_{sg_number}_no_ebr.html"
            with open(debug_filename, "w", encoding="utf-8") as f: 
                f.write(html)
            print(f"  Debug HTML saved to '{debug_filename}'")
            self.db.log_processed_space_group(sg_number)
            return True

        sg_id = self.db.get_or_create_space_group_id(sg_number)
        self.db.delete_ebrs_for_sg(sg_id)

        for ebr_data in ebr_data_list:
            try:
                ebr_id = self.db.insert_single_ebr(sg_id, ebr_data)
                print(f"  ↳ Inserted EBR '{ebr_data.get('bandrep', 'N/A')}' with ID {ebr_id}.")
                if ebr_data.get("decomposability", {}).get("type") == "decomposable":
                    payload = ebr_data["decomposability"]["payload"]
                    decomp_result = self.fetch_decomposition(payload)
                    if not decomp_result:
                        print(f"   ↳❌ WARNING: Failed to parse decomposition page for EBR {ebr_id}.")
                        continue
                    if decomp_result["type"] == "branches":
                        for d_idx, b_list in enumerate(decomp_result["data"], 1):
                            for b_idx, i_str in enumerate(b_list, 1):
                                self.db.add_decomposition_item(sg_id, ebr_id, d_idx, b_idx, i_str)
                    elif decomp_result["type"] == "ebr_list":
                        self.db.add_ebr_list_decomposition(sg_id, ebr_id, decomp_result["data"])
            except Exception as e:
                print(f"❌ An error occurred processing an EBR for SG {sg_number}: {e}")
                import traceback
                traceback.print_exc()
        return True

    def run(self, start: int, end: int):
        failed_sgs = []
        for sg in range(start, end + 1):
            if not self.process_space_group(sg):
                failed_sgs.append(sg)
            time.sleep(random.uniform(1.0, 2.5))
        print(f"\nDone. Failed SGs: {failed_sgs}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fast Non-Magnetic BandRep Scraper")
    p.add_argument("--start", type=int, required=True, help="first SG to scrape")
    p.add_argument("--end",   type=int, required=True, help="last SG to scrape")
    args = p.parse_args()

    db = EBRDatabaseManager()
    scraper = BCSRequestsScraper(db)
    scraper.run(args.start, args.end)
    db.close()