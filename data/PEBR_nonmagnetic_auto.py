import sqlite3
import re
import time
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ==============================================================================
# === PART 1: DATABASE MANAGER
# === Your complete class for managing the SQLite DB, adapted for this script.
# ==============================================================================
def save_table_as_csv(self, sg_number, output_dir="table_data"):
    """Save the extracted table data as a CSV file for inspection."""
    import csv
    import os
    
    table_data = self.extract_table_data_only(sg_number)
    if not table_data:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"sg_{sg_number}_table.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header information
        writer.writerow(['Space Group', sg_number])
        writer.writerow([])  # Empty row
        
        # Write Wyckoff positions
        writer.writerow(['Wyckoff pos.'] + table_data['wyckoff_positions'])
        
        # Write band representations
        writer.writerow(['Band-Rep.'] + table_data['band_representations'])
        
        # Write decomposability
        writer.writerow(['Decomposable'] + table_data['decomposability'])
        
        writer.writerow([])  # Empty row
        
        # Write k-point data
        writer.writerow(['K-point', 'Irreps'])
        for kpoint, irreps in table_data['kpoints'].items():
            writer.writerow([kpoint] + irreps)
    
    print(f"  -> ✅ Table data saved to {csv_path}")
    return True

def parse_kpoint_cells(irreps_txt):
    """Parses a string containing multiple k-point irrep cells into a list of individual cell strings."""
    if not irreps_txt:
        return []
    # Use regex to find all occurrences of irreps, handles complex cases
    # like "2 Γ1Γ2(2)" or "A1(1)⊕A2(1)"
    pattern = re.compile(r'(\d*\s*[A-ZΓa-z0-9+↑↓]+(?:\([^\)]+\))?(?:\s*⊕\s*)?)')
    tokens = pattern.findall(irreps_txt)
    # Clean up tokens
    cleaned_tokens = [t.replace('⊕', '').strip() for t in tokens if t.strip()]
    return cleaned_tokens if cleaned_tokens else irreps_txt.split()

class EBRDatabaseManager:
    """Manages the SQLite database for storing EBR data."""
    def __init__(self, db_path="pebr_tr_nonmagnetic_rev2.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
        print(f"Database Manager initialized. DB located at: {self.db_path}")

    def connect(self):
        """Establishes a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def create_tables(self):
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
          space_group_id INTEGER NOT NULL REFERENCES space_groups(id) ON DELETE CASCADE,
          wyckoff_letter TEXT    NOT NULL,
          site_symmetry  TEXT    NOT NULL,
          orbital_label  TEXT    NOT NULL,
          orbital_multiplicity INTEGER NOT NULL DEFAULT 1,
          time_reversal  BOOLEAN NOT NULL DEFAULT 1,
          notes          TEXT,
          created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        self.conn.commit()

    def parse_wyckoff(self, entry):
        m = re.match(r"^(\d*\w+)\(([^)]+)\)$", entry)
        return (m.group(1), m.group(2)) if m else (entry, "")

    def parse_orbital(self, entry):
        m = re.match(r"^(.+)\((\d+)\)$", entry)
        return (m.group(1), int(m.group(2))) if m else (entry, 1)

    def parse_multiplicity(self, entry):
        m = re.search(r"\((\d+)\)$", entry)
        return int(m.group(1)) if m else None

    def parse_irrep_label(self, entry):
        m = re.match(r"^(.+?)(?:\(\d+\))?$", entry)
        return m.group(1) if m else entry

    def _parse_and_insert_from_text(self, sg_number, raw_text_data):
        """Takes the structured text block and performs parsing and database insertion."""
        lines = [line.strip() for line in raw_text_data.strip().split('\n') if line.strip()]
        
        sections = {}
        current_header = None
        header_pattern = re.compile(r'^(Wyckoff pos\.|Band-Rep\.|Decomposable)')
        
        # This logic correctly groups lines under their respective headers
        for line in lines:
            match = header_pattern.match(line)
            if match:
                current_header = match.group(1)
                sections[current_header] = [line[len(current_header):].strip()]
            elif re.match(r'^[A-ZΓ]+:', line):
                current_header = "kpoints"
                if current_header not in sections: sections[current_header] = []
                sections[current_header].append(line)
            elif current_header and current_header != "kpoints":
                sections[current_header].append(line)

        wyckoff_entries = " ".join(sections.get('Wyckoff pos.', [])).split()
        orbital_entries = " ".join(sections.get('Band-Rep.', [])).split()
        notes_entries = " ".join(sections.get('Decomposable', [])).split()
        kpoint_lines = sections.get('kpoints', [])

        if not wyckoff_entries or not orbital_entries:
            raise ValueError("Failed to parse Wyckoff or Band-Rep lines from scraped text.")

        num_cols = len(wyckoff_entries)

        kpoint_data_final = []
        for kp_line_str in kpoint_lines:
            parts = kp_line_str.split(':', 1)
            kp_label = parts[0].strip()
            irreps_text = parts[1].strip() if len(parts) > 1 else ""
            cells = parse_kpoint_cells(irreps_text)
            if len(cells) != num_cols:
                raise ValueError(f"Mismatch at k-point {kp_label}. Expected {num_cols} irrep columns, found {len(cells)} in '{irreps_text}'.")
            kpoint_data_final.append((kp_label, cells))

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM space_groups WHERE number = ?", (sg_number,))
        sg_row = cursor.fetchone()
        sg_id = sg_row[0] if sg_row else cursor.execute("INSERT INTO space_groups(number) VALUES (?)", (sg_number,)).lastrowid
        
        cursor.execute("DELETE FROM ebrs WHERE space_group_id = ?", (sg_id,))
        
        for j in range(num_cols):
            wyck_letter, site_sym = self.parse_wyckoff(wyckoff_entries[j])
            orb_label, orb_mult = self.parse_orbital(orbital_entries[j])
            note = notes_entries[j] if j < len(notes_entries) else "indecomposable"

            cursor.execute("""
                INSERT INTO ebrs (space_group_id, wyckoff_letter, site_symmetry, orbital_label, orbital_multiplicity, notes, time_reversal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sg_id, wyck_letter, site_sym, orb_label, orb_mult, note, 1))
            ebr_id = cursor.lastrowid
            
            for (kp_label, cells) in kpoint_data_final:
                full_irrep_str = cells[j]
                mult = self.parse_multiplicity(full_irrep_str)
                label_no_mult = self.parse_irrep_label(full_irrep_str)
                cursor.execute("""
                    INSERT INTO irreps (ebr_id, k_point, irrep_label, multiplicity)
                    VALUES (?, ?, ?, ?)
                """, (ebr_id, kp_label, label_no_mult, mult))
        
        self.conn.commit()
        print(f"✅ Successfully ingested {num_cols} EBRs for SG {sg_number} into the database.")
        return True

# ==============================================================================
# === PART 2: WEB SCRAPER - FIXED VERSION
# ==============================================================================

class BCS_Scraper:
    """Scrapes the Bilbao Crystallographic Server for EBR data."""
    def __init__(self, database_manager):
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')  # Uncomment for headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--start-maximized")
        self.service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=self.service, options=options)
        self.db = database_manager
        print("BCS Scraper initialized.")

    def extract_table_data_only(self, sg_number):
        """Alternative method that extracts only the raw table data as a dictionary."""
        print(f"\n--- Extracting table data for Space Group: {sg_number} ---")
        target_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/bandrep.pl"

        try:
            self.driver.get(target_url)
            wait = WebDriverWait(self.driver, 20)
            
            # Input space group and submit
            sg_input_box = wait.until(EC.visibility_of_element_located((By.NAME, 'super')))
            sg_input_box.clear()
            sg_input_box.send_keys(str(sg_number))
            
            elementary_tr_button = wait.until(EC.element_to_be_clickable((By.NAME, 'elementaryTR')))
            elementary_tr_button.click()
            
            # Wait for results
            wait.until(EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Wyckoff pos')]")))
            time.sleep(3)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find the main data table
            data_table = None
            tables = soup.find_all('table')
            
            for table in tables:
                if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
                    data_table = table
                    break
            
            if not data_table:
                print("  -> ❌ ERROR: Could not find the main data table")
                return None
            
            # Extract raw table data as a structured dictionary
            rows = data_table.find_all('tr')
            table_dict = {
                'space_group': sg_number,
                'wyckoff_positions': [],
                'band_representations': [],
                'decomposability': [],
                'kpoints': {}
            }
            
            # Parse each row
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue
                    
                first_cell = cells[0].get_text(strip=True).lower()
                row_data = [cell.get_text(strip=True) for cell in cells[1:]]
                
                if 'wyckoff' in first_cell:
                    table_dict['wyckoff_positions'] = row_data
                elif 'band-rep' in first_cell:
                    table_dict['band_representations'] = row_data
                elif 'decomposable' in first_cell or 'indecomposable' in first_cell:
                    table_dict['decomposability'] = row_data
                elif ':' in cells[0].get_text(strip=True):
                    # This is a k-point row
                    kpoint_label = cells[0].get_text(strip=True)
                    irreps = [cell.get_text(strip=True) for cell in cells[1:]]
                    table_dict['kpoints'][kpoint_label] = irreps
            
            print(f"  -> ✅ Successfully extracted table data for SG {sg_number}")
            print(f"  -> Found {len(table_dict['wyckoff_positions'])} EBR columns")
            print(f"  -> Found {len(table_dict['kpoints'])} k-points")
            
            return table_dict
        
        except Exception as e:
            print(f"  -> ❌ Error extracting table data for SG {sg_number}: {e}")
            return None

    def get_ebr_data_for_sg_improved(self, sg_number):
        """Improved version that uses the structured table extraction."""
        table_data = self.extract_table_data_only(sg_number)
        
        if not table_data:
            return False
        
        # Convert the structured data back to the text format your parser expects
        text_lines = []
        
        if table_data['wyckoff_positions']:
            text_lines.append("Wyckoff pos. " + " ".join(table_data['wyckoff_positions']))
        
        if table_data['band_representations']:
            text_lines.append("Band-Rep. " + " ".join(table_data['band_representations']))
        
        if table_data['decomposability']:
            text_lines.append("Decomposable " + " ".join(table_data['decomposability']))
        
        # Add k-point data
        for kpoint, irreps in table_data['kpoints'].items():
            irreps_text = " ".join(irreps)
            text_lines.append(f"{kpoint}: {irreps_text}")
        
        formatted_text_data = "\n".join(text_lines)
        
        print(f"  -> Formatted data preview:\n{formatted_text_data[:300]}...")
        
        # Use your existing database insertion method
        try:
            self.db._parse_and_insert_from_text(sg_number, formatted_text_data)
            return True
        except Exception as e:
            print(f"  -> ❌ Database insertion failed for SG {sg_number}: {e}")
            return False


    def _parse_html_table_to_text(self, table_soup):
        """Converts the BeautifulSoup table object into a structured text block."""
        if not table_soup: 
            return ""
        
        rows = table_soup.find_all('tr')
        if len(rows) < 4:
            print(f"  -> Warning: Table has only {len(rows)} rows, expected at least 4")
            return ""
        
        # Extract all table data first
        table_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = []
            for cell in cells:
                # Clean the cell text and handle special formatting
                cell_text = cell.get_text(strip=True)
                # Handle overlined text (represented as <font style="text-decoration:overline;">)
                if cell.find('font', {'style': lambda x: x and 'overline' in x}):
                    # Replace overlined characters with a bar notation
                    overlined_parts = cell.find_all('font', {'style': lambda x: x and 'overline' in x})
                    for part in overlined_parts:
                        original_text = part.get_text(strip=True)
                        cell_text = cell_text.replace(original_text, f"̄{original_text}")
                row_data.append(cell_text)
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        if not table_data:
            return ""
        
        # Identify the structure based on the first column content
        wyckoff_row_idx = None
        bandrep_row_idx = None
        decomp_row_idx = None
        kpoint_start_idx = None
        
        for i, row_data in enumerate(table_data):
            if row_data and len(row_data) > 0:
                first_cell = row_data[0].lower()
                if 'wyckoff' in first_cell:
                    wyckoff_row_idx = i
                elif 'band-rep' in first_cell or 'band rep' in first_cell:
                    bandrep_row_idx = i
                elif 'decomposable' in first_cell or 'indecomposable' in first_cell:
                    decomp_row_idx = i
                    kpoint_start_idx = i + 1  # k-points start after decomposable row
                    break
        
        if None in [wyckoff_row_idx, bandrep_row_idx, decomp_row_idx]:
            print("  -> Error: Could not identify all required header rows")
            return ""
        
        # Extract the number of EBR columns (excluding the first label column)
        num_ebr_cols = len(table_data[wyckoff_row_idx]) - 1
        
        # Build the structured text output
        text_lines = []
        
        # Wyckoff positions (skip first column which is the label)
        wyckoff_data = table_data[wyckoff_row_idx][1:]
        text_lines.append("Wyckoff pos. " + " ".join(wyckoff_data))
        
        # Band representations
        bandrep_data = table_data[bandrep_row_idx][1:]
        text_lines.append("Band-Rep. " + " ".join(bandrep_data))
        
        # Decomposable/Indecomposable
        decomp_data = table_data[decomp_row_idx][1:]
        text_lines.append("Decomposable " + " ".join(decomp_data))
        
        # K-point data
        if kpoint_start_idx and kpoint_start_idx < len(table_data):
            kpoint_lines = []
            for i in range(kpoint_start_idx, len(table_data)):
                row_data = table_data[i]
                if len(row_data) >= 2:  # Must have k-point label and at least one irrep
                    k_label = row_data[0]
                    irreps = row_data[1:num_ebr_cols+1]  # Only take the expected number of columns
                    
                    # Skip empty or malformed k-point entries
                    if k_label and any(irrep.strip() for irrep in irreps):
                        irreps_text = " ".join(irreps)
                        kpoint_lines.append(f"{k_label}: {irreps_text}")
            
            if kpoint_lines:
                text_lines.extend(kpoint_lines)
        
        return "\n".join(text_lines)


    def get_ebr_data_for_sg(self, sg_number):
        print(f"\n--- Processing Space Group: {sg_number} ---")
        target_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/bandrep.pl"

        try:
            self.driver.get(target_url)
            wait = WebDriverWait(self.driver, 20)
            
            print("  -> Inputting space group number...")
            # Find the input field for space group number (name="super")
            sg_input_box = wait.until(EC.visibility_of_element_located((By.NAME, 'super')))
            sg_input_box.clear()
            sg_input_box.send_keys(str(sg_number))

            print("  -> Clicking 'Elementary TR' button...")
            # Find and click the Elementary TR button directly (name="elementaryTR")
            elementary_tr_button = wait.until(EC.element_to_be_clickable((By.NAME, 'elementaryTR')))
            elementary_tr_button.click()

            print("  -> Waiting for results table to load...")
            # Wait for the results page to load - look for the specific table structure
            wait.until(EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Wyckoff pos')]")))
            
            # Additional wait to ensure table is fully loaded
            time.sleep(3)
            
            print("  -> Parsing HTML table...")
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find the main data table - it should contain "Wyckoff pos" text
            data_table = None
            tables = soup.find_all('table')
            
            for table in tables:
                if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
                    data_table = table
                    break
            
            if not data_table:
                print("  -> ❌ ERROR: Could not find the main data table on the results page.")
                # Save page source for debugging
                with open(f"debug_sg_{sg_number}.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"  -> Saved page source to debug_sg_{sg_number}.html for inspection")
                return

            formatted_text_data = self._parse_html_table_to_text(data_table)

            if not formatted_text_data:
                print("  -> ❌ ERROR: Failed to parse data from HTML table.")
                return
            
            print(f"  -> Parsed data preview:\n{formatted_text_data[:200]}...")
            
            # Use the database manager to parse and insert the data
            self.db._parse_and_insert_from_text(sg_number, formatted_text_data)
            
        except Exception as e:
            print(f"  -> ❌ An unexpected error occurred for SG {sg_number}: {e}", file=sys.stderr)
            # Save page source for debugging on error
            try:
                with open(f"error_sg_{sg_number}.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"  -> Saved error page source to error_sg_{sg_number}.html")
            except:
                pass

    def run_scraper_for_range(self, start_sg, end_sg):
        print(f"Starting scraper for space groups {start_sg} to {end_sg}.")
        for sg in range(start_sg, end_sg + 1):
            #self.get_ebr_data_for_sg(sg)
            self.get_ebr_data_for_sg_improved(sg)
            print(f"--- Pausing for 5 seconds before next request ---")
            time.sleep(5)
        self.close_driver()

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            print("WebDriver closed.")

# ==============================================================================
# === PART 3: MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    START_SPACE_GROUP = 1
    END_SPACE_GROUP = 3  # Start with a small range for testing

    print("=" * 60)
    print("INITIALIZING EBR DATABASE SCRAPER")
    print("=" * 60)

    db_manager = None
    scraper = None
    try:
        db_manager = EBRDatabaseManager()
        scraper = BCS_Scraper(db_manager)
        scraper.run_scraper_for_range(
            start_sg=START_SPACE_GROUP,
            end_sg=END_SPACE_GROUP
        )
    except Exception as e:
        print(f"\nFATAL ERROR in main execution: {e}", file=sys.stderr)
    finally:
        if scraper:
            scraper.close_driver()
        if db_manager:
            db_manager.close()
        print("\nScript finished.")