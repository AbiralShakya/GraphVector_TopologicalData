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

    def _parse_html_table_to_text(self, table_soup):
        """Converts the BeautifulSoup table object into a structured text block."""
        if not table_soup: 
            return ""
        
        text_block = []
        rows = table_soup.find_all('tr')
        
        if len(rows) < 4:
            print(f"  -> Warning: Table has only {len(rows)} rows, expected at least 4")
            return ""
        
        # Find the header rows by looking for specific text content
        wyckoff_row = None
        bandrep_row = None
        decomp_row = None
        
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if cells:
                first_cell_text = cells[0].get_text(strip=True)
                if 'Wyckoff pos' in first_cell_text:
                    wyckoff_row = row
                elif 'Band-Rep' in first_cell_text:
                    bandrep_row = row
                elif 'Decomposable' in first_cell_text or 'Indecomposable' in first_cell_text:
                    decomp_row = row
                    break  # Data rows start after this
        
        if not all([wyckoff_row, bandrep_row, decomp_row]):
            print("  -> Error: Could not find all required header rows")
            return ""
        
        # Extract header data
        text_block.append("Wyckoff pos. " + " ".join(td.get_text(strip=True) for td in wyckoff_row.find_all(['td', 'th'])[1:]))
        text_block.append("Band-Rep. " + " ".join(td.get_text(strip=True) for td in bandrep_row.find_all(['td', 'th'])[1:]))
        text_block.append("Decomposable " + " ".join(td.get_text(strip=True) for td in decomp_row.find_all(['td', 'th'])[1:]))

        # Extract k-point data (rows after decomposable row)
        decomp_index = rows.index(decomp_row)
        kpoint_lines = []
        
        for row in rows[decomp_index + 1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) > 1:
                k_point_info = cells[0].get_text(strip=True)
                if k_point_info and not k_point_info.isspace():  # Skip empty rows
                    irreps = " ".join(td.get_text(strip=True) for td in cells[1:])
                    kpoint_lines.append(f"{k_point_info}: {irreps}")
        
        if kpoint_lines:
            text_block.append("\n".join(kpoint_lines))
        
        return "\n\n".join(text_block)

    def get_ebr_data_for_sg(self, sg_number):
        print(f"\n--- Processing Space Group: {sg_number} ---")
        target_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/bandrep.pl"

        try:
            self.driver.get(target_url)
            wait = WebDriverWait(self.driver, 20)
            
            print("  -> Inputting space group number...")
            # Find the input field for space group number
            sg_input_box = wait.until(EC.visibility_of_element_located((By.NAME, 'gnum')))
            sg_input_box.clear()
            sg_input_box.send_keys(str(sg_number))

            print("  -> Clicking 'choose it' button...")
            # Find and click the 'choose it' button
            choose_it_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@value='choose it']")))
            choose_it_button.click()
            
            # Wait for the page to load the space group options
            time.sleep(2)

            print("  -> Selecting 'Elementary TR' option...")
            # Look for the radio button for Elementary TR (option 2)
            elementary_tr_radio = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@name='what' and @value='2']")))
            self.driver.execute_script("arguments[0].click();", elementary_tr_radio)
            
            print("  -> Submitting form...")
            # Find and click the submit button
            submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']")))
            submit_button.click()

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
            self.get_ebr_data_for_sg(sg)
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