import sqlite3
import re
import time
import sys
import os
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.action_chains import ActionChains

class ImprovedBCS_Scraper:
    """Enhanced scraper with better error handling, fallback strategies, and stability improvements."""
    
    def __init__(self, database_manager):
        self.db = database_manager
        self.driver = None
        self.session = requests.Session()
        self.base_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/bandrep.pl"
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        print("Improved BCS Scraper initialized.")

    def init_driver(self, headless=True):
        """Initialize Chrome driver with improved stability settings."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        
        options = webdriver.ChromeOptions()
        
        # Enhanced stability options
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')
        options.add_argument('--disable-javascript')  # Try without JS first
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--disable-logging')
        options.add_argument('--disable-dev-tools')
        options.add_argument('--no-first-run')
        options.add_argument('--no-default-browser-check')
        
        # Randomize user agent
        user_agent = random.choice(self.user_agents)
        options.add_argument(f'--user-agent={user_agent}')
        
        # Memory and performance settings
        options.add_argument('--memory-pressure-off')
        options.add_argument('--max_old_space_size=4096')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(45)  # Increased timeout
            self.driver.implicitly_wait(10)
            
            # Set additional timeouts
            self.driver.set_script_timeout(30)
            
            print("  -> Chrome driver initialized successfully")
            return True
        except Exception as e:
            print(f"  -> ❌ Failed to initialize Chrome driver: {e}")
            return False

    def try_requests_fallback(self, sg_number):
        """Fallback method using requests library instead of Selenium."""
        try:
            print("  -> Attempting requests-based fallback...")
            
            # Set up session with headers
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # First, get the main page
            response = self.session.get(self.base_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the form and prepare POST data
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the form and prepare POST data
            post_data = {
                'super': str(sg_number),
                'elementaryTR': 'Elementary TR',
                'nomaximal': 'yes',
                'elementary': '',
                'wyck': '',
                'wyckTR': '',
                'wyckoff': ''
            }
            
            # Submit the form
            print(f"  -> Submitting form for SG {sg_number}...")
            response = self.session.post(self.base_url, data=post_data, headers=headers, timeout=45)
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the data table
            data_table = None
            tables = soup.find_all('table')
            
            for table in tables:
                if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
                    data_table = table
                    break
            
            if data_table:
                print("  -> ✅ Successfully retrieved data using requests fallback")
                return self.parse_table_to_dict(data_table, sg_number)
            else:
                print("  -> ❌ Could not find data table in requests response")
                return None
                
        except Exception as e:
            print(f"  -> ❌ Requests fallback failed: {e}")
            return None

    def extract_table_data_with_fallbacks(self, sg_number, max_retries=3):
        """Extract table data with multiple fallback strategies."""
        
        # Strategy 1: Try Selenium with headless mode
        for attempt in range(max_retries):
            print(f"\n--- Selenium Attempt {attempt + 1}/{max_retries} for Space Group: {sg_number} ---")
            
            try:
                if not self.driver:
                    if not self.init_driver(headless=True):
                        continue
                
                result = self.selenium_extraction(sg_number)
                if result:
                    return result
                    
            except Exception as e:
                print(f"  -> Selenium attempt {attempt + 1} failed: {e}")
                self.init_driver(headless=True)  # Reinitialize driver
            
            if attempt < max_retries - 1:
                wait_time = random.uniform(15, 25)  # Random wait
                print(f"  -> Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
        
        # Strategy 2: Try Selenium with GUI mode (less headless issues)
        print(f"\n--- Trying Selenium with GUI mode for SG {sg_number} ---")
        try:
            if not self.init_driver(headless=False):
                pass
            else:
                result = self.selenium_extraction(sg_number)
                if result:
                    return result
        except Exception as e:
            print(f"  -> GUI Selenium failed: {e}")
        
        # Strategy 3: Try requests-based fallback
        print(f"\n--- Trying requests fallback for SG {sg_number} ---")
        result = self.try_requests_fallback(sg_number)
        if result:
            return result
        
        # Strategy 4: Try alternative approach with different timing
        print(f"\n--- Trying alternative Selenium approach for SG {sg_number} ---")
        try:
            if not self.init_driver(headless=True):
                pass
            else:
                result = self.selenium_extraction_alternative(sg_number)
                if result:
                    return result
        except Exception as e:
            print(f"  -> Alternative approach failed: {e}")
        
        print(f"  -> ❌ All strategies failed for SG {sg_number}")
        return None

    def selenium_extraction(self, sg_number):
        """Standard Selenium extraction method."""
        try:
            print("  -> Loading main page...")
            self.driver.get(self.base_url)
            
            # Wait for page to load completely
            wait = WebDriverWait(self.driver, 20)
            
            # Wait for and find the space group input
            print("  -> Looking for space group input field...")
            sg_input = wait.until(EC.presence_of_element_located((By.NAME, 'super')))
            
            # Clear and input with more natural timing
            sg_input.clear()
            time.sleep(1)
            for char in str(sg_number):
                sg_input.send_keys(char)
                time.sleep(0.1)
            
            time.sleep(2)
            
            # Find and click the Elementary TR button
            print("  -> Looking for Elementary TR button...")
            elementary_tr_button = wait.until(EC.element_to_be_clickable((By.NAME, 'elementaryTR')))
            
            # Use ActionChains for more natural clicking
            actions = ActionChains(self.driver)
            actions.move_to_element(elementary_tr_button).pause(1).click().perform()
            
            print("  -> Waiting for results page...")
            # Wait for results with multiple possible indicators
            try:
                wait.until(EC.any_of(
                    EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Wyckoff pos')]")),
                    EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Band-Rep')]")),
                    EC.presence_of_element_located((By.XPATH, "//h2[contains(text(), 'Elementary band-representations')]"))
                ))
            except TimeoutException:
                print("  -> Standard wait failed, trying extended wait...")
                time.sleep(10)  # Give it more time
            
            # Parse the results
            print("  -> Parsing page content...")
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find the main data table
            data_table = self.find_data_table(soup)
            
            if data_table:
                table_dict = self.parse_table_to_dict(data_table, sg_number)
                if self.validate_table_data(table_dict):
                    return table_dict
            
            return None
            
        except Exception as e:
            print(f"  -> Selenium extraction error: {e}")
            self.save_debug_html(sg_number, "selenium_error")
            return None

    def selenium_extraction_alternative(self, sg_number):
        """Alternative Selenium extraction with different approach."""
        try:
            print("  -> Alternative approach: Direct POST simulation...")
            
            # Navigate to page
            self.driver.get(self.base_url)
            time.sleep(5)
            
            # Try to submit form via JavaScript
            js_script = f"""
            var form = document.querySelector('form');
            var superInput = document.querySelector('input[name="super"]');
            var elementaryTRInput = document.querySelector('input[name="elementaryTR"]');
            
            if (superInput) superInput.value = '{sg_number}';
            if (form && elementaryTRInput) {{
                elementaryTRInput.click();
            }}
            """
            
            self.driver.execute_script(js_script)
            time.sleep(10)
            
            # Check if we got results
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            data_table = self.find_data_table(soup)
            
            if data_table:
                table_dict = self.parse_table_to_dict(data_table, sg_number)
                if self.validate_table_data(table_dict):
                    return table_dict
            
            return None
            
        except Exception as e:
            print(f"  -> Alternative extraction error: {e}")
            return None

    def find_data_table(self, soup):
        """Find the main data table in the HTML."""
        tables = soup.find_all('table')
        
        for table in tables:
            # Look for table containing the specific structure
            if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
                return table
        
        return None

    def parse_table_to_dict(self, table_soup, sg_number):
        """Parse the HTML table into a structured dictionary."""
        try:
            rows = table_soup.find_all('tr')
            if len(rows) < 4:
                print(f"  -> Warning: Table has only {len(rows)} rows")
                return None
            
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
                
                first_cell_text = cells[0].get_text(strip=True).lower()
                row_data = []
                
                # Extract cell data, handling special formatting
                for cell in cells[1:]:
                    text = cell.get_text(strip=True)
                    # Handle overlined text if present
                    if cell.find('font', {'style': lambda x: x and 'overline' in x}):
                        text = self.handle_overlined_text(cell)
                    row_data.append(text)
                
                # Categorize the row based on first cell content
                if 'wyckoff' in first_cell_text:
                    table_dict['wyckoff_positions'] = row_data
                elif 'band-rep' in first_cell_text or 'band rep' in first_cell_text:
                    table_dict['band_representations'] = row_data
                elif 'decomposable' in first_cell_text or 'indecomposable' in first_cell_text:
                    table_dict['decomposability'] = row_data
                elif ':' in cells[0].get_text(strip=True):
                    # K-point row
                    kpoint_label = cells[0].get_text(strip=True)
                    table_dict['kpoints'][kpoint_label] = row_data
            
            return table_dict
            
        except Exception as e:
            print(f"  -> Error parsing table: {e}")
            return None

    def handle_overlined_text(self, cell):
        """Handle overlined text in table cells."""
        text = cell.get_text(strip=True)
        return text

    def validate_table_data(self, table_dict):
        """Validate that the extracted table data makes sense."""
        if not table_dict:
            return False
        
        # Check that we have the essential components
        if not table_dict['wyckoff_positions']:
            print("  -> Validation failed: No Wyckoff positions found")
            return False
        
        if not table_dict['band_representations']:
            print("  -> Validation failed: No band representations found")
            return False
        
        if not table_dict['kpoints']:
            print("  -> Validation failed: No k-points found")
            return False
        
        # Check that the number of columns is consistent
        num_ebrs = len(table_dict['wyckoff_positions'])
        if len(table_dict['band_representations']) != num_ebrs:
            print(f"  -> Warning: Mismatch in EBR columns - Wyckoff: {num_ebrs}, Band-rep: {len(table_dict['band_representations'])}")
        
        print(f"  -> Validation passed: {num_ebrs} EBRs, {len(table_dict['kpoints'])} k-points")
        return True

    def save_debug_html(self, sg_number, suffix=""):
        """Save HTML for debugging purposes."""
        try:
            debug_dir = "debug_html"
            os.makedirs(debug_dir, exist_ok=True)
            filename = f"{debug_dir}/sg_{sg_number}_{suffix}_{int(time.time())}.html"
            
            if self.driver:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"  -> Debug HTML saved to {filename}")
        except Exception as e:
            print(f"  -> Could not save debug HTML: {e}")

    def convert_to_text_format(self, table_dict):
        """Convert structured table data back to text format for database insertion."""
        if not table_dict:
            return ""
        
        text_lines = []
        
        if table_dict['wyckoff_positions']:
            text_lines.append("Wyckoff pos. " + " ".join(table_dict['wyckoff_positions']))
        
        if table_dict['band_representations']:
            text_lines.append("Band-Rep. " + " ".join(table_dict['band_representations']))
        
        if table_dict['decomposability']:
            text_lines.append("Decomposable " + " ".join(table_dict['decomposability']))
        
        # Add k-point data
        for kpoint, irreps in table_dict['kpoints'].items():
            irreps_text = " ".join(irreps)
            text_lines.append(f"{kpoint}: {irreps_text}")
        
        return "\n".join(text_lines)
    
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

    def process_space_group(self, sg_number):
        """Main method to process a single space group."""
        table_data = self.extract_table_data_with_fallbacks(sg_number)
        
        if not table_data:
            return False
        
        # Convert to text format for database insertion
        formatted_text = self.convert_to_text_format(table_data)
        
        if not formatted_text:
            print(f"  -> ❌ Failed to format data for SG {sg_number}")
            return False
        
        print(f"  -> Data preview:\n{formatted_text[:200]}...")
        
        # Insert into database
        try:
            self.db._parse_and_insert_from_text(sg_number, formatted_text)
            return True
        except Exception as e:
            print(f"  -> ❌ Database insertion failed for SG {sg_number}: {e}")
            return False

    def run_scraper_for_range(self, start_sg, end_sg):
        """Run the scraper for a range of space groups."""
        print(f"Starting improved scraper for space groups {start_sg} to {end_sg}.")
        
        successful = 0
        failed = 0
        failed_sgs = []
        
        for sg in range(start_sg, end_sg + 1):
            print(f"\n{'='*60}")
            print(f"PROCESSING SPACE GROUP {sg}")
            print(f"{'='*60}")
            
            if self.process_space_group(sg):
                successful += 1
                print(f"✅ Successfully processed SG {sg}")
            else:
                failed += 1
                failed_sgs.append(sg)
                print(f"❌ Failed to process SG {sg}")
            
            # Adaptive pause between requests
            if sg < end_sg:
                pause_time = random.uniform(12, 20)
                print(f"--- Pausing for {pause_time:.1f} seconds before next request ---")
                time.sleep(pause_time)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"Successful: {successful}, Failed: {failed}")
        if failed_sgs:
            print(f"Failed SGs: {failed_sgs}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        print(f"{'='*60}")
        
        self.close_driver()
        return failed_sgs

    def close_driver(self):
        """Close the WebDriver and session."""
        if self.driver:
            try:
                self.driver.quit()
                print("WebDriver closed.")
            except:
                pass
            self.driver = None
        
        if self.session:
            try:
                self.session.close()
                print("Requests session closed.")
            except:
                pass

# Usage example with retry logic for failures
if __name__ == "__main__":
    from PEBR_TR_nonmagnetic_query import EBRDatabaseManager
    
    db_manager = EBRDatabaseManager()
    scraper = ImprovedBCS_Scraper(db_manager)
    
    try:
        # Initial run
        failed_sgs = scraper.run_scraper_for_range(1, 10)
        
        # Retry failed space groups
        if failed_sgs:
            print(f"\n🔄 Retrying {len(failed_sgs)} failed space groups...")
            time.sleep(30)  # Longer pause before retry
            
            still_failed = []
            for sg in failed_sgs:
                print(f"\n🔄 RETRY: Space Group {sg}")
                if scraper.process_space_group(sg):
                    print(f"✅ Retry successful for SG {sg}")
                else:
                    still_failed.append(sg)
                    print(f"❌ Retry failed for SG {sg}")
                time.sleep(random.uniform(15, 25))
            
            if still_failed:
                print(f"\n⚠️  Still failed after retry: {still_failed}")
            else:
                print(f"\n🎉 All retries successful!")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()
        db_manager.close()