# import sqlite3
# import re
# import time
# import sys
# import os
# import random
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# import requests
# from selenium.webdriver.common.action_chains import ActionChains

# class ImprovedBCS_Scraper:
#     """Enhanced scraper with better error handling, fallback strategies, and stability improvements."""
    
#     def __init__(self, database_manager):
#         self.db = database_manager
#         self.driver = None
#         self.session = requests.Session()
#         self.base_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/bandrep.pl"
#         self.user_agents = [
#             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         ]
#         print("Improved BCS Scraper initialized.")

#     def init_driver(self, headless=True):
#         """Initialize Chrome driver with improved stability settings."""
#         if self.driver:
#             try:
#                 self.driver.quit()
#             except:
#                 pass
        
#         options = webdriver.ChromeOptions()
        
#         # Enhanced stability options
#         if headless:
#             options.add_argument('--headless')
#         options.add_argument('--no-sandbox')
#         options.add_argument('--disable-dev-shm-usage')
#         options.add_argument('--disable-gpu')
#         options.add_argument('--disable-extensions')
#         options.add_argument('--disable-plugins')
#         options.add_argument('--disable-images')
#         #options.add_argument('--disable-javascript')  
#         options.add_argument('--window-size=1920,1080')
#         options.add_argument('--disable-blink-features=AutomationControlled')
#         options.add_argument('--disable-web-security')
#         options.add_argument('--allow-running-insecure-content')
#         options.add_argument('--ignore-certificate-errors')
#         options.add_argument('--ignore-ssl-errors')
#         options.add_argument('--disable-logging')
#         options.add_argument('--disable-dev-tools')
#         options.add_argument('--no-first-run')
#         options.add_argument('--no-default-browser-check')
        
#         # Randomize user agent
#         user_agent = random.choice(self.user_agents)
#         options.add_argument(f'--user-agent={user_agent}')
        
#         # Memory and performance settings
#         options.add_argument('--memory-pressure-off')
#         options.add_argument('--max_old_space_size=4096')
        
#         try:
#             service = Service(ChromeDriverManager().install())
#             self.driver = webdriver.Chrome(service=service, options=options)
#             self.driver.set_page_load_timeout(45)  # Increased timeout
#             self.driver.implicitly_wait(10)
            
#             # Set additional timeouts
#             self.driver.set_script_timeout(30)
            
#             print("  -> Chrome driver initialized successfully")
#             return True
#         except Exception as e:
#             print(f"  -> ❌ Failed to initialize Chrome driver: {e}")
#             return False

#     def try_requests_fallback(self, sg_number):
#         """Fallback method using requests library instead of Selenium."""
#         try:
#             print("  -> Attempting requests-based fallback...")
            
#             # Set up session with headers
#             headers = {
#                 'User-Agent': random.choice(self.user_agents),
#                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#                 'Accept-Language': 'en-US,en;q=0.5',
#                 'Accept-Encoding': 'gzip, deflate',
#                 'Connection': 'keep-alive',
#                 'Upgrade-Insecure-Requests': '1',
#             }
            
#             # First, get the main page
#             response = self.session.get(self.base_url, headers=headers, timeout=30)
#             response.raise_for_status()
            
#             # Parse the form and prepare POST data
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Find the form and prepare POST data
#             post_data = {
#                 'super': str(sg_number),
#                 'elementaryTR': 'Elementary TR',
#                 'nomaximal': 'yes',
#                 'elementary': '',
#                 'wyck': '',
#                 'wyckTR': '',
#                 'wyckoff': ''
#             }
            
#             # Submit the form
#             print(f"  -> Submitting form for SG {sg_number}...")
#             response = self.session.post(self.base_url, data=post_data, headers=headers, timeout=45)
#             response.raise_for_status()
            
#             # Parse the response
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Find the data table
#             data_table = None
#             tables = soup.find_all('table')
            
#             for table in tables:
#                 if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
#                     data_table = table
#                     break
            
#             if data_table:
#                 print("  -> ✅ Successfully retrieved data using requests fallback")
#                 return self.parse_table_to_dict(data_table, sg_number)
#             else:
#                 print("  -> ❌ Could not find data table in requests response")
#                 return None
                
#         except Exception as e:
#             print(f"  -> ❌ Requests fallback failed: {e}")
#             return None
        
#     # Add this new method to your ImprovedBCS_Scraper class

#     def scrape_decomposition_page(self, cell_element):
#         """
#         Handles clicking a 'Decomposable' button, scraping the new tab,
#         and returning the structured decomposition data.
#         """
#         print("  -> Found 'Decomposable' button. Scraping decomposition page...")
#         original_window = self.driver.current_window_handle
        
#         try:
#             # Find and click the button within the cell
#             button = cell_element.find_element(By.TAG_NAME, "input")
#             button.click()

#             # Wait for the new tab to open and switch to it
#             WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
#             for window_handle in self.driver.window_handles:
#                 if window_handle != original_window:
#                     self.driver.switch_to.window(window_handle)
#                     break
            
#             # Now we are on the new page, parse it
#             page_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
#             # Find the specific table with the branches
#             decomposition_table = page_soup.find('table', string=lambda text: text and 'branch' in text.lower())
            
#             if not decomposition_table:
#                 print("  -> ❌ Could not find decomposition table on new page.")
#                 self.driver.close()
#                 self.driver.switch_to.window(original_window)
#                 return {'type': 'decomposable', 'data': 'Error: Table not found'}

#             rows = decomposition_table.find_all('tr')
#             branches_data = []
            
#             # The first row is headers (e.g., branch 1, branch 2)
#             # The subsequent rows are the data
#             for row in rows[1:]: # Skip header
#                 cells = row.find_all('td')
#                 if len(cells) > 1: # Ensure it's a data row
#                     # The first cell is the index, the rest are the branch data
#                     branch_irreps = [cell.get_text(strip=True) for cell in cells[1:]]
#                     branches_data.append(branch_irreps)
            
#             print(f"  -> ✅ Scraped {len(branches_data)} decomposition(s) with {len(branches_data[0])} branch(es).")
            
#             # Close the new tab and switch back
#             self.driver.close()
#             self.driver.switch_to.window(original_window)
            
#             return {'type': 'decomposable', 'data': branches_data}

#         except Exception as e:
#             print(f"  -> ❌ Error during decomposition scraping: {e}")
#             # Ensure we switch back to the main window in case of an error
#             if len(self.driver.window_handles) > 1:
#                 self.driver.close()
#             self.driver.switch_to.window(original_window)
#             return {'type': 'decomposable', 'data': f'Error: {e}'}

#     def extract_table_data_with_fallbacks(self, sg_number, max_retries=3):
#         """Extract table data with multiple fallback strategies."""
        
#         # Strategy 1: Try Selenium with headless mode
#         for attempt in range(max_retries):
#             print(f"\n--- Selenium Attempt {attempt + 1}/{max_retries} for Space Group: {sg_number} ---")
            
#             try:
#                 if not self.driver:
#                     if not self.init_driver(headless=True):
#                         continue
                
#                 result = self.selenium_extraction(sg_number)
#                 if result:
#                     return result
                    
#             except Exception as e:
#                 print(f"  -> Selenium attempt {attempt + 1} failed: {e}")
#                 self.init_driver(headless=True)  # Reinitialize driver
            
#             if attempt < max_retries - 1:
#                 wait_time = random.uniform(15, 25)  # Random wait
#                 print(f"  -> Waiting {wait_time:.1f} seconds before retry...")
#                 time.sleep(wait_time)
        
#         # Strategy 2: Try Selenium with GUI mode (less headless issues)
#         print(f"\n--- Trying Selenium with GUI mode for SG {sg_number} ---")
#         try:
#             if not self.init_driver(headless=False):
#                 pass
#             else:
#                 result = self.selenium_extraction(sg_number)
#                 if result:
#                     return result
#         except Exception as e:
#             print(f"  -> GUI Selenium failed: {e}")
        
#         # Strategy 3: Try requests-based fallback
#         print(f"\n--- Trying requests fallback for SG {sg_number} ---")
#         result = self.try_requests_fallback(sg_number)
#         if result:
#             return result
        
#         # Strategy 4: Try alternative approach with different timing
#         print(f"\n--- Trying alternative Selenium approach for SG {sg_number} ---")
#         try:
#             if not self.init_driver(headless=True):
#                 pass
#             else:
#                 result = self.selenium_extraction_alternative(sg_number)
#                 if result:
#                     return result
#         except Exception as e:
#             print(f"  -> Alternative approach failed: {e}")
        
#         print(f"  -> ❌ All strategies failed for SG {sg_number}")
#         return None
    
#     # Add or replace this function in your ImprovedBCS_Scraper class.
# # This function will now be the main entry point for processing and saving a space group.

#     def process_and_ingest_space_group(self, sg_number):
#         """
#         Orchestrates the entire process:
#         1. Scrapes the data from the website, including decomposable branches.
#         2. Ingests the main EBR data into the database.
#         3. Ingests the decomposition branch data, linking it to the main EBR entries.
#         """
#         print(f"--- Starting full processing and ingestion for SG {sg_number} ---")

#         # STEP 1: Scrape the data from the website using the previously defined logic
#         # This returns the structured dictionary, including the special dict for decomposable data.
#         table_data = self.extract_table_data_with_fallbacks(sg_number) # Assumes this is your main scraping method
        
#         if not table_data or not table_data.get('wyckoff_positions'):
#             print(f"❌ Scraping failed or returned no data for SG {sg_number}. Aborting ingestion.")
#             return False

#         # STEP 2: Prepare and insert the MAIN EBR data to get the ebr_ids
#         try:
#             wyckoff_entries = table_data['wyckoff_positions']
#             orbital_entries = table_data['band_representations']
            
#             # Create a simple list of notes for the main insertion.
#             # If the entry is a complex dict, we just mark it as 'decomposable'.
#             notes_for_main_insertion = []
#             for item in table_data.get('decomposability', []):
#                 if isinstance(item, dict) and item.get('type') == 'decomposable':
#                     notes_for_main_insertion.append('decomposable')
#                 else:
#                     notes_for_main_insertion.append(str(item).lower())

#             # Prepare k-point data in the format expected by _insert_data
#             kpoint_data_list = []
#             for kpoint_label, irrep_cells in table_data['kpoints'].items():
#                 kpoint_data_list.append((kpoint_label.split(':')[0], irrep_cells))
            
#             print(f"  -> Ingesting main EBR data for {len(wyckoff_entries)} columns...")
            
#             # Use the powerful _insert_data method from your DB manager
#             # This returns a list of dicts with the crucial 'ebr_id' for each column
#             inserted_ebrs_info = self.db._insert_data(
#                 sg_number,
#                 wyckoff_entries,
#                 orbital_entries,
#                 notes_for_main_insertion,
#                 kpoint_data_list
#             )
#             print(f"  -> ✅ Main data ingestion complete. {len(inserted_ebrs_info)} EBRs created.")

#         except Exception as e:
#             print(f"❌ An error occurred during the main data ingestion for SG {sg_number}: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#         # STEP 3: Loop through the results and insert the DECOMPOSITION BRANCH data
#         decomposable_ebrs_found = 0
#         for col_index, ebr_info in enumerate(inserted_ebrs_info):
#             # Check if this column was marked as decomposable
#             if ebr_info['note'].lower() == 'decomposable':
#                 decomposable_ebrs_found += 1
#                 ebr_id = ebr_info['ebr_id']
                
#                 # Get the corresponding structured data from our original scrape
#                 decomposition_info = table_data['decomposability'][col_index]
                
#                 if isinstance(decomposition_info, dict) and 'data' in decomposition_info:
#                     branch_data_rows = decomposition_info['data']
#                     print(f"  -> Found {len(branch_data_rows)} decomposition branches for EBR ID {ebr_id}. Ingesting...")
                    
#                     for decomposition_index, branch_row in enumerate(branch_data_rows, 1):
#                         # The branch_row is a list like ['branch1_str', 'branch2_str']
#                         if len(branch_row) == 2:
#                             branch1_str, branch2_str = branch_row
#                             try:
#                                 # Use the DB manager's dedicated function to add the branch
#                                 self.db.add_ebr_decomposition_branch(
#                                     ebr_id,
#                                     decomposition_index,
#                                     branch1_str,
#                                     branch2_str
#                                 )
#                                 print(f"    -> ✅ Added branch index {decomposition_index} for EBR ID {ebr_id}.")
#                             except ValueError as ve:
#                                 print(f"    -> ❌ Validation Error adding branch for EBR ID {ebr_id}: {ve}")
#                             except Exception as e:
#                                 print(f"    -> ❌ Error adding branch for EBR ID {ebr_id}: {e}")
#                         else:
#                             print(f"    -> ❌ Malformed branch data for EBR ID {ebr_id}: expected 2 branches, found {len(branch_row)}")

#         if decomposable_ebrs_found > 0:
#             print(f"  -> ✅ Decomposition branch ingestion complete.")
#         else:
#             print("  -> No decomposable EBRs found in this scrape.")

#         return True

#     def selenium_extraction(self, sg_number):
#         """Standard Selenium extraction method."""
#         try:
#             print("  -> Loading main page...")
#             self.driver.get(self.base_url)
            
#             # Wait for page to load completely
#             wait = WebDriverWait(self.driver, 20)
            
#             # Wait for and find the space group input
#             print("  -> Looking for space group input field...")
#             sg_input = wait.until(EC.presence_of_element_located((By.NAME, 'super')))
            
#             # Clear and input with more natural timing
#             sg_input.clear()
#             time.sleep(1)
#             for char in str(sg_number):
#                 sg_input.send_keys(char)
#                 time.sleep(0.1)
            
#             time.sleep(2)
            
#             # Find and click the Elementary TR button
#             print("  -> Looking for Elementary TR button...")
#             elementary_tr_button = wait.until(EC.element_to_be_clickable((By.NAME, 'elementaryTR')))
            
#             # Use ActionChains for more natural clicking
#             actions = ActionChains(self.driver)
#             actions.move_to_element(elementary_tr_button).pause(1).click().perform()
            
#             print("  -> Waiting for results page...")
#             # Wait for results with multiple possible indicators
#             try:
#                 wait.until(EC.any_of(
#                     EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Wyckoff pos')]")),
#                     EC.presence_of_element_located((By.XPATH, "//td[contains(text(), 'Band-Rep')]")),
#                     EC.presence_of_element_located((By.XPATH, "//h2[contains(text(), 'Elementary band-representations')]"))
#                 ))
#             except TimeoutException:
#                 print("  -> Standard wait failed, trying extended wait...")
#                 time.sleep(10)  # Give it more time
            
#             # Parse the results
#             print("  ->  content...")
#             soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
#             # Find the main data table
#             data_table = self.find_data_table(soup)
            
#             if data_table:
#                 table_dict = self.parse_table_to_dict(data_table, sg_number)
#                 if self.validate_table_data(table_dict):
#                     return table_dict
            
#             return None
            
#         except Exception as e:
#             print(f"  -> Selenium extraction error: {e}")
#             self.save_debug_html(sg_number, "selenium_error")
#             return None

#     def selenium_extraction_alternative(self, sg_number):
#         """Alternative Selenium extraction with different approach."""
#         try:
#             print("  -> Alternative approach: Direct POST simulation...")
            
#             # Navigate to page
#             self.driver.get(self.base_url)
#             time.sleep(5)
            
#             # Try to submit form via JavaScript
#             js_script = f"""
#             var form = document.querySelector('form');
#             var superInput = document.querySelector('input[name="super"]');
#             var elementaryTRInput = document.querySelector('input[name="elementaryTR"]');
            
#             if (superInput) superInput.value = '{sg_number}';
#             if (form && elementaryTRInput) {{
#                 elementaryTRInput.click();
#             }}
#             """
            
#             self.driver.execute_script(js_script)
#             time.sleep(10)
            
#             # Check if we got results
#             soup = BeautifulSoup(self.driver.page_source, 'html.parser')
#             data_table = self.find_data_table(soup)
            
#             if data_table:
#                 table_dict = self.parse_table_to_dict(data_table, sg_number)
#                 if self.validate_table_data(table_dict):
#                     return table_dict
            
#             return None
            
#         except Exception as e:
#             print(f"  -> Alternative extraction error: {e}")
#             return None

#     def find_data_table(self, soup):
#         """Find the main data table in the HTML."""
#         tables = soup.find_all('table')
        
#         for table in tables:
#             # Look for table containing the specific structure
#             if table.find('td', string=lambda text: text and 'Wyckoff pos' in text):
#                 return table
        
#         return None

#     # This is the MODIFIED version of your parse_table_to_dict method
# # It includes the fix for the NoSuchElementException

#     # This is the corrected version of your parse_table_to_dict function.
# # It fixes the StaleElementReferenceException.

#     def parse_table_to_dict(self, table_soup, sg_number):
#         """
#         Parse the HTML table into a structured dictionary. Fixes StaleElementReferenceException
#         by re-finding the table element after potential page changes.
#         """
#         try:
#             rows = table_soup.find_all('tr')
#             table_dict = {
#                 'space_group': sg_number,
#                 'wyckoff_positions': [], 'band_representations': [],
#                 'decomposability': [], 'kpoints': {}
#             }

#             # We will re-find the main_table_element inside the loop when necessary
#             # to ensure it's not stale.

#             for row_index, row in enumerate(rows):
#                 cells = row.find_all(['td', 'th'])
#                 if not cells: continue

#                 first_cell_text = cells[0].get_text(strip=True).lower()
                
#                 if 'decomposable' not in first_cell_text:
#                     # Handle simple text rows with BeautifulSoup as before
#                     row_data = [cell.get_text(strip=True).replace('\n', ' ') for cell in cells[1:]]
#                     if 'wyckoff' in first_cell_text:
#                         table_dict['wyckoff_positions'] = row_data
#                     elif 'band-rep' in first_cell_text:
#                         table_dict['band_representations'] = row_data
#                     elif ':' in cells[0].get_text(strip=True):
#                         kpoint_label = cells[0].get_text(strip=True)
#                         table_dict['kpoints'][kpoint_label] = row_data
#                 else:
#                     # --- Handle the Decomposable/Indecomposable row with Selenium ---
#                     decomposability_data = []
                    
#                     # --- START: THE FIX ---
#                     # We re-find the table here because a previous iteration might have made the
#                     # original reference stale by switching tabs.
#                     try:
#                         wait = WebDriverWait(self.driver, 10)
#                         xpath_selector = "//td[contains(text(), 'Wyckoff pos')]/ancestor::table"
#                         main_table_element = wait.until(EC.presence_of_element_located((By.XPATH, xpath_selector)))
#                     except TimeoutException:
#                         print("❌ Could not re-find the main data table. Aborting parse.")
#                         return None
#                     # --- END: THE FIX ---

#                     selenium_cells = main_table_element.find_elements(By.XPATH, f".//tr[{row_index + 1}]/td")[1:]

#                     for cell_element in selenium_cells:
#                         cell_html = cell_element.get_attribute('innerHTML')
#                         if 'bandrepdesc.pl' in cell_html and 'Decomposable' in cell_html:
#                             decomposition = self.scrape_decomposition_page(cell_element)
#                             decomposability_data.append(decomposition)
#                         else:
#                             decomposability_data.append("Indecomposable")
                    
#                     table_dict['decomposability'] = decomposability_data

#             return table_dict

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             print(f"  -> Error parsing table: {e}")
#             return None
        
#     def parse_cell_content(self, cell):
#         """Parse cell content, handling overlined text and multiple irreps properly."""
#         try:
#             # Get the raw text first
#             text = cell.get_text(strip=True)
#             decomposable = cell.find('form', action=lambda x: x and 'bandrepdesc.pl' in x)
#             if decomposable:
#                 return "DECOMPOSABLE_BUTTON"

            
#             # Check if this cell contains overlined text
#             overlined_fonts = cell.find_all('font', {'style': lambda x: x and 'overline' in x})
            
#             if overlined_fonts:
#                 # This is a complex cell with overlined text
#                 # Need to parse it more carefully
#                 cell_parts = []
                
#                 # Get all text nodes and font elements
#                 for element in cell.children:
#                     if element.name == 'font':
#                         # Check if it's overlined
#                         if element.get('style') and 'overline' in element.get('style'):
#                             # This is overlined text - add a bar marker
#                             overlined_text = element.get_text(strip=True)
#                             cell_parts.append(f"¯{overlined_text}")
#                         else:
#                             cell_parts.append(element.get_text(strip=True))
#                     elif hasattr(element, 'strip'):
#                         # Text node
#                         text_content = element.strip()
#                         if text_content:
#                             cell_parts.append(text_content)
                
#                 # Join the parts
#                 processed_text = ''.join(cell_parts)
                
#                 # Handle cases like "R2R2(2)" which should be treated as one irrep
#                 # Look for pattern: letters followed by numbers, then same letters+numbers, then (dimension)
#                 import re
#                 pattern = r'([A-Z])(\d+)\1\2\((\d+)\)'
#                 match = re.search(pattern, processed_text)
                
#                 if match:
#                     # This is a doubled irrep like R2R2(2)
#                     return processed_text
#                 else:
#                     return processed_text
#             else:
#                 # Simple text cell
#                 return text
                
#         except Exception as e:
#             print(f"  -> Error parsing cell content: {e}")
#             return cell.get_text(strip=True)

#     def validate_table_data(self, table_dict):
#         """Validate that the extracted table data makes sense."""
#         if not table_dict:
#             return False
        
#         # Check that we have the essential components
#         if not table_dict['wyckoff_positions']:
#             print("  -> Validation failed: No Wyckoff positions found")
#             return False
        
#         if not table_dict['band_representations']:
#             print("  -> Validation failed: No band representations found")
#             return False
        
#         if not table_dict['kpoints']:
#             print("  -> Validation failed: No k-points found")
#             return False
        
#         # Check that the number of columns is consistent
#         num_ebrs = len(table_dict['wyckoff_positions'])
#         if len(table_dict['band_representations']) != num_ebrs:
#             print(f"  -> Warning: Mismatch in EBR columns - Wyckoff: {num_ebrs}, Band-rep: {len(table_dict['band_representations'])}")
        
#         print(f"  -> Validation passed: {num_ebrs} EBRs, {len(table_dict['kpoints'])} k-points")
#         return True

#     def save_debug_html(self, sg_number, suffix=""):
#         """Save HTML for debugging purposes."""
#         try:
#             debug_dir = "debug_html"
#             os.makedirs(debug_dir, exist_ok=True)
#             filename = f"{debug_dir}/sg_{sg_number}_{suffix}_{int(time.time())}.html"
            
#             if self.driver:
#                 with open(filename, "w", encoding="utf-8") as f:
#                     f.write(self.driver.page_source)
#                 print(f"  -> Debug HTML saved to {filename}")
#         except Exception as e:
#             print(f"  -> Could not save debug HTML: {e}")

#     def convert_to_text_format(self, table_dict):
#         """Convert structured table data back to text format for database insertion."""
#         if not table_dict:
#             return ""
        
#         text_lines = []
        
#         if table_dict['wyckoff_positions']:
#             text_lines.append("Wyckoff pos. " + " ".join(table_dict['wyckoff_positions']))
        
#         if table_dict['band_representations']:
#             text_lines.append("Band-Rep. " + " ".join(table_dict['band_representations']))
        
#         if table_dict['decomposability']:
#             text_lines.append("Decomposable " + " ".join(table_dict['decomposability']))
        
#         # Add k-point data - fix the formatting issue
#         for kpoint, irreps in table_dict['kpoints'].items():
#             # Clean up the k-point label (remove the coordinate part for the line format)
#             kpoint_clean = kpoint.split(':')[0] if ':' in kpoint else kpoint
#             irreps_text = " ".join(irreps)
#             text_lines.append(f"{kpoint_clean}: {irreps_text}")
        
#         return "\n".join(text_lines)

#     def convert_to_database_format(self, table_dict):
#         """Convert table data to the exact format expected by the database."""
#         if not table_dict:
#             return ""
        
#         text_lines = []
        
#         # Header rows
#         if table_dict['wyckoff_positions']:
#             text_lines.append("Wyckoff pos. " + " ".join(table_dict['wyckoff_positions']))
        
#         if table_dict['band_representations']:
#             text_lines.append("Band-Rep. " + " ".join(table_dict['band_representations']))
        
#         if table_dict['decomposability']:
#             text_lines.append("Decomposable " + " ".join(table_dict['decomposability']))
        
#         # K-point data - ensure proper formatting for database
#         num_ebrs = len(table_dict['wyckoff_positions']) if table_dict['wyckoff_positions'] else 0
        
#         for kpoint_full, irreps in table_dict['kpoints'].items():
#             # Extract just the k-point label (e.g., "R" from "R:(1/2,1/2,1/2)")
#             kpoint_label = kpoint_full.split(':')[0] if ':' in kpoint_full else kpoint_full
            
#             # Process the irreps to match database expectations
#             processed_irreps = []
            
#             for irrep in irreps:
#                 # Handle cases like "R2R2(2)" - this should be treated as one irrep entry
#                 if irrep.strip():
#                     processed_irreps.append(irrep.strip())
            
#             # Ensure we have the right number of irrep columns
#             while len(processed_irreps) < num_ebrs:
#                 processed_irreps.append("")
            
#             # Truncate if we have too many (shouldn't happen but safety check)
#             processed_irreps = processed_irreps[:num_ebrs]
            
#             irreps_text = " ".join(processed_irreps)
#             text_lines.append(f"{kpoint_label}: {irreps_text}")
        
#         result = "\n".join(text_lines)
#         print(f"  -> Database format preview:\n{result}")
#         return result

#     def process_space_group(self, sg_number):
#         """Main method to process a single space group."""
#         table_data = self.extract_table_data_with_fallbacks(sg_number)
        
#         if not table_data:
#             return False
        
#         # Convert to database format 
#         formatted_text = self.convert_to_database_format(table_data)
        
#         if not formatted_text:
#             print(f"  -> ❌ Failed to format data for SG {sg_number}")
#             return False
        
#         # Insert into database
#         try:
#             self.db._parse_and_insert_from_text(sg_number, formatted_text)
#             return True
#         except Exception as e:
#             print(f"  -> ❌ Database insertion failed for SG {sg_number}: {e}")
#             print(f"  -> Problematic data:\n{formatted_text}")
#             return False

#     def run_scraper_for_range(self, start_sg, end_sg):
#         """Run the scraper for a range of space groups."""
#         print(f"Starting improved scraper for space groups {start_sg} to {end_sg}.")
        
#         successful = 0
#         failed = 0
#         failed_sgs = []
        
#         for sg in range(start_sg, end_sg + 1):
#             print(f"\n{'='*60}")
#             print(f"PROCESSING SPACE GROUP {sg}")
#             print(f"{'='*60}")
            
#             # Call the new, all-in-one function
#             if self.process_and_ingest_space_group(sg):
#                 successful += 1
#                 print(f"✅ Successfully processed and ingested SG {sg}")
#             else:
#                 failed += 1
#                 failed_sgs.append(sg)
#                 print(f"❌ Failed to process SG {sg}")
#             print(f"\n{'='*60}")
#             print(f"PROCESSING SPACE GROUP {sg}")
#             print(f"{'='*60}")
            
#             if self.process_space_group(sg):
#                 successful += 1
#                 print(f"✅ Successfully processed SG {sg}")
#             else:
#                 failed += 1
#                 failed_sgs.append(sg)
#                 print(f"❌ Failed to process SG {sg}")
            
#             # Adaptive pause between requests
#             if sg < end_sg:
#                 pause_time = random.uniform(12, 20)
#                 print(f"--- Pausing for {pause_time:.1f} seconds before next request ---")
#                 time.sleep(pause_time)
        
#         # Summary
#         print(f"\n{'='*60}")
#         print(f"SCRAPING COMPLETE")
#         print(f"Successful: {successful}, Failed: {failed}")
#         if failed_sgs:
#             print(f"Failed SGs: {failed_sgs}")
#         print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
#         print(f"{'='*60}")
        
#         self.close_driver()
#         return failed_sgs

#     def close_driver(self):
#         """Close the WebDriver and session."""
#         if self.driver:
#             try:
#                 self.driver.quit()
#                 print("WebDriver closed.")
#             except:
#                 pass
#             self.driver = None
        
#         if self.session:
#             try:
#                 self.session.close()
#                 print("Requests session closed.")
#             except:
#                 pass

# if __name__ == "__main__":
#     from PEBR_TR_nonmagnetic_query import EBRDatabaseManager
    
#     db_manager = EBRDatabaseManager()
#     scraper = ImprovedBCS_Scraper(db_manager)
    
#     try:
#         # Initial run
#         failed_sgs = scraper.run_scraper_for_range(12, 13)
        
#         # Retry failed space groups
#         if failed_sgs:
#             print(f"\n🔄 Retrying {len(failed_sgs)} failed space groups...")
#             time.sleep(30)  # Longer pause before retry
            
#             still_failed = []
#             for sg in failed_sgs:
#                 print(f"\n🔄 RETRY: Space Group {sg}")
#                 if scraper.process_space_group(sg):
#                     print(f"✅ Retry successful for SG {sg}")
#                 else:
#                     still_failed.append(sg)
#                     print(f"❌ Retry failed for SG {sg}")
#                 time.sleep(random.uniform(15, 25))
            
#             if still_failed:
#                 print(f"\n⚠️  Still failed after retry: {still_failed}")
#             else:
#                 print(f"\n🎉 All retries successful!")
        
#     except KeyboardInterrupt:
#         print("\nScraping interrupted by user")
#     except Exception as e:
#         print(f"\nFatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         scraper.close_driver()
#         db_manager.close()
import argparse
import random
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# It's good practice to install lxml (pip install lxml) and use it
# as it's generally faster and more lenient with broken HTML.
# But "html.parser" is a great built-in fallback.
try:
    import lxml
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"

# Make sure this import points to your actual database manager file
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
        # A single POST is sufficient and faster
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
        # --- END: FIX 1 ---

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
                    # Find the form to get the POST data needed for the decomposition page
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

        # --- START: FIX 2 ---
        # Robustly find the decomposition table.
        branch_table = None
        for table in soup.find_all("table"):
            # Check for header cells containing "branch"
            if table.find(['td', 'th'], string=lambda s: s and "branch" in s.lower()):
                branch_table = table
                break
        # --- END: FIX 2 ---

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
            inserted = self.db._insert_data(
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
                            info["ebr_id"], bi, branch_row[0], branch_row[1]
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

    db = EBRDatabaseManager() # Your DB Manager class
    scraper = BCSRequestsScraper(db)
    scraper.run(args.start, args.end)
    db.close()