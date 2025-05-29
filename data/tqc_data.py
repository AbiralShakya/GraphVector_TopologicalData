# there is no api :(
# build webscraper.. ?
# also analyze https://cryst.ehu.es/#solidtop for further classification recommendations

import time
import json
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re

class TopologicalQuantumChemistryScraper:
    def __init__(self, headless=True):
        """Initialize the scraper with Chrome webdriver"""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
        self.driver = None
        self.base_url = "https://topologicalquantumchemistry.org/#/"
        
    def start_driver(self):
        """Start the Chrome webdriver"""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        
    def close_driver(self):
        """Close the webdriver"""
        if self.driver:
            self.driver.quit()
            
    def wait_for_element(self, locator, timeout=10):
        """Wait for an element to be present"""
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
        
    def wait_for_clickable(self, locator, timeout=10):
        """Wait for an element to be clickable"""
        return WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
        
    def search_by_element(self, element_symbol):
        """Search for materials containing a specific element"""
        try:
            # Navigate to the main page
            self.driver.get(self.base_url)
            time.sleep(3)
            
            # Find and click on the element in the periodic table
            element_xpath = f"//div[contains(@class, 'element') or contains(@class, 'cell')]//text()[normalize-space()='{element_symbol}']/parent::*"
            element_button = self.wait_for_clickable((By.XPATH, element_xpath))
            element_button.click()
            
            # Wait for search results to load
            time.sleep(2)
            
            # Click search button if it exists
            try:
                search_button = self.wait_for_clickable((By.XPATH, "//button[contains(text(), 'Search') or contains(@class, 'search')]"))
                search_button.click()
                time.sleep(3)
            except TimeoutException:
                print("Search button not found, proceeding...")
                
            return True
            
        except Exception as e:
            print(f"Error searching for element {element_symbol}: {str(e)}")
            return False
            
    def scrape_search_results(self):
        """Scrape the search results page for materials"""
        materials = []
        
        try:
            # Wait for results to load
            time.sleep(3)
            
            # Find all material entries in the results
            # Look for common patterns in the results table/list
            material_rows = self.driver.find_elements(By.XPATH, "//tr[contains(@class, 'material') or contains(@class, 'entry') or td]")
            
            if not material_rows:
                # Try alternative selectors
                material_rows = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'result') or contains(@class, 'material')]")
            
            print(f"Found {len(material_rows)} material entries")
            
            for row in material_rows:
                try:
                    material_data = {}
                    
                    # Extract compound name
                    compound_elem = row.find_elements(By.XPATH, ".//td[1] | .//div[1] | .//*[contains(@class, 'compound')]")
                    if compound_elem:
                        material_data['compound'] = compound_elem[0].text.strip()
                    
                    # Extract symmetry group
                    symmetry_elem = row.find_elements(By.XPATH, ".//td[2] | .//*[contains(@class, 'symmetry')]")
                    if symmetry_elem:
                        material_data['symmetry_group'] = symmetry_elem[0].text.strip()
                    
                    # Extract topological indices
                    topo_elem = row.find_elements(By.XPATH, ".//td[3] | .//*[contains(@class, 'topological')]")
                    if topo_elem:
                        material_data['topological_indices'] = topo_elem[0].text.strip()
                    
                    # Extract crossing type
                    crossing_elem = row.find_elements(By.XPATH, ".//td[4] | .//*[contains(@class, 'crossing')]")
                    if crossing_elem:
                        material_data['crossing_type'] = crossing_elem[0].text.strip()
                    
                    # Extract material type
                    type_elem = row.find_elements(By.XPATH, ".//td[5] | .//*[contains(@class, 'type')]")
                    if type_elem:
                        material_data['type'] = type_elem[0].text.strip()
                    
                    # Look for clickable links to detail pages
                    detail_links = row.find_elements(By.XPATH, ".//a[contains(@href, 'detail') or contains(@href, '#')]")
                    if detail_links:
                        href = detail_links[0].get_attribute('href')
                        material_data['detail_url'] = href
                        
                        # Extract ICSD number from URL if present
                        if 'detail' in href:
                            icsd_match = re.search(r'detail/(\d+)', href)
                            if icsd_match:
                                material_data['icsd_number'] = icsd_match.group(1)
                    
                    if material_data and ('compound' in material_data or len(material_data) > 1):
                        materials.append(material_data)
                        
                except Exception as e:
                    print(f"Error extracting data from row: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping search results: {str(e)}")
            
        return materials
        
    def scrape_material_details(self, detail_url):
        """Scrape detailed information for a specific material"""
        try:
            self.driver.get(detail_url)
            time.sleep(3)
            
            material_details = {}
            
            # Extract basic information
            compound_elem = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Compound')]/following-sibling::* | //*[contains(@class, 'compound')]")
            if compound_elem:
                material_details['compound'] = compound_elem[0].text.strip()
            
            # Extract crystallographic data
            try:
                # Cell parameters
                cell_length_a = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Cell Length A')]/following-sibling::* | //*[contains(text(), 'a =')]")
                if cell_length_a:
                    material_details['cell_length_a'] = cell_length_a[0].text.strip()
                    
                cell_length_b = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Cell Length B')]/following-sibling::* | //*[contains(text(), 'b =')]")
                if cell_length_b:
                    material_details['cell_length_b'] = cell_length_b[0].text.strip()
                    
                cell_length_c = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Cell Length C')]/following-sibling::* | //*[contains(text(), 'c =')]")
                if cell_length_c:
                    material_details['cell_length_c'] = cell_length_c[0].text.strip()
                    
                # Cell angles
                cell_angles = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Cell Angle')]/following-sibling::*")
                for angle in cell_angles:
                    text = angle.text.strip()
                    if 'α' in text or 'alpha' in text.lower():
                        material_details['cell_angle_alpha'] = text
                    elif 'β' in text or 'beta' in text.lower():
                        material_details['cell_angle_beta'] = text
                    elif 'γ' in text or 'gamma' in text.lower():
                        material_details['cell_angle_gamma'] = text
                        
                # Cell volume
                cell_volume = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Cell Volume')]/following-sibling::*")
                if cell_volume:
                    material_details['cell_volume'] = cell_volume[0].text.strip()
                    
            except Exception as e:
                print(f"Error extracting crystallographic data: {str(e)}")
                
            # Extract topological information
            try:
                symmetry_group = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Symmetry Group')]/following-sibling::*")
                if symmetry_group:
                    material_details['symmetry_group'] = symmetry_group[0].text.strip()
                    
                topological_indices = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Topological Indices')]/following-sibling::*")
                if topological_indices:
                    material_details['topological_indices'] = topological_indices[0].text.strip()
                    
            except Exception as e:
                print(f"Error extracting topological data: {str(e)}")
                
            return material_details
            
        except Exception as e:
            print(f"Error scraping material details from {detail_url}: {str(e)}")
            return {}
            
    def scrape_element_materials(self, element_symbol, include_details=False):
        """Main method to scrape all materials for a given element"""
        all_materials = []
        
        try:
            self.start_driver()
            
            # Search for the element
            if not self.search_by_element(element_symbol):
                return all_materials
                
            # Scrape search results
            materials = self.scrape_search_results()
            print(f"Found {len(materials)} materials for element {element_symbol}")
            
            # Optionally scrape detailed information
            if include_details:
                for i, material in enumerate(materials):
                    if 'detail_url' in material:
                        print(f"Scraping details for material {i+1}/{len(materials)}")
                        details = self.scrape_material_details(material['detail_url'])
                        material.update(details)
                        time.sleep(1)  # Be respectful to the server
                        
            all_materials = materials
            
        except Exception as e:
            print(f"Error in scrape_element_materials: {str(e)}")
            
        finally:
            self.close_driver()
            
        return all_materials
        
    def save_to_csv(self, materials, filename):
        """Save materials data to CSV file"""
        if not materials:
            print("No materials to save")
            return
            
        df = pd.DataFrame(materials)
        df.to_csv(filename, index=False)
        print(f"Saved {len(materials)} materials to {filename}")
        
    def save_to_json(self, materials, filename):
        """Save materials data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(materials, f, indent=2)
        print(f"Saved {len(materials)} materials to {filename}")

# Example usage
if __name__ == "__main__":
    scraper = TopologicalQuantumChemistryScraper(headless=False)  # Set to True for headless mode
    
    # Scrape materials for Scandium (Sc)
    element = "Sc"
    materials = scraper.scrape_element_materials(element, include_details=True)
    
    if materials:
        # Save results
        scraper.save_to_csv(materials, f"{element}_materials.csv")
        scraper.save_to_json(materials, f"{element}_materials.json")
        
        print(f"\nSample material data:")
        for material in materials[:3]:  # Show first 3 materials
            print(json.dumps(material, indent=2))
    else:
        print(f"No materials found for element {element}")