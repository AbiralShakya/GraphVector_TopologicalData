import spglib
import pandas as pd
import httpx
import asyncio
import os
from bs4 import BeautifulSoup
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from pymatgen.core import Lattice, Structure
from typing import Dict, Optional, Tuple, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TopologicalMaterialAnalyzer:
    def __init__(self, csv_path: str, poscar_storage_dir: str = "./poscar_files"):
        """
        Initialize the analyzer with local dataset and storage directory.
        
        Args:
            csv_path: Path to the local materials database CSV
            poscar_storage_dir: Directory to store downloaded POSCAR files
        """
        self.csv_path = csv_path
        self.poscar_storage_dir = poscar_storage_dir
        self.materials_db = None
        self.load_local_database()
        
        # Create storage directory if it doesn't exist
        os.makedirs(poscar_storage_dir, exist_ok=True)
    
    def normalize_formula(self, formula: str) -> str:
        """
        Normalize formula by parsing elements and counts, then sorting alphabetically.
        
        Examples:
        - "Fe1 S3 U1" -> "Fe1S3U1"
        - "UFeS3" -> "Fe1S3U1"
        - "H2 Cu1 K1 O5 P1" -> "Cu1H2K1O5P1"
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Normalized formula string with elements sorted alphabetically
        """
        import re
        
        # Remove spaces and split into element-count pairs
        formula_clean = formula.replace(" ", "")
        
        # Find all element-count pairs using regex
        # Matches: Capital letter + optional lowercase + optional digits
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula_clean)
        
        # Create dictionary of element -> count
        element_counts = {}
        for element, count in matches:
            count = int(count) if count else 1
            element_counts[element] = element_counts.get(element, 0) + count
        
        # Sort elements alphabetically and reconstruct formula
        sorted_elements = sorted(element_counts.keys())
        normalized = ""
        for element in sorted_elements:
            count = element_counts[element]
            if count == 1:
                normalized += element + "1"
            else:
                normalized += element + str(count)
        
        return normalized
    
    def download_with_requests(self, url: str, filepath: str, max_retries: int = 3) -> Optional[str]:
        """Alternative download using requests with retry logic."""
        session = requests.Session()
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = session.get(url, headers=headers, verify=False, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            return filepath
            
        except Exception as e:
            print(f"Requests download failed: {e}")
            return None

    def load_local_database(self):
        """Load and process the local materials database CSV."""
        try:
            # Read CSV with proper handling of the header
            self.materials_db = pd.read_csv(self.csv_path)
            
            # Clean column names (remove extra spaces)
            self.materials_db.columns = self.materials_db.columns.str.strip()
            
            # Create a lookup dictionary using normalized formulas
            self.formula_lookup = {}
            self.original_formulas = {}  # Keep track of original formulas for reference
            
            for idx, row in self.materials_db.iterrows():
                original_formula = row['Formula'].strip() if pd.notna(row['Formula']) else ""
                normalized_formula = self.normalize_formula(original_formula)
                
                self.formula_lookup[normalized_formula] = {
                    'property': row['Property'],
                    'icsd_id': row['ICSD_ID'],
                    'poscar_link': row['POSCAR_link'],
                    'space_group': row['Space Group'],
                    'original_formula': original_formula
                }
                self.original_formulas[normalized_formula] = original_formula
            
            # print(f"Loaded {len(self.materials_db)} materials from local database")
            # print("Sample normalized formulas:")
            # for i, (norm_formula, orig_formula) in enumerate(list(self.original_formulas.items())[:5]):
            #     print(f"  '{orig_formula}' -> '{norm_formula}'")
            
        except Exception as e:
            print(f"Error loading local database: {e}")
            self.materials_db = None
    
    def classify_topology(self, property_str: str) -> str:
        """
        Classify the material based on the property string.
        
        Args:
            property_str: Property string from the database
            
        Returns:
            Classification: 'SM', 'TI', or 'trivial'
        """
        property_str = property_str.upper()
        
        if 'SM' in property_str or 'ESFD' in property_str:
            return 'SM'
        elif 'TI' in property_str or 'NLC' in property_str:
            return 'TI'
        elif 'TRIVIAL' in property_str or 'LCEBR' in property_str:
            return 'trivial'
        else:
            return 'unknown'
    
    def download_poscar(self, poscar_url: str, material_formula: str, icsd_id: str) -> Optional[str]:
        """
        Download POSCAR file from the given URL using httpx for faster performance.
        
        Args:
            poscar_url: URL to download POSCAR from
            material_formula: Formula of the material for filename
            icsd_id: ICSD ID for unique identification
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Create safe filename
            safe_formula = "".join(c if c.isalnum() else "_" for c in material_formula)
            filename = f"{safe_formula}_ICSD_{icsd_id}_POSCAR"
            filepath = os.path.join(self.poscar_storage_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                print(f"POSCAR already exists: {filename}")
                return filepath
            
            import certifi
            with httpx.Client(timeout=30.0, follow_redirects=True, verify=certifi.where()) as client:
                response = client.get(poscar_url)
                response.raise_for_status()
                
                # Check if response contains HTML (might be an error page)
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    # Parse HTML to extract POSCAR content or handle error
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for POSCAR content in <pre> tags (common for raw text display)
                    pre_tag = soup.find('pre')
                    if pre_tag:
                        poscar_content = pre_tag.get_text()
                    else:
                        # Look for code blocks or other text containers
                        code_tag = soup.find('code')
                        if code_tag:
                            poscar_content = code_tag.get_text()
                        else:
                            # If no structured content found, try to extract from body
                            body = soup.find('body')
                            if body:
                                poscar_content = body.get_text().strip()
                            else:
                                poscar_content = response.text
                    
                    # Basic validation - POSCAR should have multiple lines
                    if len(poscar_content.split('\n')) < 5:
                        raise ValueError("Downloaded content doesn't look like a valid POSCAR file")
                        
                else:
                    # Direct text content (ideal case)
                    poscar_content = response.text
                
                # Save the file
                with open(filepath, 'w') as f:
                    f.write(poscar_content)
                
                print(f"Downloaded POSCAR: {filename}")
                return filepath
                
        except Exception as e:
            print(f"Error downloading POSCAR from {poscar_url}: {e}")
            return None
    
    async def download_poscar_async(self, poscar_url: str, material_formula: str, icsd_id: str) -> Optional[str]:
        """
        Asynchronously download POSCAR file for batch processing.
        
        Args:
            poscar_url: URL to download POSCAR from
            material_formula: Formula of the material for filename
            icsd_id: ICSD ID for unique identification
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Create safe filename
            safe_formula = "".join(c if c.isalnum() else "_" for c in material_formula)
            filename = f"{safe_formula}_ICSD_{icsd_id}_POSCAR"
            filepath = os.path.join(self.poscar_storage_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                return filepath
            
            # Use httpx async client for concurrent downloads
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(poscar_url)
                response.raise_for_status()
                
                # Handle HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    pre_tag = soup.find('pre')
                    poscar_content = pre_tag.get_text() if pre_tag else response.text
                else:
                    poscar_content = response.text
                
                # Save the file
                with open(filepath, 'w') as f:
                    f.write(poscar_content)
                
                return filepath
                
        except Exception as e:
            print(f"Error downloading POSCAR from {poscar_url}: {e}")
            return None
    
    def analyze_material(self, jvasp_id: str) -> Dict:
        """
        Analyze a single material from JARVIS database.
        
        Args:
            jvasp_id: JARVIS ID (e.g., "JVASP-125448")
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'jvasp_id': jvasp_id,
            'formula': None,
            'is_topological': False,
            'topology_class': None,
            'poscar_path': None,
            'magnetic_info': {},
            'error': None
        }
        
        try:
            # Get JARVIS data
            print(f"\nAnalyzing {jvasp_id}...")
            entry = get_jid_data(jid=jvasp_id, dataset="dft_3d")
            
            if not entry:
                results['error'] = "Failed to retrieve JARVIS data"
                return results
            
            formula = entry.get('formula', '')
            results['formula'] = formula
            
            print(f"Formula: {formula}")
            
            # Normalize both JARVIS formula and check against database
            normalized_jarvis_formula = self.normalize_formula(formula)
            print(f"Normalized JARVIS formula: {normalized_jarvis_formula}")
            
            # Check if material is in local topological database
            if normalized_jarvis_formula not in self.formula_lookup:
                print("Material not found in topological database - assuming trivial/non-topological")
                results['is_topological'] = False
                return results
            
            # Material is in the database - get topological info
            material_info = self.formula_lookup[normalized_jarvis_formula]
            results['is_topological'] = True
            results['topology_class'] = self.classify_topology(material_info['property'])
            
            print(f"Found match! Original database formula: {material_info['original_formula']}")
            print(f"Topological classification: {results['topology_class']}")
            print(f"Property: {material_info['property']}")
            
            # Download POSCAR file
            poscar_url = material_info['poscar_link']
            if poscar_url and pd.notna(poscar_url):
                poscar_path = self.download_poscar(
                    poscar_url, 
                    formula, 
                    str(material_info['icsd_id'])
                )
                results['poscar_path'] = poscar_path
            
            # Perform magnetic symmetry analysis
            magnetic_info = self.get_magnetic_symmetry_info(entry)
            results['magnetic_info'] = magnetic_info
            
            # Print summary
            print(f"Space Group: {entry.get('spg_symbol')} ({entry.get('spg_number')})")
            if magnetic_info.get('bns_number'):
                print(f"BNS number: {magnetic_info['bns_number']}")
                print(f"MSG type: {magnetic_info.get('msg_type')}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error analyzing {jvasp_id}: {e}")
        
        return results
    
    def get_magnetic_symmetry_info(self, entry: Dict) -> Dict:
        """
        Extract magnetic symmetry information from JARVIS entry.
        
        Args:
            entry: JARVIS database entry
            
        Returns:
            Dictionary with magnetic symmetry info
        """
        magnetic_info = {}
        
        try:
            atoms = Atoms.from_dict(entry["atoms"])
            latt = Lattice(atoms.lattice_mat)
            struct = Structure(latt, atoms.elements, atoms.coords)
            
            # Get magnetic moments
            magmoms = entry.get("magmom_outcar") or entry.get("magmom_oszicar")
            if isinstance(magmoms, (int, float)):
                magmoms = [magmoms] * len(struct)
            
            if magmoms:
                lattice = struct.lattice.matrix
                positions = struct.frac_coords
                numbers = [site.specie.Z for site in struct]
                magmoms_vec = [[0, 0, m] for m in magmoms]
                cell = (lattice, positions, numbers, magmoms_vec)
                
                # Get magnetic symmetry dataset
                mag_dataset = spglib.get_magnetic_symmetry_dataset(cell, symprec=1e-5)
                
                if mag_dataset is not None:
                    magnetic_info['uni_number'] = mag_dataset.uni_number
                    magnetic_info['hall_number'] = mag_dataset.hall_number
                    magnetic_info['msg_type'] = mag_dataset.msg_type
                    
                    # Get additional magnetic space group info
                    from spglib import get_magnetic_spacegroup_type
                    msg_info = get_magnetic_spacegroup_type(mag_dataset.uni_number)
                    magnetic_info['bns_number'] = msg_info.bns_number
                    magnetic_info['og_number'] = msg_info.og_number
                
        except Exception as e:
            magnetic_info['error'] = str(e)
        
        return magnetic_info
    
    async def analyze_multiple_materials_async(self, jvasp_ids: List[str]) -> Dict:
        """
        Analyze multiple materials concurrently with async POSCAR downloads.
        
        Args:
            jvasp_ids: List of JARVIS IDs
            
        Returns:
            Dictionary with results and statistics
        """
        results = []
        stats = {'total': 0, 'topological': 0, 'SM': 0, 'TI': 0, 'trivial': 0, 'errors': 0}
        
        # First, get all the JARVIS data and identify topological materials
        topological_materials = []
        
        for jvasp_id in jvasp_ids:
            try:
                print(f"\nAnalyzing {jvasp_id}...")
                entry = get_jid_data(jid=jvasp_id, dataset="dft_3d")
                
                if not entry:
                    results.append({
                        'jvasp_id': jvasp_id,
                        'error': "Failed to retrieve JARVIS data"
                    })
                    stats['errors'] += 1
                    continue
                
                formula = entry.get('formula', '')
                normalized_formula = self.normalize_formula(formula)
                
                result = {
                    'jvasp_id': jvasp_id,
                    'formula': formula,
                    'is_topological': False,
                    'topology_class': None,
                    'poscar_path': None,
                    'magnetic_info': {},
                    'error': None
                }
                
                if normalized_formula in self.formula_lookup:
                    material_info = self.formula_lookup[normalized_formula]
                    result['is_topological'] = True
                    result['topology_class'] = self.classify_topology(material_info['property'])
                    result['magnetic_info'] = self.get_magnetic_symmetry_info(entry)
                    
                    # Store info for async download
                    topological_materials.append({
                        'result': result,
                        'poscar_url': material_info['poscar_link'],
                        'icsd_id': str(material_info['icsd_id'])
                    })
                    
                    print(f"Found topological material: {formula} -> {result['topology_class']}")
                else:
                    print(f"Non-topological material: {formula}")
                
                results.append(result)
                stats['total'] += 1
                
            except Exception as e:
                results.append({
                    'jvasp_id': jvasp_id,
                    'error': str(e)
                })
                stats['errors'] += 1
        
        # Now download all POSCAR files concurrently
        if topological_materials:
            print(f"\nDownloading {len(topological_materials)} POSCAR files concurrently...")
            
            download_tasks = []
            for mat in topological_materials:
                if mat['poscar_url'] and pd.notna(mat['poscar_url']):
                    task = self.download_poscar_async(
                        mat['poscar_url'],
                        mat['result']['formula'],
                        mat['icsd_id']
                    )
                    download_tasks.append((task, mat['result']))
            
            # Execute all downloads concurrently
            download_results = await asyncio.gather(*[task for task, _ in download_tasks], return_exceptions=True)
            
            # Update results with download paths
            for (task, result), download_result in zip(download_tasks, download_results):
                if isinstance(download_result, Exception):
                    print(f"Download failed for {result['formula']}: {download_result}")
                else:
                    result['poscar_path'] = download_result
        
        # Calculate final statistics
        for result in results:
            if result.get('is_topological') and not result.get('error'):
                stats['topological'] += 1
                if result.get('topology_class'):
                    stats[result['topology_class']] += 1
        
        print(f"\n=== Analysis Summary ===")
        print(f"Total materials analyzed: {stats['total']}")
        print(f"Topological materials: {stats['topological']}")
        print(f"  - Semimetals (SM): {stats['SM']}")
        print(f"  - Topological Insulators (TI): {stats['TI']}")
        print(f"  - Trivial: {stats['trivial']}")
        print(f"Non-topological materials: {stats['total'] - stats['topological'] - stats['errors']}")
        print(f"Errors: {stats['errors']}")
        
        return {'results': results, 'statistics': stats}

# Example usage
def main():
    # Initialize analyzer with your CSV path
    csv_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/materials_database.csv"
    analyzer = TopologicalMaterialAnalyzer(csv_path)
    
    # Analyze single material
    jvasp_input = "JVASP-14202"
    result = analyzer.analyze_material(jvasp_input)
    result2 = analyzer.download_with_requests(jvasp_input)
    print(result2)
    
    # Analyze multiple materials (example)
    # jvasp_ids = ["JVASP-125448", "JVASP-12345", "JVASP-67890"]
    # results = analyzer.analyze_multiple_materials(jvasp_ids)

if __name__ == "__main__":
    main()