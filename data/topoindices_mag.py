import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MagneticIndicesDatabase:
    """Handles the SQLite database for storing topological indices."""
    
    def __init__(self, db_path: str = "topological_indices_magnetic.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS topological_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    space_group_id TEXT NOT NULL,
                    space_group_symbol TEXT,
                    index_name TEXT NOT NULL,
                    mod_value INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(space_group_id, index_name)
                )''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS irrep_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topo_index_id INTEGER,
                    irrep_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (topo_index_id) 
                        REFERENCES topological_indices (id)
                )''')
            c.execute('CREATE INDEX IF NOT EXISTS '
                      'idx_space_group ON topological_indices(space_group_id)')
            conn.commit()
    
    def insert_data(self,
                    space_group_id: str,
                    space_group_symbol: str,
                    indices_data: List[Dict]):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            try:
                for idx in indices_data:
                    c.execute('''
                        INSERT OR REPLACE INTO topological_indices
                          (space_group_id, space_group_symbol, index_name, mod_value)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        space_group_id,
                        space_group_symbol,
                        idx['index_name'],
                        idx['mod_value']
                    ))
                    
                    # Get the row ID
                    c.execute('''
                        SELECT id FROM topological_indices 
                        WHERE space_group_id=? AND index_name=?
                    ''', (space_group_id, idx['index_name']))
                    
                    result = c.fetchone()
                    if result:
                        topo_id = result[0]
                        
                        # Clear existing irrep values for this index
                        c.execute('DELETE FROM irrep_values WHERE topo_index_id=?', (topo_id,))
                        
                        # Insert new irrep values
                        rows = [
                            (topo_id, name, val)
                            for name, val in idx['irreps'].items()
                        ]
                        if rows:
                            c.executemany('''
                                INSERT INTO irrep_values
                                  (topo_index_id, irrep_name, value)
                                VALUES (?, ?, ?)
                            ''', rows)
                            
                conn.commit()
                logger.info(f"✅ Inserted SG {space_group_id} [{space_group_symbol}] "
                            f"({len(indices_data)} indices)")
            except Exception:
                conn.rollback()
                logger.exception(f"❌ Failed inserting SG {space_group_id}")
                raise
    
    def get_space_group_data(self, space_group_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT ti.index_name, ti.mod_value, ti.space_group_symbol,
                       iv.irrep_name, iv.value
                  FROM topological_indices ti
             LEFT JOIN irrep_values iv
                    ON ti.id = iv.topo_index_id
                 WHERE ti.space_group_id = ?
              ORDER BY ti.index_name, iv.irrep_name
            ''', (space_group_id,))
            rows = c.fetchall()
        
        out: Dict[str, Dict] = {}
        for name, modv, sym, irrep, val in rows:
            if name not in out:
                out[name] = {
                    'index_name': name,
                    'mod_value': modv,
                    'space_group_symbol': sym,
                    'irreps': {}
                }
            if irrep:
                out[name]['irreps'][irrep] = val
        return list(out.values())


class MagneticIndicesScraper:
    """
    Scrapes double-valued topological indices for magnetic space groups
    from the Bilbao server.
    """
    def __init__(self,
                 bns_filepath: str,
                 base_url: str = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/topologicalindices.pl"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.all_bns = self._load_bns_numbers(bns_filepath)
    
    def _load_bns_numbers(self, filepath: str) -> List[str]:
        nums: List[str] = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Try different patterns
                    m = re.search(r"BNS:\s*([\d\.]+)", line)
                    if m:
                        nums.append(m.group(1))
                    elif re.match(r"^[\d\.]+$", line):
                        nums.append(line)
            logger.info(f"🔢 Loaded {len(nums)} magnetic BNS IDs from {filepath}")
            return nums
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return []
    
    def _parse_table(self, table) -> List[Dict]:
        """Turn the HTML <table> into our list-of-dicts format."""
        rows = table.find_all('tr')
        if len(rows) < 2:
            logger.debug("Table has fewer than 2 rows")
            return []
        
        # Find header row - usually first or second row
        headers = []
        header_row_idx = -1
        
        for i, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
            
            # Check if this looks like a header row
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if any(text and ('irrep' in text.lower() or 'rep' in text.lower() or len(text) < 10) for text in cell_texts[1:]):
                headers = cell_texts[1:]  # Skip first column (index names)
                header_row_idx = i
                break
        
        if not headers:
            logger.warning("Could not find header row in table")
            return []
        
        logger.debug(f"Found headers: {headers}")
        
        out: List[Dict] = []
        for row_idx, row in enumerate(rows[header_row_idx + 1:], header_row_idx + 1):
            cols = row.find_all(['td', 'th'])
            if len(cols) < 2:
                continue
                
            idx_name = cols[0].get_text(strip=True)
            if not idx_name:
                continue
            
            # Look for mod value - usually last column
            mod_val = None
            mod_txt = cols[-1].get_text(strip=True)
            if mod_txt.isdigit():
                mod_val = int(mod_txt)
            
            # Extract irrep values
            irr_map: Dict[str, str] = {}
            for i, header in enumerate(headers):
                if i + 1 < len(cols):
                    val = cols[i + 1].get_text(strip=True)
                    if val and val != mod_txt:  # Don't include mod value as irrep
                        irr_map[header] = val
            
            if irr_map:  # Only add if we have irrep data
                out.append({
                    'index_name': idx_name,
                    'mod_value': mod_val,
                    'irreps': irr_map
                })
        
        logger.debug(f"Parsed {len(out)} indices from table")
        return out
    
    def get_double_irreps_data(self, space_group_id: str) -> Optional[Tuple[str, List[Dict]]]:
        """
        Fetch topological indices for a magnetic space group.
        """
        try:
            # Use POST method as shown in the original HTML form
            post_data = {
                'tipog': 'gmag',
                'super': space_group_id,
                'double': 'Submit'  # This corresponds to the "Submit" button for double irreps
            }
            
            logger.debug(f"Sending POST request for SG {space_group_id}")
            response = self.session.post(self.base_url, data=post_data, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Fetched SG {space_group_id} → {response.url}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the title/heading
            space_group_symbol = f"BNS {space_group_id}"
            title_tags = soup.find_all(['h1', 'h2', 'h3', 'title'])
            for tag in title_tags:
                text = tag.get_text()
                if 'topological' in text.lower():
                    # Try to extract space group symbol from parentheses
                    m = re.search(r'\((.*?)\)', text)
                    if m:
                        space_group_symbol = m.group(1).strip()
                    break
            
            # Look for data table
            tables = soup.find_all('table')
            data = []
            
            for table in tables:
                # Skip navigation/header tables
                if table.find('a') or 'signature' in str(table.get('class', [])):
                    continue
                
                parsed_data = self._parse_table(table)
                if parsed_data:
                    data.extend(parsed_data)
                    break
            
            if not data:
                logger.info(f"No topological indices data found for SG {space_group_id}")
                # Check if there's an error message
                error_indicators = soup.find_all(text=re.compile(r'(error|not found|invalid)', re.I))
                if error_indicators:
                    logger.warning(f"Possible error for SG {space_group_id}: {error_indicators[0]}")
            
            logger.info(f"Parsed {len(data)} indices for SG {space_group_id} ({space_group_symbol})")
            return space_group_symbol, data
            
        except requests.RequestException as e:
            logger.error(f"Request failed for SG {space_group_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for SG {space_group_id}: {e}")
            return None
    
    def scrape_all_groups(self, delay: float = 1.0):
        db = MagneticIndicesDatabase()
        count = 0
        total = len(self.all_bns)
        
        for i, bns in enumerate(self.all_bns, 1):
            logger.info(f"→ Processing BNS {bns} ({i}/{total}) ...")
            
            result = self.get_double_irreps_data(bns)
            if result:
                symbol, data = result
                if data:
                    db.insert_data(bns, symbol, data)
                    count += 1
                else:
                    logger.info(f"No data found for SG {bns}")
            
            if i < total:  # Don't sleep after the last request
                time.sleep(delay)
        
        logger.info(f"Finished: stored data for {count}/{total} groups")


def main():
    # Update this path to your BNS file
    bns_file = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/magnetic_table_bns.txt"
    
    try:
        scraper = MagneticIndicesScraper(bns_file)
        if not scraper.all_bns:
            logger.error("No BNS numbers loaded. Check your file format.")
            return
            
        logger.info(f"Starting scrape of {len(scraper.all_bns)} magnetic space groups...")
        scraper.scrape_all_groups(delay=0.5) 
        
        # Example: show results for first group
        if scraper.all_bns:
            db = MagneticIndicesDatabase()
            test_sg = scraper.all_bns[0]
            results = db.get_space_group_data(test_sg)
            print(f"\nResults for SG {test_sg}:")
            for idx in results:
                print(f" • {idx['index_name']} (mod {idx['mod_value']}): "
                      f"{idx['irreps']}")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")


if __name__ == "__main__":
    main()