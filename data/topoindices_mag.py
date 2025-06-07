import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

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
                        INSERT OR IGNORE INTO topological_indices
                          (space_group_id, space_group_symbol, index_name, mod_value)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        space_group_id,
                        space_group_symbol,
                        idx['index_name'],
                        idx['mod_value']
                    ))
                    topo_id = c.lastrowid
                    if topo_id:
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
    Scrapes double‐valued topological indices for magnetic space groups
    from the Bilbao server.
    """
    def __init__(self,
                 bns_filepath: str,
                 base_url: str = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/topologicalindices.pl"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.all_bns = self._load_bns_numbers(bns_filepath)
    
    def _load_bns_numbers(self, filepath: str) -> List[str]:
        nums: List[str] = []
        with open(filepath, 'r') as f:
            for L in f:
                m = re.search(r"BNS:\s*([\d\.]+)", L)
                if m:
                    nums.append(m.group(1))
        logger.info(f"🔢 Loaded {len(nums)} magnetic BNS IDs from {filepath}")
        return nums
    
    def _parse_table(self, table) -> List[Dict]:
        """Turn the HTML <table> into our list-of-dicts format."""
        rows = table.find_all('tr')
        if len(rows) < 2:
            return []
        # second row holds irreps names
        headers = [td.get_text(strip=True) for td in rows[1].find_all('td')[1:]]
        
        out: List[Dict] = []
        for row in rows[2:]:
            cols = row.find_all('td')
            if not cols:
                continue
            idx_name = cols[0].get_text(strip=True)
            mod_txt = cols[-1].get_text(strip=True)
            mod_val = int(mod_txt) if mod_txt.isdigit() else None
            irr_map: Dict[str,str] = {}
            for i, ir in enumerate(headers):
                irr_map[ir] = cols[i+1].get_text(strip=True)
            out.append({
                'index_name': idx_name,
                'mod_value': mod_val,
                'irreps': irr_map
            })
        return out
    
    def get_double_irreps_data(self,
                               space_group_id: str
                              ) -> Optional[Tuple[str, List[Dict]]]:
        """
        1) Fetch the blank form (?tipog=gmag)
        2) Parse its <form> action + all hidden/text/radio inputs
        3) Populate the 'magn' field + pick the 'double' radio
        4) GET the real action URL with those params
        5) Scrape the title & table
        """
        # 1) fetch blank form
        form_url = f"{self.base_url}?tipog=gmag"
        r1 = self.session.get(form_url, timeout=30)
        r1.raise_for_status()
        soup1 = BeautifulSoup(r1.content, 'html.parser')
        
        form = soup1.find('form')
        if not form:
            logger.error("No <form> on blank page – HTML dump:\n%s", r1.text[:300])
            return None
        
        # 2) figure out where to send the request
        action = form.get('action')
        method = form.get('method','GET').upper()
        scrape_url = urljoin(form_url, action)
        logger.debug("Form method=%s, action=%s → %s", method, action, scrape_url)
        
        # 3) collect all default inputs
        params: Dict[str,str] = {}
        for inp in form.find_all('input'):
            name = inp.get('name')
            if not name:
                continue
            val  = inp.get('value','')
            typ  = inp.get('type', 'text')
            
            if typ == 'radio':
                # choose the “double‐valued irreps” option
                if 'double' in val.lower():
                    params[name] = val
            else:
                # hidden or text → take default
                params[name] = val
        
        # override our BNS
        params['magn'] = space_group_id
        
        # 4) submit
        if method == 'GET':
            r2 = self.session.get(scrape_url, params=params, timeout=30)
        else:
            r2 = self.session.post(scrape_url, data=params, timeout=30)
        r2.raise_for_status()
        logger.info("Fetched SG %s → %s", space_group_id, r2.url)
        
        # 5) scrape title + table
        soup2 = BeautifulSoup(r2.content, 'html.parser')
        title = soup2.find(
            lambda t: t.name in ('h1','h2','h3') 
                      and 'topological indices' in t.get_text().lower()
        )
        if not title:
            logger.warning("No title on results page for SG %s", space_group_id)
            return None
        
        text = title.get_text()
        m = re.search(r'\((.*?)\)', text)
        symbol = m.group(1).strip() if m else f"BNS {space_group_id}"
        
        table = soup2.find('table')
        if not table:
            logger.info("No data table for SG %s", space_group_id)
            return symbol, []
        
        data = self._parse_table(table)
        logger.info("Parsed %d indices for SG %s (%s)",
                    len(data), space_group_id, symbol)
        return symbol, data
    
    def scrape_all_groups(self, delay: float = 1.0):
        db = MagneticIndicesDatabase()
        count = 0
        for bns in self.all_bns:
            logger.info("→ Processing BNS %s …", bns)
            out = self.get_double_irreps_data(bns)
            if out:
                sym, dat = out
                if dat:
                    db.insert_data(bns, sym, dat)
                    count += 1
            time.sleep(delay)
        logger.info("Finished: stored data for %d/%d groups",
                    count, len(self.all_bns))


def main():
    bns_file = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/magnetic_table_bns.txt"
    scraper = MagneticIndicesScraper(bns_file)
    logger.info("Starting full scrape …")
    scraper.scrape_all_groups(delay=0.2)
    
    # example:
    db = MagneticIndicesDatabase()
    results = db.get_space_group_data("2.5")
    print("\nResults for SG 2.5:")
    for idx in results:
        print(f" • {idx['index_name']} (mod {idx['mod_value']}): "
              f"{idx['irreps']}")


if __name__ == "__main__":
    main()
