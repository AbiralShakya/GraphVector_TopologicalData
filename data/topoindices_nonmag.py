import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ——— Database code (unchanged) ———
class TopologicalIndicesDatabase:
    def __init__(self, db_path: str = "topological_indices_nonmagnetic.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS topological_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_group_id INTEGER NOT NULL,
                space_group_symbol TEXT,
                index_name TEXT NOT NULL,
                mod_value INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS irrep_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topo_index_id INTEGER,
                irrep_name TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (topo_index_id) REFERENCES topological_indices(id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_space_group ON topological_indices(space_group_id)")
        conn.commit()
        conn.close()

    def insert_data(self,
                    space_group_id: int,
                    space_group_symbol: str,
                    indices_data: List[Dict]):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            for idx in indices_data:
                c.execute("""
                    INSERT INTO topological_indices
                    (space_group_id, space_group_symbol, index_name, mod_value)
                    VALUES (?, ?, ?, ?)
                """, (
                    space_group_id,
                    space_group_symbol,
                    idx['index_name'],
                    idx['mod_value']
                ))
                topo_id = c.lastrowid
                for name, val in idx['irreps'].items():
                    c.execute("""
                        INSERT INTO irrep_values (topo_index_id, irrep_name, value)
                        VALUES (?, ?, ?)
                    """, (topo_id, name, val))
            conn.commit()
            logger.info(f"Inserted SG {space_group_id} ({space_group_symbol}) "
                        f"→ {len(indices_data)} indices")
        except Exception:
            conn.rollback()
            logger.exception(f"Failed inserting SG {space_group_id}")
            raise
        finally:
            conn.close()

    def get_space_group_data(self, space_group_id: int) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT ti.index_name, ti.mod_value, ti.space_group_symbol,
                   iv.irrep_name, iv.value
              FROM topological_indices ti
         LEFT JOIN irrep_values iv
                ON ti.id = iv.topo_index_id
             WHERE ti.space_group_id = ?
          ORDER BY ti.index_name, iv.irrep_name
        """, (space_group_id,))
        rows = c.fetchall()
        conn.close()

        out: Dict[str, Dict] = {}
        for idx_name, modv, sym, irr, val in rows:
            if idx_name not in out:
                out[idx_name] = {
                    'index_name': idx_name,
                    'mod_value': modv,
                    'space_group_symbol': sym,
                    'irreps': {}
                }
            if irr:
                out[idx_name]['irreps'][irr] = val
        return list(out.values())

# ——— Scraper code ———
class TopologicalIndicesScraper:
    def __init__(self,
                 base_url: str = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/topologicalindices.pl"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })

    def _parse_table(self, table) -> List[Dict]:
        rows = table.find_all('tr')
        if len(rows) < 2:
            return []
        # second row holds the irrep names (minus the final “mod” cell)
        irrep_names = [td.get_text(strip=True)
                       for td in rows[1].find_all('td')[:-1]]

        out = []
        for row in rows[2:]:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            idx_name = cols[0].get_text(strip=True)
            mod_txt  = cols[-1].get_text(strip=True)
            try:
                mod_val = int(mod_txt)
            except ValueError:
                mod_val = None

            irr_map = {}
            for i, name in enumerate(irrep_names):
                irr_map[name] = cols[i+1].get_text(strip=True)

            out.append({
                'index_name': idx_name,
                'mod_value':  mod_val,
                'irreps':     irr_map
            })
        return out

    def get_double_irreps_data(self,
                               space_group_id: int
                              ) -> Optional[Tuple[str, List[Dict]]]:
        """
        1) GET the blank form (same URL, no params)
        2) Parse its <form> tag: method + action + all <input>
        3) Override the 'super' field & pick 'double' radio
        4) Submit via requests (auto‐redirects)
        5) Scrape the <h2> title + first <table>
        """
        # 1) Fetch blank form
        r1 = self.session.get(self.base_url, timeout=30)
        r1.raise_for_status()
        soup1 = BeautifulSoup(r1.content, 'html.parser')

        form = soup1.find('form')
        if not form:
            logger.error("No <form> on the blank page (HTML snippet):\n%s",
                         r1.text[:300])
            return None

        method = form.get('method', 'GET').upper()
        action = form.get('action') or self.base_url
        submit_url = urljoin(self.base_url, action)
        logger.debug("Form → method=%s, action=%s", method, submit_url)

        # 2) gather all default inputs
        params: Dict[str, str] = {}
        for inp in form.find_all('input'):
            name = inp.get('name')
            if not name:
                continue
            val = inp.get('value', '')
            typ = inp.get('type', 'text').lower()

            if typ == 'radio':
                # pick whichever radio is the “double‐valued” button
                if 'double' in val.lower():
                    params[name] = val
            else:
                # hidden/text/submit → default into params
                params[name] = val

        # 3) override the space‐group field (called 'super' in your snippet)
        params['super'] = str(space_group_id)

        # 4) submit
        if method == 'GET':
            r2 = self.session.get(submit_url, params=params, timeout=30)
        else:
            r2 = self.session.post(submit_url, data=params, timeout=30)
        r2.raise_for_status()
        logger.info("Submitted SG %d → %s", space_group_id, r2.url)

        # 5) scrape
        soup2 = BeautifulSoup(r2.content, 'html.parser')

        title = soup2.find('h2')
        if not title:
            logger.warning("No <h2> on results page for SG %d", space_group_id)
            return None
        title_txt = title.get_text()
        sym_m = re.search(r'\((.*?)\)', title_txt)
        symbol = sym_m.group(1).strip() if sym_m else f"SG {space_group_id}"

        table = soup2.find('table')
        if not table:
            logger.info("No data table for SG %d", space_group_id)
            return symbol, []

        data = self._parse_table(table)
        logger.info("Parsed %d indices for SG %d (%s)",
                    len(data), space_group_id, symbol)
        return symbol, data

    def scrape_all_space_groups(self,
                                start: int = 1,
                                end:   int = 230,
                                delay: float = 1.0):
        db = TopologicalIndicesDatabase()
        ok = 0
        for sg in range(start, end+1):
            logger.info("→ SG %d…", sg)
            out = self.get_double_irreps_data(sg)
            if out:
                sym, dat = out
                if dat:
                    db.insert_data(sg, sym, dat)
                    ok += 1
            time.sleep(delay)
        logger.info("Done. Inserted %d/%d groups", ok, end-start+1)


if __name__ == "__main__":
    scraper = TopologicalIndicesScraper()
    logger.info("Starting scrape of SG 1–230 …")
    scraper.scrape_all_space_groups(delay=0.2)

    # example
    db = TopologicalIndicesDatabase()
    res = db.get_space_group_data(10)
    print("\nExample SG 10 results:")
    for idx in res:
        print(f" • {idx['index_name']} (mod {idx['mod_value']}): "
              f"{idx['irreps']}")
