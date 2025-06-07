import requests
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# Set this to the BNS number that is failing, e.g., '1.1', '1.2', or '4.7'
BNS_TO_TEST = '1.2'
# --- END CONFIGURATION ---

BASE_URL = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/"
PARSER = "html.parser"
OUTPUT_FILENAME = f"debug_output_{BNS_TO_TEST}.html"

def fetch_main(session, bns_number: str) -> str:
    """Fetches the main page using the correct minimal payload."""
    url = urljoin(BASE_URL, "mbandrep.pl")
    try:
        session.get(url, timeout=30)
        payload = { "super": bns_number, "elementary": "Elementary" }
        r_post = session.post(url, data=payload, timeout=60)
        r_post.raise_for_status()
        return r_post.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch main page for BNS {bns_number}: {e}")

def parse_main(html: str, bns_number: str) -> list[dict]:
    """The latest version of the parser logic."""
    soup = BeautifulSoup(html, PARSER)
    main_table = None
    for table in soup.find_all("table"):
        if table.find(string=lambda s: s and "Wyckoff pos" in s):
            main_table = table
            break
    if not main_table:
        raise RuntimeError(f"No main table found for BNS {bns_number}.")

    all_rows = main_table.find_all("tr")
    if not all_rows: return []

    header_rows = {}
    k_vector_rows_start_index = -1
    for i, tr in enumerate(all_rows):
        cells = tr.find_all(['td', 'th'])
        if not cells: continue
        text = cells[0].get_text(strip=True).lower()
        if "wyckoff pos" in text: header_rows["wyckoff"] = cells
        elif "band-rep" in text: header_rows["bandrep"] = cells
        elif "decomposable" in text: header_rows["decomposability"] = cells
        elif ":" in cells[0].get_text(strip=True):
            k_vector_rows_start_index = i
            break
    if not all(k in header_rows for k in ["wyckoff", "bandrep"]):
         raise ValueError(f"Could not find all header rows for BNS {bns_number}")

    num_ebr_columns = max(len(header_rows.get(k, [])) for k in header_rows) - 1
    if num_ebr_columns <= 0: return []

    ebr_data_list = []
    for i in range(1, num_ebr_columns + 1):
        ebr_obj = {"kpoints": {}}
        if i < len(header_rows["wyckoff"]): ebr_obj["wyckoff"] = header_rows["wyckoff"][i].get_text(strip=True)
        if i < len(header_rows["bandrep"]): ebr_obj["bandrep"] = header_rows["bandrep"][i].get_text(strip=True)
        decomposability_cells = header_rows.get("decomposability", [])
        if i < len(decomposability_cells):
            form = decomposability_cells[i].find("form")
            if form:
                ebr_obj["decomposability"] = {"type": "decomposable"}
            else:
                ebr_obj["decomposability"] = {"type": "indecomposable"}
        else:
            ebr_obj["decomposability"] = {"type": "indecomposable"}
        ebr_data_list.append(ebr_obj)

    if k_vector_rows_start_index != -1:
        for i in range(k_vector_rows_start_index, len(all_rows)):
            k_vector_row_cells = all_rows[i].find_all(['td', 'th'])
            k_point_label_full = k_vector_row_cells[0].get_text(strip=True)
            if ":" not in k_point_label_full: continue
            k_point_label = k_point_label_full.split(":")[0]
            for j in range(num_ebr_columns):
                column_index_in_html = j + 1
                if column_index_in_html < len(k_vector_row_cells):
                    irrep_str = k_vector_row_cells[column_index_in_html].get_text(strip=True)
                    ebr_data_list[j]["kpoints"][k_point_label] = irrep_str
    return ebr_data_list

# --- Main execution block for this script ---
if __name__ == "__main__":
    print(f"--- Running Debugger for BNS: {BNS_TO_TEST} ---")
    session = requests.Session()
    session.headers.update({ "User-Agent": "Mozilla/5.0" })
    parsed_data = None

    try:
        # 1. Fetch the HTML
        html_content = fetch_main(session, BNS_TO_TEST)

        # 2. Save it to a file for our inspection
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Raw HTML saved to '{OUTPUT_FILENAME}'")

        # 3. Try to parse it
        parsed_data = parse_main(html_content, BNS_TO_TEST)
        print("✅ HTML parsing completed without errors.")

    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED DURING FETCH OR PARSE ❌")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")

    # 4. Print the final data structure that was created
    print("\n--- PARSED DATA STRUCTURE ---")
    if parsed_data is not None:
        # Use json.dumps for pretty printing the structure
        print(json.dumps(parsed_data, indent=2))
        print(f"\nFound data for {len(parsed_data)} EBR columns.")
    else:
        print("No data was parsed.")
    print("--- END OF DEBUGGING SCRIPT ---")