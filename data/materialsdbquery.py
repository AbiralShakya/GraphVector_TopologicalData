# import requests

# def fetch_tqc_detail_by_icsd(icsd_id):
#     url = f"https://www.topologicalquantumchemistry.com/api/material/icsd/{icsd_id}/"
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# detail615755 = fetch_tqc_detail_by_icsd(638505)
# print (detail615755)
# # Now detail615755["irrep_data"] is a dict like 
# #   { "Gamma": ["Γ₁⁺","Γ₁₂⁻",…], "M": [ "M₁", "M₂", … ], … }
# # detail615755["compatibility"] is a list of {"from":[k0,i], "to":[k1,j]}
import requests
import urllib3

# Suppress the InsecureRequestWarning if you like:
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_tqc_detail_by_icsd(icsd_id):
    url = f"https://www.topologicalquantumchemistry.com/#/detail/{icsd_id}/"
    # verify=False disables SSL-certificate checking
    r = requests.get(url, verify=False)
    if r.status_code != 200:
        return None
    return r.json()

detail = fetch_tqc_detail_by_icsd(638505)
print(detail)
