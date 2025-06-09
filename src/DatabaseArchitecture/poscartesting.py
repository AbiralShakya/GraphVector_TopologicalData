import requests

url = "https://www.topologicalquantumchemistry.com/data/nonalloycompounds/SG-166/Mat-71/POSCAR"
resp = requests.get(url, verify=False)   # disable SSL verify if it gives you trouble
resp.raise_for_status()
print(resp.text)
