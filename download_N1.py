from bs4 import BeautifulSoup
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

# ---------------- CONFIG ---------------- #
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/gold/level1c"
YEARS = np.arange(2018, 2027)
DOY = np.arange(1, 367)

OUTDIR = r"D:\gold_l1c_NI\mission_data"
MAX_WORKERS = 6          # polite but fast
REQUEST_TIMEOUT = 15     # seconds
SLEEP_BETWEEN_DAYS = 0.05  # light rate limiting
# --------------------------------------- #

os.makedirs(OUTDIR, exist_ok=True)
downloaded = set(os.listdir(OUTDIR))

session = requests.Session()
session.headers.update(
    {"User-Agent": "GOLD-data-downloader (research use)"}
)

def get_file_list(year, doy):
    """Return list of NI files for a given year/day."""
    url = f"{BASE_URL}/{year}/{str(doy).zfill(3)}"
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.content, "html.parser")
    return [
        a["href"]
        for a in soup.find_all("a", href=True)
        if "NI" in a["href"]
    ]

def download_file(file_url, filename):
    """Download a single file safely."""
    path = os.path.join(OUTDIR, filename)

    if os.path.exists(path):
        return "skipped"
    r = session.get(file_url, stream=True, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    downloaded.add(filename)
    return "downloaded"

# ---------------- MAIN LOOP ---------------- #

for year in YEARS:
    print(f"\n Year {year}")
    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for doy in tqdm(DOY, desc=f"Scanning {year}", leave=False):
            files = get_file_list(year, doy)

            for fname in files:
                if fname in downloaded:
                    continue

                file_url = (
                    f"{BASE_URL}/{year}/{str(doy).zfill(3)}/{fname}"
                )
                tasks.append(
                    executor.submit(download_file, file_url, fname)
                )

            time.sleep(SLEEP_BETWEEN_DAYS)

        # progress bar for downloads
        for _ in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc=f"Downloading {year}"
        ):
            pass

print("\n✅ All done.")
