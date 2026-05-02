# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:15:58 2026

@author: dogbl
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- CONFIG ---------------- #
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/gold/level1c"
YEARS = np.arange(2018, 2026)
DOY = np.arange(1, 367)

OUTDIR = r"D:\gold_l1c_NI"
MAX_WORKERS = 3              # safer for NASA servers
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_DAYS = 0.2
MAX_RETRIES = 5
# ---------------------------------------- #

os.makedirs(OUTDIR, exist_ok=True)

# ---------------- SESSION SETUP ---------------- #
session = requests.Session()
session.headers.update({
    "User-Agent": "GOLD-data-downloader (research use)"
})

retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
# ------------------------------------------------ #

def get_file_list(year, doy, retries=3):
    """Get NI file list for a given day with retry."""
    url = f"{BASE_URL}/{year}/{str(doy).zfill(3)}"

    for attempt in range(retries):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return []

            soup = BeautifulSoup(r.content, "html.parser")
            return [
                a["href"]
                for a in soup.find_all("a", href=True)
                if "NI" in a["href"]
            ]

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"⚠️ Failed listing {year}/{doy}: {e}")
                return []


def download_file(file_url, filename, retries=3):
    """Download with resume + size checking."""
    path = os.path.join(OUTDIR, filename)

    for attempt in range(retries):
        try:
            # --- get remote file size ---
            head = session.head(file_url, timeout=REQUEST_TIMEOUT)
            if head.status_code != 200:
                return "error"

            remote_size = int(head.headers.get("Content-Length", 0))

            # --- check local file ---
            if os.path.exists(path):
                local_size = os.path.getsize(path)

                if local_size == remote_size and remote_size > 0:
                    return "skipped"

                # resume if partial
                headers = {"Range": f"bytes={local_size}-"}
                mode = "ab"
            else:
                headers = {}
                mode = "wb"
                local_size = 0

            # --- download ---
            r = session.get(
                file_url,
                stream=True,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()

            with open(path, mode) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return "downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"❌ Failed {filename}: {e}")
                return "error"


# ---------------- MAIN LOOP ---------------- #

for year in YEARS:
    print(f"\n📅 Year {year}")
    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        for doy in tqdm(DOY, desc=f"Scanning {year}", leave=False):
            files = get_file_list(year, doy)

            for fname in files:
                file_url = f"{BASE_URL}/{year}/{str(doy).zfill(3)}/{fname}"

                tasks.append(
                    executor.submit(download_file, file_url, fname)
                )

            # rate limiting + jitter
            time.sleep(SLEEP_BETWEEN_DAYS + np.random.uniform(0, 0.1))

        # track downloads
        results = {"downloaded": 0, "skipped": 0, "error": 0}

        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc=f"Downloading {year}"
        ):
            result = future.result()
            if result in results:
                results[result] += 1

    print(f"✅ {year} summary: {results}")

print("\n🎉 All done.")