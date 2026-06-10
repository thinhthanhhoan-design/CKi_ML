# download_uiuc_airfoils.py
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
OUT_DIR = Path("uiuc_airfoils_dat")
DELAY = 0.15


def safe_name(name: str) -> str:
    name = name.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name


def get_dat_links():
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; academic-airfoil-downloader/1.0)"
    }

    r = requests.get(BASE_URL, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".dat"):
            links.append(urljoin(BASE_URL, href))

    return sorted(set(links))


def download_file(url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = safe_name(url)
    path = out_dir / filename

    if path.exists() and path.stat().st_size > 0:
        return "skip", path

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; academic-airfoil-downloader/1.0)"
    }

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        return "empty", path

    path.write_text(text + "\n", encoding="utf-8", errors="ignore")
    return "ok", path


def main():
    links = get_dat_links()
    print(f"Found {len(links)} .dat files")

    ok = skip = fail = empty = 0

    for url in tqdm(links):
        try:
            status, path = download_file(url, OUT_DIR)

            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "empty":
                empty += 1

            time.sleep(DELAY)

        except Exception as e:
            fail += 1
            print(f"\nFAILED: {url}\n{e}")

    print("\nDone")
    print(f"Downloaded: {ok}")
    print(f"Skipped:    {skip}")
    print(f"Empty:      {empty}")
    print(f"Failed:     {fail}")
    print(f"Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()