"""Download women's Torvik T-Rank data from barttorvik.com.

Scrapes the HTML tables since CSV export isn't available for women's.
Saves to data/external/torvik_w_{year}.csv
"""

import re
import sys
import time
import pandas as pd
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "external"

# Column headers from the table (positions 0-23 in data cells)
COL_HEADERS = [
    "rank", "team", "conf", "games", "record",
    "adjoe", "adjde", "barthag",
    "efg_pct", "efgd_pct", "tor", "tord", "orb", "drb",
    "ftr", "ftrd", "two_pt_pct", "two_pt_pct_d",
    "three_pt_pct", "three_pt_pct_d", "three_pt_rate", "three_pt_rate_d",
    "adjt", "wab"
]


def download_page(year):
    """Download Torvik women's page HTML using curl with cookie trick."""
    import tempfile
    cookie_file = tempfile.mktemp(suffix='.txt')

    # Step 1: Get cookies
    subprocess.run([
        "curl", "-s", "-L",
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-c", cookie_file,
        "-d", "js_test_submitted=1",
        f"https://barttorvik.com/trank.php?year={year}"
    ], capture_output=True)

    # Step 2: Download women's page using those cookies
    result = subprocess.run([
        "curl", "-s", "-L",
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-b", cookie_file,
        f"https://barttorvik.com/wncaa/trank.php?year={year}&csv=1"
    ], capture_output=True)

    return result.stdout.decode('utf-8', errors='ignore')


def parse_page(html):
    """Parse team data from the HTML table."""
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)

    rows = []
    for tr in trs:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
        if len(tds) < 20:
            continue

        row = {}
        for i, td in enumerate(tds):
            if i >= len(COL_HEADERS):
                break

            # Clean HTML
            val = re.sub(r'<[^>]+>', ' ', td).strip()
            val = re.sub(r'&nbsp;', ' ', val)
            val = re.sub(r'\s+', ' ', val).strip()

            col = COL_HEADERS[i]

            if col == "team":
                # Extract just team name (remove seed info)
                # Format: "Duke    1 seed, ✅"
                match = re.match(r'([A-Za-z][\w\s.&\'-]+?)(?:\s+\d+ seed|\s*$)', val)
                if match:
                    row[col] = match.group(1).strip()
                else:
                    row[col] = val.split('seed')[0].strip().rstrip('0123456789 ')
                    if not row[col]:
                        row[col] = val
            elif col in ("rank", "games"):
                try:
                    row[col] = int(val.split()[0])
                except (ValueError, IndexError):
                    row[col] = val
            elif col in ("conf", "record"):
                row[col] = val
            else:
                # Numeric columns: "128.2 4" → take first number
                try:
                    parts = val.split()
                    row[col] = float(parts[0]) if parts else None
                except (ValueError, IndexError):
                    row[col] = None

        if "team" in row and row.get("adjoe") is not None:
            rows.append(row)

    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Years to download (women's Torvik likely available from ~2020)
    years = list(range(2015, 2027))

    for year in years:
        outfile = DATA_DIR / f"torvik_w_{year}.csv"
        if outfile.exists():
            print(f"  {year}: already exists, skipping")
            continue

        print(f"  {year}: downloading...", end=" ", flush=True)
        html = download_page(year)

        if len(html) < 1000:
            print(f"FAILED (only {len(html)} bytes)")
            continue

        rows = parse_page(html)
        if not rows:
            print("FAILED (no rows parsed)")
            continue

        df = pd.DataFrame(rows)
        df.to_csv(outfile, index=False)
        print(f"OK ({len(df)} teams, saved to {outfile.name})")

        time.sleep(1)  # Be polite

    # Summary
    print("\nSummary:")
    for year in years:
        f = DATA_DIR / f"torvik_w_{year}.csv"
        if f.exists():
            df = pd.read_csv(f)
            print(f"  {year}: {len(df)} teams, adjoe range [{df['adjoe'].min():.1f}, {df['adjoe'].max():.1f}]")


if __name__ == "__main__":
    main()
