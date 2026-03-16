"""Bart Torvik T-Rank features (external data).

Uses adjusted efficiency metrics from barttorvik.com as additional features.
These are free, publicly available ratings similar to KenPom.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

from marchmadness.config import PROJECT_ROOT


TORVIK_DIR = PROJECT_ROOT / "data" / "external"

# Key features to extract from Torvik data
TORVIK_FEATURES = ["adjoe", "adjde", "barthag", "adjt", "sos"]

# Manual name mappings for common mismatches between Kaggle and Torvik
# Format: Kaggle name -> Torvik name
NAME_OVERRIDES = {
    "Abilene Chr": "Abilene Christian",
    "Alabama A&M": "Alabama A&M",
    "SUNY Albany": "Albany",
    "American Univ": "American",
    "Appalachian St": "Appalachian St.",
    "Arizona St": "Arizona St.",
    "Ark Little Rock": "Little Rock",
    "Ark Pine Bluff": "Arkansas Pine Bluff",
    "Arkansas St": "Arkansas St.",
    "Ball St": "Ball St.",
    "Boise St": "Boise St.",
    "Bowling Green": "Bowling Green",
    "CS Bakersfield": "Cal St. Bakersfield",
    "CS Fullerton": "Cal St. Fullerton",
    "CS Northridge": "Cal St. Northridge",
    "CS Sacramento": "Sacramento St.",
    "Cal Poly": "Cal Poly",
    "Cal St Long Beach": "Long Beach St.",
    "Central Conn": "Central Connecticut",
    "Charleston So": "Charleston Southern",
    "Coastal Car": "Coastal Carolina",
    "Col Charleston": "Charleston",
    "Coppin St": "Coppin St.",
    "Delaware St": "Delaware St.",
    "Detroit": "Detroit Mercy",
    "East Tenn St": "East Tennessee St.",
    "Eastern Ill": "Eastern Illinois",
    "Eastern Ky": "Eastern Kentucky",
    "Eastern Mich": "Eastern Michigan",
    "Eastern Wash": "Eastern Washington",
    "F Dickinson": "Fairleigh Dickinson",
    "Fla Atlantic": "Florida Atlantic",
    "Fla Gulf Coast": "Florida Gulf Coast",
    "Florida St": "Florida St.",
    "Fresno St": "Fresno St.",
    "Ga Southern": "Georgia Southern",
    "Gardner Webb": "Gardner-Webb",
    "George Mason": "George Mason",
    "Georgia St": "Georgia St.",
    "Grambling": "Grambling St.",
    "Green Bay": "Green Bay",
    "Houston Bap": "Houston Christian",
    "IL Chicago": "UIC",
    "Idaho St": "Idaho St.",
    "Illinois St": "Illinois St.",
    "Indiana St": "Indiana St.",
    "Iowa St": "Iowa St.",
    "Jackson St": "Jackson St.",
    "Jacksonville St": "Jacksonville St.",
    "Kansas St": "Kansas St.",
    "Kennesaw": "Kennesaw St.",
    "Kent": "Kent St.",
    "Lamar": "Lamar",
    "Long Island": "LIU",
    "Loyola MD": "Loyola Maryland",
    "Loyola-Chicago": "Loyola Chicago",
    "MD Baltimore Co": "UMBC",
    "MD Eastern Shore": "Maryland Eastern Shore",
    "Miami FL": "Miami FL",
    "Miami OH": "Miami OH",
    "Michigan St": "Michigan St.",
    "Middle Tenn": "Middle Tennessee",
    "Milwaukee": "Milwaukee",
    "Mississippi St": "Mississippi St.",
    "Mississippi Val": "Mississippi Valley St.",
    "Missouri St": "Missouri St.",
    "Montana St": "Montana St.",
    "Morehead St": "Morehead St.",
    "Morgan St": "Morgan St.",
    "Mt St Marys": "Mount St. Mary's",
    "Murray St": "Murray St.",
    "N Colorado": "Northern Colorado",
    "N Dakota St": "North Dakota St.",
    "N Illinois": "Northern Illinois",
    "N Kentucky": "Northern Kentucky",
    "NC A&T": "North Carolina A&T",
    "NC Central": "North Carolina Central",
    "NC Greensboro": "UNC Greensboro",
    "NC State": "N.C. State",
    "New Mexico St": "New Mexico St.",
    "Nicholls St": "Nicholls St.",
    "Norfolk St": "Norfolk St.",
    "North Texas": "North Texas",
    "Northwestern St": "Northwestern St.",
    "Ohio St": "Ohio St.",
    "Oklahoma St": "Oklahoma St.",
    "Oral Roberts": "Oral Roberts",
    "Oregon St": "Oregon St.",
    "Penn St": "Penn St.",
    "Pittsburgh": "Pittsburgh",
    "Portland St": "Portland St.",
    "Prairie View": "Prairie View A&M",
    "S Carolina St": "South Carolina St.",
    "S Dakota St": "South Dakota St.",
    "S Illinois": "Southern Illinois",
    "Sacred Heart": "Sacred Heart",
    "Sam Houston St": "Sam Houston St.",
    "San Diego St": "San Diego St.",
    "San Jose St": "San Jose St.",
    "Savannah St": "Savannah St.",
    "SE Louisiana": "Southeastern Louisiana",
    "SE Missouri St": "Southeast Missouri St.",
    "SIUE": "SIU Edwardsville",
    "SIU Edwardsville": "SIU Edwardsville",
    "Southern Miss": "Southern Miss",
    "St Bonaventure": "St. Bonaventure",
    "St Francis PA": "St. Francis PA",
    "St Francis NY": "St. Francis NY",
    "St Josephs": "Saint Joseph's",
    "St Johns": "St. John's",
    "St Louis": "Saint Louis",
    "St Marys": "Saint Mary's",
    "St Peters": "Saint Peter's",
    "Stephen F Austin": "Stephen F. Austin",
    "Stony Brook": "Stony Brook",
    "TAM C. Christi": "Texas A&M Corpus Chris",
    "Tenn Martin": "UT Martin",
    "Tennessee St": "Tennessee St.",
    "Tennessee Tech": "Tennessee Tech",
    "Texas A&M": "Texas A&M",
    "Texas Southern": "Texas Southern",
    "Texas St": "Texas St.",
    "Troy": "Troy",
    "Tulsa": "Tulsa",
    "UC Davis": "UC Davis",
    "UC Irvine": "UC Irvine",
    "UC Riverside": "UC Riverside",
    "UC Santa Barbara": "UC Santa Barbara",
    "UCLA": "UCLA",
    "UNC Asheville": "UNC Asheville",
    "UNC Wilmington": "UNC Wilmington",
    "USC": "USC",
    "USC Upstate": "USC Upstate",
    "UT Arlington": "UT Arlington",
    "UT San Antonio": "UTSA",
    "Utah St": "Utah St.",
    "Utah Valley": "Utah Valley",
    "VCU": "VCU",
    "W Carolina": "Western Carolina",
    "W Illinois": "Western Illinois",
    "W Kentucky": "Western Kentucky",
    "W Michigan": "Western Michigan",
    "W Virginia": "West Virginia",
    "Wash State": "Washington St.",
    "Weber St": "Weber St.",
    "Wichita St": "Wichita St.",
    "Winthrop": "Winthrop",
    "WI Green Bay": "Green Bay",
    "WI Milwaukee": "Milwaukee",
    "Wright St": "Wright St.",
    "Youngstown St": "Youngstown St.",
    "NE Omaha": "Nebraska Omaha",
    "St Mary's CA": "Saint Mary's",
    "Mt St Mary's": "Mount St. Mary's",
    "St Francis PA": "St. Francis PA",
}


def _build_name_map(kaggle_names: list[str], torvik_names: list[str]) -> dict[str, str]:
    """Build mapping from Kaggle team name to Torvik team name."""
    name_map = {}
    torvik_lower = {n.lower(): n for n in torvik_names}

    for kname in kaggle_names:
        # Check manual overrides first
        if kname in NAME_OVERRIDES:
            override = NAME_OVERRIDES[kname]
            if override.lower() in torvik_lower:
                name_map[kname] = torvik_lower[override.lower()]
                continue

        # Exact match (case insensitive)
        if kname.lower() in torvik_lower:
            name_map[kname] = torvik_lower[kname.lower()]
            continue

        # Fuzzy match
        best_score = 0
        best_match = None
        for tname in torvik_names:
            score = SequenceMatcher(None, kname.lower(), tname.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = tname
        if best_score >= 0.7:
            name_map[kname] = best_match

    return name_map


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute Torvik features for each team in the given season.

    Returns DataFrame with columns: [TeamID, Torvik_adjoe, Torvik_adjde, Torvik_barthag, ...]
    """
    # Only available for men's (Torvik doesn't cover women's)
    if gender != "M":
        return pd.DataFrame(columns=["TeamID"])

    torvik_path = TORVIK_DIR / f"torvik_{season}.csv"
    if not torvik_path.exists():
        return pd.DataFrame(columns=["TeamID"])

    try:
        # Read by position to avoid header/data column count mismatch
        # (older Torvik CSVs have "Fun Rk, adjt" as a quoted header but
        # two separate data columns, causing pandas to shift everything)
        raw = pd.read_csv(torvik_path, header=None, skiprows=1)
        ncols = raw.shape[1]

        # Determine if rank column is present (numeric) or missing (shifted)
        first_val = str(raw.iloc[0, 0])
        if first_val.replace(".", "").replace("-", "").isdigit():
            # Normal format: col 0=rank, 1=team, 4=adjoe, 6=adjde, 8=barthag, 15=sos, last=adjt
            torvik_df = pd.DataFrame({
                "team": raw.iloc[:, 1].astype(str),
                "adjoe": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
                "adjde": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
                "barthag": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
                "sos": pd.to_numeric(raw.iloc[:, 15], errors="coerce"),
                "adjt": pd.to_numeric(raw.iloc[:, ncols - 1], errors="coerce"),
            })
        else:
            # Shifted: col 0=team, 3=adjoe, 5=adjde, 7=barthag, 14=sos, last=adjt
            torvik_df = pd.DataFrame({
                "team": raw.iloc[:, 0].astype(str),
                "adjoe": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
                "adjde": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
                "barthag": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
                "sos": pd.to_numeric(raw.iloc[:, 14], errors="coerce"),
                "adjt": pd.to_numeric(raw.iloc[:, ncols - 1], errors="coerce"),
            })
    except Exception as e:
        print(f"  Torvik load error for {season}: {e}")
        return pd.DataFrame(columns=["TeamID"])

    # Get Kaggle teams
    teams_key = f"{gender}Teams"
    if teams_key not in data:
        return pd.DataFrame(columns=["TeamID"])
    kaggle_teams = data[teams_key][["TeamID", "TeamName"]].copy()

    # Build name mapping
    name_map = _build_name_map(
        kaggle_teams["TeamName"].tolist(),
        torvik_df["team"].tolist()
    )

    # Map Kaggle teams to Torvik data
    rows = []
    for _, team in kaggle_teams.iterrows():
        kname = team["TeamName"]
        if kname not in name_map:
            continue

        tname = name_map[kname]
        torvik_row = torvik_df[torvik_df["team"] == tname]
        if torvik_row.empty:
            continue

        torvik_row = torvik_row.iloc[0]
        row = {"TeamID": team["TeamID"]}
        for feat in TORVIK_FEATURES:
            if feat in torvik_row.index:
                val = torvik_row[feat]
                row[f"Torvik_{feat}"] = float(val) if pd.notna(val) else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["TeamID"])

    result = pd.DataFrame(rows)
    matched = len(result)
    total_kaggle = len(kaggle_teams)
    print(f"  Torvik: matched {matched}/{total_kaggle} teams for season {season}")

    return result
