"""Load and cache all competition CSV files."""

import pandas as pd
from marchmadness.config import DATA_DIR

# Map of logical names to CSV filenames
DATASET_FILES = {
    # Men's data
    "MTeams": "MTeams.csv",
    "MSeasons": "MSeasons.csv",
    "MRegularSeasonCompactResults": "MRegularSeasonCompactResults.csv",
    "MRegularSeasonDetailedResults": "MRegularSeasonDetailedResults.csv",
    "MNCAATourneyCompactResults": "MNCAATourneyCompactResults.csv",
    "MNCAATourneyDetailedResults": "MNCAATourneyDetailedResults.csv",
    "MNCAATourneySeeds": "MNCAATourneySeeds.csv",
    "MNCAATourneySlots": "MNCAATourneySlots.csv",
    "MMasseyOrdinals": "MMasseyOrdinals.csv",
    "MGameCities": "MGameCities.csv",
    "MConferenceTourneyGames": "MConferenceTourneyGames.csv",
    "MTeamCoaches": "MTeamCoaches.csv",
    "MTeamConferences": "MTeamConferences.csv",
    "MSecondaryTourneyCompactResults": "MSecondaryTourneyCompactResults.csv",
    "MSecondaryTourneyTeams": "MSecondaryTourneyTeams.csv",
    # Women's data
    "WTeams": "WTeams.csv",
    "WSeasons": "WSeasons.csv",
    "WRegularSeasonCompactResults": "WRegularSeasonCompactResults.csv",
    "WRegularSeasonDetailedResults": "WRegularSeasonDetailedResults.csv",
    "WNCAATourneyCompactResults": "WNCAATourneyCompactResults.csv",
    "WNCAATourneyDetailedResults": "WNCAATourneyDetailedResults.csv",
    "WNCAATourneySeeds": "WNCAATourneySeeds.csv",
    "WNCAATourneySlots": "WNCAATourneySlots.csv",
    "WMasseyOrdinals": "WMasseyOrdinals.csv",
    "WGameCities": "WGameCities.csv",
    "WTeamConferences": "WTeamConferences.csv",
    # Shared
    "Cities": "Cities.csv",
    # Sample submissions
    "SampleSubmissionStage1": "SampleSubmissionStage1.csv",
    "SampleSubmissionStage2": "SampleSubmissionStage2.csv",
}

_cache: dict[str, pd.DataFrame] = {}


def load(name: str) -> pd.DataFrame:
    """Load a single dataset by logical name. Cached after first load."""
    if name in _cache:
        return _cache[name]
    filename = DATASET_FILES.get(name)
    if filename is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_FILES.keys())}")
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    df = pd.read_csv(filepath)
    _cache[name] = df
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load all available datasets. Skips files that don't exist."""
    data = {}
    for name, filename in DATASET_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            data[name] = load(name)
    return data


def clear_cache():
    """Clear the dataset cache."""
    _cache.clear()


if __name__ == "__main__":
    data = load_all()
    print(f"Loaded {len(data)} datasets:")
    for name, df in sorted(data.items()):
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
