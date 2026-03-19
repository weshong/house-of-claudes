#!/usr/bin/env python
"""
Tournament Game Monitor

Automatically fetches NCAA tournament results from ESPN, updates the
dashboard data files, generates liveblog commentary, and re-renders
the notebook.

Usage:
    python scripts/game_monitor.py                # Run once
    python scripts/game_monitor.py --loop 3600    # Poll every hour
    python scripts/game_monitor.py --loop 3600 --no-render  # Skip notebook re-render
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
LIVE_DIR = os.path.join(PROJECT_ROOT, 'data', 'live')
SUBS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'submissions')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

os.makedirs(LIVE_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(LIVE_DIR, 'results.json')
LIVEBLOG_FILE = os.path.join(LIVE_DIR, 'liveblog.json')

# ESPN API
ESPN_SCOREBOARD = {
    'M': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard',
    'W': 'https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard',
}

# 2026 tournament window
TOURNEY_DATES = {
    'M': (datetime(2026, 3, 17), datetime(2026, 4, 7)),
    'W': (datetime(2026, 3, 19), datetime(2026, 4, 7)),
}

# ESPN headline → round name
ROUND_MAP = {
    'first four': 'Play-In',
    '1st round': 'R64',
    'first round': 'R64',
    '2nd round': 'R32',
    'second round': 'R32',
    'sweet 16': 'Sweet 16',
    'regional semifinal': 'Sweet 16',
    'elite eight': 'Elite 8',
    'elite 8': 'Elite 8',
    'regional final': 'Elite 8',
    'final four': 'Final Four',
    'national semifinal': 'Final Four',
    'national championship': 'Championship',
    'championship': 'Championship',
}


# ═══════════════════════════════════════════════════
# Team Name Matching
# ═══════════════════════════════════════════════════

class TeamMapper:
    """Maps ESPN team names/abbreviations to Kaggle TeamIDs."""

    def __init__(self):
        # Store lists of IDs per name to handle men's/women's overlap
        self.name_to_ids = {}
        self._load()

    def _load(self):
        for prefix in ['M', 'W']:
            # Team names
            teams = pd.read_csv(os.path.join(DATA_DIR, f'{prefix}Teams.csv'))
            for _, r in teams.iterrows():
                name = r['TeamName'].lower().strip()
                self.name_to_ids.setdefault(name, []).append(r['TeamID'])

            # Spellings (many alternate names per team)
            path = os.path.join(DATA_DIR, f'{prefix}TeamSpellings.csv')
            if os.path.exists(path):
                sp = pd.read_csv(path, encoding='latin-1')
                for _, r in sp.iterrows():
                    name = str(r['TeamNameSpelling']).lower().strip()
                    self.name_to_ids.setdefault(name, []).append(int(r['TeamID']))

    # ESPN location/abbrev -> Kaggle spelling for tricky teams
    MANUAL_MAP = {
        'miami (oh)': 'miami oh', 'm-oh': 'miami oh',
        'prairie view a&m': 'prairie view', 'pv': 'prairie view',
        "saint mary's": "st mary's ca", 'smc': "st mary's ca",
        'saint peter\'s': "st peter's", 'spe': "st peter's",
        'saint joseph\'s': "st joseph's pa",
        'uconn': 'connecticut', 'conn': 'connecticut',
        'lsu': 'louisiana st',
        'ole miss': 'mississippi',
        'pitt': 'pittsburgh',
        'byu': 'brigham young',
        'ucf': 'ucf',
        'vcu': 'vcu',
        'smu': 'smu',
        'tcu': 'tcu',
        'usc': 'southern cal',
        'unc': 'north carolina',
        'umbc': 'umbc',
        'liu brooklyn': 'liu brooklyn', 'liu': 'liu brooklyn',
        'uc davis': 'uc davis',
        'uc irvine': 'uc irvine',
        'csun': 'cal st northridge',
        'etsu': 'east tenn st',
        'sf austin': 'stephen f austin',
        'sfau': 'stephen f austin',
        'utsa': 'ut san antonio',
        'siue': 'siu edwardsville',
        'siu edwardsville': 'siu edwardsville',
        'unlv': 'unlv',
        'utep': 'utep',
    }

    def match(self, espn_location, espn_abbrev, espn_display, gender):
        """Try to match ESPN team info to a Kaggle TeamID."""
        id_range = (1000, 1999) if gender == 'M' else (3000, 3999)

        # Build candidates from ESPN fields
        candidates = []
        for raw in [espn_location, espn_abbrev, espn_display]:
            if raw:
                c = raw.lower().strip()
                candidates.append(c)
                candidates.append(c.replace('-', ' '))
                # Check manual overrides
                if c in self.MANUAL_MAP:
                    candidates.append(self.MANUAL_MAP[c])

        # Try each candidate against the spelling database
        for name in candidates:
            for tid in self.name_to_ids.get(name, []):
                if tid and id_range[0] <= tid <= id_range[1]:
                    return tid

        return None


# ═══════════════════════════════════════════════════
# ESPN API
# ═══════════════════════════════════════════════════

def fetch_espn(gender, date_str):
    """Fetch ESPN scoreboard for a given date. Returns list of events."""
    url = f"{ESPN_SCOREBOARD[gender]}?dates={date_str}&limit=100&groups=100"
    req = Request(url, headers={'User-Agent': 'MarchMadnessDashboard/1.0'})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get('events', [])
    except (URLError, json.JSONDecodeError, OSError) as e:
        print(f"    ESPN error ({gender}, {date_str}): {e}")
        return []


def parse_round(notes):
    """Extract tournament round from ESPN competition notes."""
    for note in notes:
        headline = note.get('headline', '').lower()
        # Must contain a basketball championship indicator
        # Exclude NIT, CBI, CIT and other non-NCAA tournaments
        if any(excl in headline for excl in ['nit ', 'nit-', 'cbi', 'cit ', 'vegas']):
            continue
        is_tourney = any(kw in headline for kw in [
            'basketball championship', 'ncaa tournament', 'march madness',
            'final four', 'national championship',
        ])
        if not is_tourney:
            continue
        for key, rnd in ROUND_MAP.items():
            if key in headline:
                return rnd
    return None


def parse_event(event, gender, mapper):
    """Parse a completed ESPN event into a game result dict."""
    comps = event.get('competitions', [])
    if not comps:
        return None

    comp = comps[0]

    # Must be completed
    status = comp.get('status', {}).get('type', {})
    if not status.get('completed', False):
        return None

    # Must be a tournament game
    notes = comp.get('notes', [])
    rnd = parse_round(notes)
    if rnd is None:
        return None

    # Parse competitors
    competitors = comp.get('competitors', [])
    if len(competitors) != 2:
        return None

    teams = []
    for c in competitors:
        team_info = c.get('team', {})
        score_str = c.get('score', '0')
        try:
            score = int(score_str)
        except (ValueError, TypeError):
            score = 0

        location = team_info.get('location', '')
        abbrev = team_info.get('abbreviation', '')
        display = team_info.get('displayName', '')
        tid = mapper.match(location, abbrev, display, gender)

        teams.append({
            'tid': tid,
            'score': score,
            'is_winner': c.get('winner', False),
            'location': location,
            'abbrev': abbrev,
            'display': display,
        })

    # Determine winner/loser
    winner = next((t for t in teams if t['is_winner']), None)
    loser = next((t for t in teams if not t['is_winner']), None)

    # Fallback: compare scores
    if not winner or not loser:
        teams.sort(key=lambda t: t['score'], reverse=True)
        winner, loser = teams[0], teams[1]

    if winner['tid'] is None or loser['tid'] is None:
        return None

    # Verify both teams are in our 2026 tournament seeds
    seeds_key_m = 'MNCAATourneySeeds'
    seeds_key_w = 'WNCAATourneySeeds'
    # We'll do this check in the monitor using seed_info instead

    game_date = comp.get('date', event.get('date', ''))[:10]

    return {
        'winner_id': winner['tid'],
        'loser_id': loser['tid'],
        'winner_score': winner['score'],
        'loser_score': loser['score'],
        'round': rnd,
        'date': game_date,
        'gender': gender,
        'espn_event_id': str(event.get('id', '')),
        'winner_name': winner['display'],
        'loser_name': loser['display'],
    }


# ═══════════════════════════════════════════════════
# Liveblog Commentary Generator
# ═══════════════════════════════════════════════════

class CommentaryGenerator:
    """Generates liveblog commentary for completed games."""

    def __init__(self):
        self.v7_preds = self._load_preds(
            'submission_stage2_v7_lgb40-lr60-blend_qtl1w_20260318_202646.csv')
        self.seed_preds = self._load_preds('baseline_seed_2026.csv')
        self.rf_preds = self._load_preds('baseline_rf_2026.csv')

        self.seed_info = {}
        for prefix in ['M', 'W']:
            seeds = pd.read_csv(os.path.join(DATA_DIR, f'{prefix}NCAATourneySeeds.csv'))
            for _, r in seeds[seeds['Season'] == 2026].iterrows():
                self.seed_info[r['TeamID']] = (r['Seed'][0], int(r['Seed'][1:3]), r['Seed'])

        self.team_names = {}
        for prefix in ['M', 'W']:
            teams = pd.read_csv(os.path.join(DATA_DIR, f'{prefix}Teams.csv'))
            for _, r in teams.iterrows():
                self.team_names[r['TeamID']] = r['TeamName']

    def _load_preds(self, filename):
        path = os.path.join(SUBS_DIR, filename)
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        return dict(zip(df['ID'], df['Pred']))

    def _mid(self, t1, t2):
        return f"2026_{min(t1,t2)}_{max(t1,t2)}"

    def _get_pred(self, winner, loser, preds):
        """P(winner wins) from predictions dict."""
        low, high = min(winner, loser), max(winner, loser)
        p_low = preds.get(self._mid(winner, loser), 0.5)
        return p_low if winner == low else 1.0 - p_low

    def _team_label(self, tid):
        seed = self.seed_info.get(tid)
        name = self.team_names.get(tid, str(tid))
        return f"({seed[1]}) {name}" if seed else name

    def generate(self, game, all_results):
        """Generate commentary for a single completed game."""
        w, l = game['winner_id'], game['loser_id']
        ws, ls = game['winner_score'], game['loser_score']
        margin = ws - ls

        v7_p = self._get_pred(w, l, self.v7_preds)
        seed_p = self._get_pred(w, l, self.seed_preds)
        rf_p = self._get_pred(w, l, self.rf_preds)

        v7_correct = v7_p >= 0.5
        seed_correct = seed_p >= 0.5
        rf_correct = rf_p >= 0.5

        w_seed = self.seed_info.get(w, ('?', 99, '?'))[1]
        l_seed = self.seed_info.get(l, ('?', 99, '?'))[1]
        is_upset = isinstance(w_seed, int) and isinstance(l_seed, int) and w_seed > l_seed

        w_label = self._team_label(w)
        l_label = self._team_label(l)

        # Build commentary
        parts = []

        # Outcome description
        if margin >= 20:
            parts.append(f"Blowout. {w_label} cruised past {l_label} by {margin}.")
        elif margin >= 10:
            parts.append(f"Comfortable win for {w_label}, up {margin} over {l_label}.")
        elif margin <= 3:
            parts.append(f"Nail-biter! {w_label} edges {l_label} by just {margin}.")
        else:
            parts.append(f"{w_label} takes it over {l_label}, winning by {margin}.")

        # Model assessment
        if v7_correct:
            if v7_p > 0.9:
                parts.append(f"Model had this at {v7_p:.0%} — easy call.")
            elif v7_p > 0.7:
                parts.append(f"Model correctly favored the winner at {v7_p:.0%}.")
            elif v7_p > 0.55:
                parts.append(f"Close call — model gave {v7_p:.0%}. We'll take it.")
            else:
                parts.append(f"Coin flip at {v7_p:.0%}. Got lucky on this one.")
        else:
            miss_conf = 1.0 - v7_p  # confidence in the loser
            if miss_conf > 0.85:
                parts.append(
                    f"Brutal miss. Model was {miss_conf:.0%} on {l_label}. "
                    f"This is a Brier score killer.")
            elif miss_conf > 0.7:
                parts.append(
                    f"Significant miss — we had {miss_conf:.0%} on {l_label}. Hurts the Brier.")
            elif miss_conf > 0.55:
                parts.append(f"Model leaned wrong at {miss_conf:.0%} for {l_label}. Moderate miss.")
            else:
                parts.append(f"Essentially a toss-up miss ({miss_conf:.0%}). Minimal Brier damage.")

        # Upset flag
        if is_upset:
            parts.append(f"UPSET: {w_seed}-seed over {l_seed}-seed!")

        # Model comparison
        models_right = [m for m, c in [('v7', v7_correct), ('Seed', seed_correct), ('RF', rf_correct)] if c]
        models_wrong = [m for m, c in [('v7', v7_correct), ('Seed', seed_correct), ('RF', rf_correct)] if not c]
        if models_right and models_wrong:
            parts.append(f"[Got it: {', '.join(models_right)} | Missed: {', '.join(models_wrong)}]")

        # Compute running Brier impact
        n_prev = len(all_results)
        if n_prev > 0:
            # Simple per-game Brier contribution
            low = min(w, l)
            actual = 1.0 if w == low else 0.0
            v7_raw = self.v7_preds.get(self._mid(w, l), 0.5)
            game_brier = (v7_raw - actual) ** 2
            if game_brier > 0.15:
                parts.append(f"Per-game Brier contribution: {game_brier:.3f} (ouch).")
            elif game_brier < 0.02:
                parts.append(f"Per-game Brier contribution: {game_brier:.3f} (great).")

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'game',
            'title': f"{w_label} {ws}, {l_label} {ls}",
            'commentary': ' '.join(parts),
            'round': game['round'],
            'gender': game['gender'],
            'date': game['date'],
            'model_correct': v7_correct,
            'is_upset': is_upset,
            'v7_pred': round(v7_p, 4),
            'seed_pred': round(seed_p, 4),
            'rf_pred': round(rf_p, 4),
        }

    def generate_daily_summary(self, results, date_str):
        """Generate end-of-day summary."""
        day_games = [r for r in results if r['date'][:10] == date_str]
        if not day_games:
            return None

        n = len(day_games)
        v7_right = sum(1 for g in day_games
                       if self._get_pred(g['winner_id'], g['loser_id'], self.v7_preds) >= 0.5)
        upsets = sum(1 for g in day_games
                     if self.seed_info.get(g['winner_id'], ('?', 99))[1] >
                     self.seed_info.get(g['loser_id'], ('?', 99))[1])

        # Compute day's Brier
        brier_sum = 0
        for g in day_games:
            low = min(g['winner_id'], g['loser_id'])
            actual = 1.0 if g['winner_id'] == low else 0.0
            p = self.v7_preds.get(self._mid(g['winner_id'], g['loser_id']), 0.5)
            brier_sum += (p - actual) ** 2
        day_brier = brier_sum / n

        summary = (
            f"Day complete: {v7_right}/{n} correct ({v7_right/n:.0%}). "
            f"{upsets} upset(s). "
            f"Today's average Brier: {day_brier:.4f}."
        )

        if day_brier < 0.10:
            summary += " Excellent day for the model."
        elif day_brier < 0.15:
            summary += " Solid performance."
        elif day_brier < 0.20:
            summary += " Mixed results."
        else:
            summary += " Rough day — upsets took their toll."

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'daily_summary',
            'title': f"Day Summary — {date_str}",
            'commentary': summary,
            'date': date_str,
        }


# ═══════════════════════════════════════════════════
# Monitor
# ═══════════════════════════════════════════════════

class GameMonitor:
    def __init__(self, render=True):
        self.mapper = TeamMapper()
        self.commentary = CommentaryGenerator()
        self.render = render
        self.results = self._load_json(RESULTS_FILE, [])
        self.liveblog = self._load_json(LIVEBLOG_FILE, [])
        # Track seen ESPN event IDs to avoid duplicates
        self.seen_ids = {r.get('espn_event_id') for r in self.results if r.get('espn_event_id')}

    @staticmethod
    def _load_json(path, default):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default

    def _save(self):
        with open(RESULTS_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)
        with open(LIVEBLOG_FILE, 'w') as f:
            json.dump(self.liveblog, f, indent=2)

    def _dates_to_check(self, gender):
        """Return list of date strings to check for tournament games."""
        start, end = TOURNEY_DATES[gender]
        today = datetime.now()
        end = min(end, today)
        if today < start:
            return []
        dates = []
        d = start
        while d <= end:
            dates.append(d.strftime('%Y%m%d'))
            d += timedelta(days=1)
        return dates

    def check_espn(self):
        """Fetch all tournament dates from ESPN, return list of new games."""
        new_games = []
        # Build set of 2026 tournament team IDs for validation
        tourney_teams = set(self.commentary.seed_info.keys())

        for gender in ['M', 'W']:
            dates = self._dates_to_check(gender)
            for date_str in dates:
                events = fetch_espn(gender, date_str)
                for event in events:
                    eid = str(event.get('id', ''))
                    if eid in self.seen_ids:
                        continue
                    game = parse_event(event, gender, self.mapper)
                    if game is None:
                        continue
                    # Only accept games where both teams are in our tournament seeds
                    if game['winner_id'] not in tourney_teams or game['loser_id'] not in tourney_teams:
                        continue
                    new_games.append(game)
        return new_games

    def process_new_games(self, new_games):
        """Add new games to results and generate liveblog entries."""
        if not new_games:
            return False

        for game in new_games:
            self.results.append(game)
            self.seen_ids.add(game.get('espn_event_id', ''))

            # Generate liveblog entry
            entry = self.commentary.generate(game, self.results)
            self.liveblog.append(entry)

            w_label = self.commentary._team_label(game['winner_id'])
            l_label = self.commentary._team_label(game['loser_id'])
            status = "CORRECT" if entry['model_correct'] else "MISS"
            print(f"    NEW: {w_label} {game['winner_score']}, "
                  f"{l_label} {game['loser_score']} "
                  f"[{game['round']}] — {status}")

        # Check if any date's games are all done (for daily summary)
        # Remove stale summaries for dates with new games, then regenerate
        dates_in_new = {g['date'][:10] for g in new_games}
        for date_str in dates_in_new:
            try:
                game_date = datetime.strptime(date_str, '%Y-%m-%d')
                if game_date.date() < datetime.now().date():
                    # Remove existing summary for this date (may be stale)
                    self.liveblog = [
                        e for e in self.liveblog
                        if not (e['type'] == 'daily_summary' and e['date'] == date_str)
                    ]
                    summary = self.commentary.generate_daily_summary(self.results, date_str)
                    if summary:
                        self.liveblog.append(summary)
                        print(f"    SUMMARY: {summary['commentary']}")
            except ValueError:
                pass

        return True

    def regenerate_dashboard(self):
        """Re-render the HTML dashboard and push to GitHub."""
        if not self.render:
            return
        print("  Rendering dashboard...")
        try:
            from render_html import render
            render()
        except Exception as e:
            print(f"  Render failed: {e}")
            return

        # Auto-commit and push
        print("  Pushing to GitHub...")
        try:
            import subprocess
            cwd = PROJECT_ROOT
            n = len(self.results)
            msg = f"dashboard: {n} games tracked"
            subprocess.run(['git', 'add', 'docs/index.html', 'data/live/'],
                           cwd=cwd, capture_output=True, timeout=15)
            subprocess.run(['git', 'commit', '-m', msg],
                           cwd=cwd, capture_output=True, timeout=15)
            result = subprocess.run(['git', 'push'],
                                    cwd=cwd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("  Pushed to GitHub.")
            else:
                print(f"  Push warning: {result.stderr[:150]}")
        except Exception as e:
            print(f"  Git push failed: {e}")

    def run_once(self, force_render=False):
        """Check for new games, update files, re-render."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{now}] Checking ESPN for new tournament games...")

        new_games = self.check_espn()

        if new_games:
            print(f"  Found {len(new_games)} new game(s):")
            changed = self.process_new_games(new_games)
            if changed:
                self._save()
                print(f"  Saved: {len(self.results)} total results, "
                      f"{len(self.liveblog)} liveblog entries")
                self.regenerate_dashboard()
        elif force_render:
            print(f"  No new games. Rendering initial dashboard...")
            self.regenerate_dashboard()
        else:
            print(f"  No new games. ({len(self.results)} total results)")

    def run_loop(self, interval):
        """Poll ESPN on an interval."""
        print(f"Starting game monitor (polling every {interval}s / {interval//60}min)")
        print(f"Results file: {RESULTS_FILE}")
        print(f"Liveblog file: {LIVEBLOG_FILE}")
        print(f"Existing results: {len(self.results)}")
        print(f"Press Ctrl+C to stop.\n")

        try:
            self.run_once(force_render=True)  # Always render on first loop start
            while True:
                next_check = datetime.now() + timedelta(seconds=interval)
                print(f"  Next check: {next_check.strftime('%H:%M:%S')}")
                time.sleep(interval)
                self.run_once()
        except KeyboardInterrupt:
            print("\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(description='NCAA Tournament Game Monitor')
    parser.add_argument('--loop', type=int, default=0,
                        help='Poll interval in seconds (0 = run once)')
    parser.add_argument('--no-render', action='store_true',
                        help='Skip notebook re-rendering')
    args = parser.parse_args()

    monitor = GameMonitor(render=not args.no_render)

    if args.loop > 0:
        monitor.run_loop(args.loop)
    else:
        monitor.run_once()


if __name__ == '__main__':
    main()
