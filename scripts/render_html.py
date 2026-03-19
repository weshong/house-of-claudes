"""
Render the tournament dashboard as a self-contained static HTML page.

Reads from:
  - data/live/results.json    (auto-updated by game_monitor.py)
  - data/live/liveblog.json   (auto-updated by game_monitor.py)
  - outputs/submissions/*.csv (cached model predictions)
  - data/raw/*.csv            (teams, seeds, slots)

Writes:
  - docs/index.html           (served by GitHub Pages)
"""

import os
import json
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
LIVE_DIR = os.path.join(PROJECT_ROOT, 'data', 'live')
SUBS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'submissions')
OUT_PATH = os.path.join(PROJECT_ROOT, 'docs', 'index.html')


# ═══════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════

def load_json(path, default):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default


def load_preds(filename):
    path = os.path.join(SUBS_DIR, filename)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['ID'], df['Pred']))


def load_all_data():
    teams_m = pd.read_csv(os.path.join(DATA_DIR, 'MTeams.csv'))
    teams_w = pd.read_csv(os.path.join(DATA_DIR, 'WTeams.csv'))
    seeds_m = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneySeeds.csv'))
    seeds_w = pd.read_csv(os.path.join(DATA_DIR, 'WNCAATourneySeeds.csv'))
    slots_m = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneySlots.csv'))
    slots_w = pd.read_csv(os.path.join(DATA_DIR, 'WNCAATourneySlots.csv'))

    team_name = {}
    for _, r in pd.concat([teams_m, teams_w]).iterrows():
        team_name[int(r['TeamID'])] = r['TeamName']

    seed_info = {}
    for df in [seeds_m, seeds_w]:
        for _, r in df[df['Season'] == 2026].iterrows():
            s = r['Seed']
            seed_info[int(r['TeamID'])] = (s[0], int(s[1:3]), s)

    v7 = load_preds('submission_stage2_v7_lgb40-lr60-blend_qtl1w_20260318_202646.csv')
    seed = load_preds('baseline_seed_2026.csv')
    rf = load_preds('baseline_rf_2026.csv')

    results = load_json(os.path.join(LIVE_DIR, 'results.json'), [])
    liveblog = load_json(os.path.join(LIVE_DIR, 'liveblog.json'), [])

    return {
        'team_name': team_name, 'seed_info': seed_info,
        'v7': v7, 'seed': seed, 'rf': rf,
        'results': results, 'liveblog': liveblog,
        'seeds_m': seeds_m, 'seeds_w': seeds_w,
        'slots_m': slots_m, 'slots_w': slots_w,
    }


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def mid(t1, t2):
    return f"2026_{min(t1,t2)}_{max(t1,t2)}"


def get_pred(t1, t2, preds):
    low = min(t1, t2)
    p_low = preds.get(mid(t1, t2), 0.5)
    return p_low if t1 == low else 1.0 - p_low


def team_label(tid, data):
    s = data['seed_info'].get(tid)
    name = data['team_name'].get(tid, str(tid))
    return f"({s[1]}) {name}" if s else name


# ═══════════════════════════════════════════════════
# Brier Score Computation
# ═══════════════════════════════════════════════════

def compute_brier(data):
    results = data['results']
    if not results:
        return None

    records = []
    cum = {'v7': 0, 'seed': 0, 'rf': 0}
    for i, r in enumerate(results):
        w, l = r['winner_id'], r['loser_id']
        low = min(w, l)
        actual = 1.0 if w == low else 0.0
        m = mid(w, l)

        for name, preds in [('v7', data['v7']), ('seed', data['seed']), ('rf', data['rf'])]:
            p = preds.get(m, 0.5)
            cum[name] += (p - actual) ** 2

        n = i + 1
        ws = data['seed_info'].get(w, ('?', 99))[1]
        ls = data['seed_info'].get(l, ('?', 99))[1]
        is_upset = isinstance(ws, int) and isinstance(ls, int) and ws > ls

        records.append({
            'game_num': n,
            'label': f"{team_label(w, data)} def. {team_label(l, data)}",
            'round': r['round'], 'date': r['date'], 'gender': r['gender'],
            'winner': w, 'loser': l,
            'w_score': r['winner_score'], 'l_score': r['loser_score'],
            'is_upset': is_upset,
            'v7_brier': cum['v7'] / n,
            'seed_brier': cum['seed'] / n,
            'rf_brier': cum['rf'] / n,
            'v7_pred': get_pred(w, l, data['v7']),
            'v7_correct': get_pred(w, l, data['v7']) >= 0.5,
            'seed_correct': get_pred(w, l, data['seed']) >= 0.5,
            'rf_correct': get_pred(w, l, data['rf']) >= 0.5,
        })
    return records


# ═══════════════════════════════════════════════════
# Plotly Chart as JSON
# ═══════════════════════════════════════════════════

def brier_chart_json(brier_data):
    """Build Plotly chart data as JSON for embedding."""
    if not brier_data:
        return 'null', 'null'

    traces = []

    # v7
    traces.append({
        'x': [r['game_num'] for r in brier_data],
        'y': [round(r['v7_brier'], 5) for r in brier_data],
        'text': [r['label'] for r in brier_data],
        'name': 'v7 (LGB+LR Blend)',
        'type': 'scatter', 'mode': 'lines',
        'line': {'color': '#2563eb', 'width': 3},
        'hovertemplate': 'Game %{x}: %{text}<br>Brier: %{y:.4f}<extra>v7</extra>',
    })

    # Seed
    traces.append({
        'x': [r['game_num'] for r in brier_data],
        'y': [round(r['seed_brier'], 5) for r in brier_data],
        'name': 'Seed Baseline',
        'type': 'scatter', 'mode': 'lines',
        'line': {'color': '#9ca3af', 'width': 2, 'dash': 'dash'},
        'hovertemplate': 'Game %{x}<br>Brier: %{y:.4f}<extra>Seed</extra>',
    })

    # RF
    traces.append({
        'x': [r['game_num'] for r in brier_data],
        'y': [round(r['rf_brier'], 5) for r in brier_data],
        'name': 'R Model (RF)',
        'type': 'scatter', 'mode': 'lines',
        'line': {'color': '#f59e0b', 'width': 2, 'dash': 'dot'},
        'hovertemplate': 'Game %{x}<br>Brier: %{y:.4f}<extra>RF</extra>',
    })

    # Upset markers
    upsets = [r for r in brier_data if r['is_upset']]
    if upsets:
        traces.append({
            'x': [r['game_num'] for r in upsets],
            'y': [round(r['v7_brier'], 5) for r in upsets],
            'text': [r['label'] for r in upsets],
            'name': 'Upset',
            'type': 'scatter', 'mode': 'markers',
            'marker': {'color': '#ef4444', 'size': 10, 'symbol': 'x'},
            'hovertemplate': 'UPSET: %{text}<extra></extra>',
        })

    layout = {
        'title': {'text': 'Cumulative Brier Score by Game', 'font': {'size': 18}},
        'xaxis': {'title': 'Games Completed'},
        'yaxis': {'title': 'Brier Score (lower = better)', 'rangemode': 'tozero'},
        'template': 'plotly_white',
        'height': 420,
        'margin': {'l': 55, 'r': 30, 't': 50, 'b': 40},
        'legend': {'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(255,255,255,0.8)'},
        'shapes': [
            {'type': 'line', 'x0': 0, 'x1': 1, 'xref': 'paper',
             'y0': 0.1267, 'y1': 0.1267, 'line': {'color': '#2563eb', 'width': 1, 'dash': 'dot'}, 'opacity': 0.4},
            {'type': 'line', 'x0': 0, 'x1': 1, 'xref': 'paper',
             'y0': 0.1915, 'y1': 0.1915, 'line': {'color': '#9ca3af', 'width': 1, 'dash': 'dot'}, 'opacity': 0.4},
            {'type': 'line', 'x0': 0, 'x1': 1, 'xref': 'paper',
             'y0': 0.1937, 'y1': 0.1937, 'line': {'color': '#f59e0b', 'width': 1, 'dash': 'dot'}, 'opacity': 0.4},
        ],
        'annotations': [
            {'x': 1, 'xref': 'paper', 'y': 0.1267, 'text': 'v7 CV', 'showarrow': False,
             'xanchor': 'left', 'font': {'size': 10, 'color': '#2563eb'}, 'opacity': 0.6},
            {'x': 1, 'xref': 'paper', 'y': 0.1915, 'text': 'Seed CV', 'showarrow': False,
             'xanchor': 'left', 'font': {'size': 10, 'color': '#9ca3af'}, 'opacity': 0.6},
            {'x': 1, 'xref': 'paper', 'y': 0.1937, 'text': 'RF CV', 'showarrow': False,
             'xanchor': 'left', 'font': {'size': 10, 'color': '#f59e0b'}, 'opacity': 0.6},
        ],
    }

    return json.dumps(traces), json.dumps(layout)


# ═══════════════════════════════════════════════════
# Bracket HTML
# ═══════════════════════════════════════════════════

ROUND_NAMES = {
    'R1': 'R64', 'R2': 'R32', 'R3': 'Sweet 16', 'R4': 'Elite 8',
    'R5': 'Final Four', 'R6': 'Championship',
}


class BracketTracker:
    def __init__(self, slots_df, seeds_df, gender, data):
        self.data = data
        self.seeds = {}
        self.slots = {}
        self.results = {}

        for _, r in seeds_df[seeds_df['Season'] == 2026].iterrows():
            self.seeds[r['Seed']] = int(r['TeamID'])
        for _, r in slots_df[slots_df['Season'] == 2026].iterrows():
            self.slots[r['Slot']] = (r['StrongSeed'], r['WeakSeed'])

        for r in data['results']:
            if r['gender'] == gender:
                w, l = r['winner_id'], r['loser_id']
                self.results[mid(w, l)] = (w, r['winner_score'], r['loser_score'])

    def resolve(self, slot):
        if slot in self.seeds:
            return (self.seeds[slot], True)
        if slot not in self.slots:
            return (None, False)
        strong, weak = self.slots[slot]
        t1, a1 = self.resolve(strong)
        t2, a2 = self.resolve(weak)
        if t1 is None or t2 is None:
            return (None, False)
        m = mid(t1, t2)
        if m in self.results:
            return (self.results[m][0], True)
        p1 = get_pred(t1, t2, self.data['v7'])
        return (t1 if p1 >= 0.5 else t2, False)

    def get_games(self):
        games = []
        for slot, (strong, weak) in self.slots.items():
            if slot[0] != 'R':
                rnd = 'Play-In'
            else:
                rnd = ROUND_NAMES.get(slot[:2], 'Play-In')

            region = ''
            for ch in slot[2:]:
                if ch in 'WXYZ':
                    region = ch
                    break

            t1, t1a = self.resolve(strong)
            t2, t2a = self.resolve(weak)

            game = {'slot': slot, 'round': rnd, 'region': region, 'team1': t1, 'team2': t2}

            if t1 is not None and t2 is not None:
                m = mid(t1, t2)
                if m in self.results:
                    w, ws, ls = self.results[m]
                    lo = t2 if w == t1 else t1
                    p1 = get_pred(t1, t2, self.data['v7'])
                    pred_w = t1 if p1 >= 0.5 else t2
                    game.update({
                        'completed': True, 'winner': w, 'loser': lo,
                        'w_score': ws, 'l_score': ls,
                        'model_correct': pred_w == w,
                        'v7_conf': get_pred(w, lo, self.data['v7']),
                    })
                else:
                    p1 = get_pred(t1, t2, self.data['v7'])
                    game.update({
                        'completed': False,
                        'pred_winner': t1 if p1 >= 0.5 else t2,
                        'pred_conf': max(p1, 1.0 - p1),
                    })
            else:
                game['completed'] = False

            games.append(game)
        return games


def bracket_region_html(tracker, region, region_name, data):
    games = tracker.get_games()
    rg = [g for g in games if g['region'] == region]

    round_order = ['Play-In', 'R64', 'R32', 'Sweet 16', 'Elite 8']
    rounds = {r: sorted([g for g in rg if g['round'] == r], key=lambda g: g['slot'])
              for r in round_order}

    h = f'<div class="region"><div class="region-hdr">Region {region} \u2014 {region_name}</div>'
    h += '<div class="bkt-row">'

    for rnd in round_order:
        if not rounds[rnd]:
            continue
        h += f'<div class="bkt-col"><div class="rnd-lbl">{rnd}</div>'
        for g in rounds[rnd]:
            h += game_html(g, data)
        h += '</div>'
    h += '</div></div>'
    return h


def final_four_html(tracker, label, data):
    games = tracker.get_games()
    ff = [g for g in games if g['round'] in ('Final Four', 'Championship')]
    if not ff:
        return ''

    h = f'<div class="region"><div class="region-hdr">{label} \u2014 Final Four</div>'
    h += '<div class="bkt-row">'
    for rnd in ['Final Four', 'Championship']:
        rg = sorted([g for g in ff if g['round'] == rnd], key=lambda g: g['slot'])
        if not rg:
            continue
        h += f'<div class="bkt-col"><div class="rnd-lbl">{rnd}</div>'
        for g in rg:
            h += game_html(g, data)
        h += '</div>'
    h += '</div></div>'
    return h


def game_html(g, data):
    t1, t2 = g.get('team1'), g.get('team2')
    if t1 is None or t2 is None:
        return '<div class="gm gm-p"><div class="tm">TBD</div><div class="tm">TBD</div></div>'

    if g.get('completed'):
        w, lo = g['winner'], g['loser']
        ok = g['model_correct']
        cls = 'gm-ok' if ok else 'gm-x'
        icon = '&#10003;' if ok else '&#10007;'
        icls = 'i-ok' if ok else 'i-x'
        return (
            f'<div class="gm {cls}">'
            f'<div class="tm tw">{team_label(w, data)} <span class="sc">{g["w_score"]}</span></div>'
            f'<div class="tm tl">{team_label(lo, data)} <span class="sc">{g["l_score"]}</span></div>'
            f'<div class="gm-m"><span class="{icls}">{icon}</span> v7: {g["v7_conf"]:.0%}</div>'
            f'</div>'
        )
    else:
        pw = g.get('pred_winner')
        conf = g.get('pred_conf', 0.5)
        h = '<div class="gm gm-p">'
        for t in [t1, t2]:
            cls = ' tp' if t == pw else ''
            h += f'<div class="tm{cls}">{team_label(t, data)}</div>'
        h += f'<div class="gm-m pm">v7: {team_label(pw, data)} ({conf:.0%})</div></div>'
        return h


# ═══════════════════════════════════════════════════
# Liveblog HTML
# ═══════════════════════════════════════════════════

def liveblog_html(data):
    entries = []
    for entry in reversed(data['liveblog']):
        etype = entry.get('type', 'game')
        ok = entry.get('model_correct', True)

        if etype == 'daily_summary':
            icon, cls = '&#128202;', 'bl-note'
        elif ok:
            icon, cls = '&#10003;', 'bl-ok'
        else:
            icon, cls = '&#10007;', 'bl-x'

        meta = ' &middot; '.join(
            x for x in [entry.get('round', ''), entry.get('gender', ''), entry.get('date', '')] if x)

        entries.append(
            f'<div class="bl-e {cls}">'
            f'<div class="bl-h"><span class="bl-i">{icon}</span>'
            f'<span class="bl-t">{entry.get("title", "")}</span>'
            f'<span class="bl-r">{meta}</span></div>'
            f'<div class="bl-c">{entry.get("commentary", "")}</div></div>'
        )

    if not entries:
        entries.append(
            '<div class="bl-e bl-note" style="text-align:center;padding:30px;">'
            '<div class="bl-c" style="color:#94a3b8;">'
            '<b>Waiting for tournament to begin...</b><br><br>'
            'Monitor is polling ESPN hourly.<br><br>'
            '<b>v7 predictions:</b><br>'
            "Men's: (1) Michigan over (1) Duke<br>"
            "Women's: (1) UConn over (1) UCLA"
            '</div></div>'
        )

    return '\n'.join(entries)


# ═══════════════════════════════════════════════════
# Full Page
# ═══════════════════════════════════════════════════

def render(data=None):
    if data is None:
        data = load_all_data()

    brier_data = compute_brier(data)
    now = datetime.now().strftime('%b %d, %Y %I:%M %p')
    n_games = len(data['results'])

    # Stats
    if brier_data:
        last = brier_data[-1]
        v7c = sum(1 for r in brier_data if r['v7_correct'])
        sc = sum(1 for r in brier_data if r['seed_correct'])
        rfc = sum(1 for r in brier_data if r['rf_correct'])
        ups = sum(1 for r in brier_data if r['is_upset'])
        ups_caught = sum(1 for r in brier_data if r['is_upset'] and r['v7_correct'])
    else:
        last = None

    # Chart
    traces_json, layout_json = brier_chart_json(brier_data)

    # Brackets
    bm = BracketTracker(data['slots_m'], data['seeds_m'], 'M', data)
    bw = BracketTracker(data['slots_w'], data['seeds_w'], 'W', data)

    M_REGIONS = {'W': 'East', 'X': 'South', 'Y': 'Midwest', 'Z': 'West'}
    W_REGIONS = {'W': 'Albany', 'X': 'Spokane', 'Y': 'Fort Worth', 'Z': 'Portland'}

    m_bkt = '<h3 class="sec-hdr">Men\'s Bracket</h3>'
    for rc, rn in M_REGIONS.items():
        m_bkt += bracket_region_html(bm, rc, rn, data)
    m_bkt += final_four_html(bm, "Men's", data)

    w_bkt = '<h3 class="sec-hdr" style="margin-top:24px">Women\'s Bracket</h3>'
    for rc, rn in W_REGIONS.items():
        w_bkt += bracket_region_html(bw, rc, rn, data)
    w_bkt += final_four_html(bw, "Women's", data)

    # Liveblog
    blog = liveblog_html(data)

    # Scoreboard
    if brier_data:
        scoreboard = f'''
        <div class="sb">
          <div class="sc-card accent"><div class="sc-val">{last['v7_brier']:.4f}</div><div class="sc-lbl">v7 Brier</div></div>
          <div class="sc-card"><div class="sc-val">{v7c}/{n_games}</div><div class="sc-lbl">v7 Correct</div></div>
          <div class="sc-card"><div class="sc-val">{last['seed_brier']:.4f}</div><div class="sc-lbl">Seed Brier</div></div>
          <div class="sc-card"><div class="sc-val">{last['rf_brier']:.4f}</div><div class="sc-lbl">RF Brier</div></div>
          <div class="sc-card"><div class="sc-val">{ups}</div><div class="sc-lbl">Upsets ({ups_caught} caught)</div></div>
        </div>'''
    else:
        scoreboard = '''
        <div class="sb">
          <div class="sc-card accent"><div class="sc-val">&mdash;</div><div class="sc-lbl">v7 Brier</div></div>
          <div class="sc-card"><div class="sc-val">0/0</div><div class="sc-lbl">Games</div></div>
          <div class="sc-card"><div class="sc-val">0.127 / 0.134</div><div class="sc-lbl">v7 CV (M/W)</div></div>
          <div class="sc-card"><div class="sc-val">0.192 / 0.149</div><div class="sc-lbl">Seed CV (M/W)</div></div>
        </div>'''

    # Chart section
    if brier_data:
        chart_section = '<div id="brier-chart" style="width:100%;margin-bottom:24px"></div>'
    else:
        chart_section = '''
        <div style="background:#f0f9ff;border:2px solid #2563eb;border-radius:12px;
                    padding:40px;text-align:center;margin:0 0 24px 0;">
            <h3 style="color:#2563eb;margin:0 0 8px 0;">Waiting for games...</h3>
            <p style="color:#64748b;margin:0;font-size:14px;">
                v7 CV: <b>0.1267</b> (M) / <b>0.1335</b> (W) &nbsp;|&nbsp;
                Seed: 0.192 / 0.149 &nbsp;|&nbsp; RF: 0.194 / 0.147
            </p>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>March Madness 2026 Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f1f5f9;color:#1e293b}}
.wrap{{max-width:1440px;margin:0 auto;padding:16px}}
header{{background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:20px 24px;border-radius:12px;margin-bottom:20px}}
header h1{{font-size:20px;font-weight:700}}
header .sub{{font-size:12px;color:#94a3b8;margin-top:4px}}

.sb{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.sc-card{{flex:1;min-width:140px;background:linear-gradient(135deg,#1e293b,#334155);color:white;border-radius:10px;padding:14px 16px;text-align:center}}
.sc-card.accent{{background:linear-gradient(135deg,#1e40af,#2563eb)}}
.sc-val{{font-size:24px;font-weight:700}}
.sc-lbl{{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-top:2px}}
.sc-card.accent .sc-lbl{{color:#bfdbfe}}

.grid{{display:grid;grid-template-columns:1fr 360px;gap:20px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.blog{{max-height:500px!important}}}}

.main{{min-width:0}}
.sec-hdr{{font-size:17px;font-weight:700;color:#1e293b;margin:0 0 12px 0}}

.region{{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:14px;overflow-x:auto}}
.region-hdr{{font-size:14px;font-weight:700;color:#1e293b;margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.bkt-row{{display:flex;gap:10px;align-items:stretch}}
.bkt-col{{display:flex;flex-direction:column;justify-content:space-around;min-width:140px;flex:1}}
.rnd-lbl{{font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px;text-align:center}}

.gm{{border-radius:5px;margin:2px 0;font-size:11px;overflow:hidden;border:1px solid #e2e8f0}}
.gm-ok{{border-left:3px solid #22c55e;background:#f0fdf4}}
.gm-x{{border-left:3px solid #ef4444;background:#fef2f2}}
.gm-p{{border-left:3px solid #94a3b8;background:#f8fafc}}
.tm{{padding:3px 7px;white-space:nowrap;border-bottom:1px solid #f1f5f9}}
.tm:last-of-type{{border-bottom:none}}
.tw{{font-weight:700;color:#166534}}
.tl{{color:#991b1b;text-decoration:line-through;opacity:.65}}
.tp{{font-weight:600;color:#2563eb}}
.sc{{float:right;font-weight:700}}
.gm-m{{padding:1px 7px 3px;font-size:9px;color:#64748b;background:rgba(0,0,0,.03)}}
.pm{{color:#2563eb;font-style:italic}}
.i-ok{{color:#22c55e;font-weight:700}}
.i-x{{color:#ef4444;font-weight:700}}

.blog{{background:white;border:1px solid #e2e8f0;border-radius:10px;max-height:1400px;overflow-y:auto;position:sticky;top:16px}}
.blog-hdr{{position:sticky;top:0;background:linear-gradient(135deg,#1e293b,#334155);color:white;padding:12px 14px;font-size:13px;font-weight:700;border-radius:10px 10px 0 0;z-index:1}}
.bl-e{{padding:10px 14px;border-bottom:1px solid #f1f5f9}}
.bl-ok{{border-left:3px solid #22c55e}}
.bl-x{{border-left:3px solid #ef4444}}
.bl-note{{border-left:3px solid #3b82f6;background:#f0f9ff}}
.bl-h{{display:flex;align-items:center;gap:6px;flex-wrap:wrap}}
.bl-i{{font-size:14px;font-weight:700}}
.bl-ok .bl-i{{color:#22c55e}}
.bl-x .bl-i{{color:#ef4444}}
.bl-t{{font-weight:600;font-size:12px;color:#1e293b}}
.bl-r{{font-size:10px;color:#94a3b8;margin-left:auto}}
.bl-c{{font-size:11px;color:#475569;margin-top:4px;line-height:1.5}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>March Machine Learning Mania 2026</h1>
    <div class="sub">
      house-of-claudes &nbsp;|&nbsp; v7 model (LGB 40% + LR 60%) &nbsp;|&nbsp;
      CV Brier: 0.1267 (M) / 0.1335 (W) &nbsp;|&nbsp; Updated: {now}
    </div>
  </header>

  {scoreboard}

  <div class="grid">
    <div class="main">
      {chart_section}
      {m_bkt}
      {w_bkt}
    </div>
    <div class="blog">
      <div class="blog-hdr">Live Commentary &middot; {n_games} games</div>
      {blog}
    </div>
  </div>
</div>

<script>
var traces = {traces_json};
var layout = {layout_json};
if (traces) {{
  Plotly.newPlot('brier-chart', traces, layout, {{responsive: true}});
}}
</script>
</body>
</html>'''

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Dashboard written to {OUT_PATH} ({len(html)//1024} KB)")


if __name__ == '__main__':
    render()
