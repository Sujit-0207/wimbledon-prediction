# Code Explanation — Wimbledon QF Predictor

This document explains how the prediction model works: what each function does, what data it uses, and how the scoring metrics are calculated.

---

## Files Overview

| File | Purpose |
|------|---------|
| `wimbledon_predictor.py` | Builds the model, scores all active players, outputs top 8 per gender |

---

## wimbledon_predictor.py — Step by Step

### Step 1: Load the data

```python
load_csvs(pattern)
```

Reads all part-files matching a glob pattern (e.g. `men_matches_combined_part*.csv`) and concatenates them into a single DataFrame. Only the 10 columns the model actually needs are loaded (`usecols=LOAD_COLS`) to keep memory usage low across the ~880,000 total rows.

The `tourney_date` column is parsed to datetime, and `winner_rank` / `loser_rank` are coerced to numeric (non-numeric entries become `NaN`).

---

### Step 2: Filter to competitive matches

```python
filter_matches(df)
```

Keeps rows whose `event_type` is one of:

| Event type | Description |
|------------|-------------|
| `main_tour` | ATP/WTA main tour results (2007–2025) |
| `main_tour_ongoing_2026` | 2026 main tour results up to the reference date |
| `legacy_odds_format` | Historical bookmaker-odds data (abbreviated player names) |
| `kaggle_historical` | Kaggle dataset (both genders, up to 2021) |
| `qualifying` | Qualifying-round matches |
| `challenger` | ATP Challenger / ITF matches |
| `challenger_ongoing_2026` | 2026 Challenger results |

Two round types are excluded regardless of event type:

| Round code | Reason excluded |
|------------|-----------------|
| `RR` (Round Robin) | Group-stage matches — not single-elimination, not comparable |
| `BR` (Bronze Medal) | Third-place matches — not part of the QF path |

All other rounds (R128 through the Final) are kept.

---

### Step 3: Identify active players

```python
active_players(df)
```

Returns the set of all player names (winner or loser) who appeared in at least one match since **1 January 2025**. This prevents retired or long-inactive players from being scored — for example, players who retired in 2022 would otherwise still have strong historical grass win rates.

---

### Step 4: Build feature data

Six features are computed for each active player. Each function operates on a pre-filtered version of the match DataFrame.

#### Feature 1 — Current Ranking (35% weight)

```python
rank_map(df)
```

Takes all matches from **1 January 2024 onwards**, extracts every `(player_name, rank)` pair from both the winner and loser columns, sorts by date descending, and keeps only the most recent entry per player. This gives the most up-to-date known ranking for each player.

For women, the `women_rankings_current.csv` snapshot (dated 22 Jun 2026) overrides this where available, since the snapshot is more current than the match data alone.

The raw rank is converted to a score:

```
rank_score = 1 - (min(rank, 200) - 1) / 199
```

This maps rank 1 → 1.0, rank 200 → 0.0. Ranks above 200 are clipped to 200 before scoring.

---

#### Feature 2 — Recent Grass Win Rate (22% weight)

```python
win_rates(grass[grass["tourney_date"] >= RECENT_CUT])
```

Win rate on grass surfaces from **1 January 2024 onwards**, across all event tiers.

```
grass_wr_recent = wins_on_grass_since_2024 / (wins + losses)_on_grass_since_2024
```

This captures current grass form — a player on a hot grass-court run this season scores highly here even if their overall career grass record is modest.

Players with zero recent grass matches get `NaN` for this feature (filled with the median before normalisation).

---

#### Feature 3 — Wimbledon Historical Average Depth (15% weight)

```python
wimbledon_depth(df)  →  wimb_avg_all
```

For each Wimbledon edition a player entered, their deepest round reached is scored using the `ROUND_PTS` table below. The average of this score across all editions they entered is `wimb_avg_all`.

This rewards players who consistently go deep at Wimbledon, not just those who had one good run years ago.

**Round points table:**

| Round | Points |
|-------|--------|
| 1st Round (R128 / "1st Round") | 1 |
| 2nd Round (R64 / "2nd Round") | 1 |
| 3rd Round (R32 / "3rd Round") | 2 |
| 4th Round (R16 / "4th Round") | 3 |
| Quarterfinals (QF) | 4 |
| Semifinals (SF) | 5 |
| Final (F) | 6 |
| Winner (W) | 7 |

Both abbreviated forms (ATP schema: `QF`, `SF`, `F`, `W`) and full-word forms (WTA legacy format: `Quarterfinals`, `Semifinals`, `The Final`) are mapped.

**Winner depth credit:** The winner of each round goes one step further than the loser, so they receive `min(round_pts + 1, 7)` rather than `round_pts`. This avoids penalising players for winning — for example, a player who wins the QF match gets 5 points (Semifinalist depth), not 4.

---

#### Feature 4 — All-time Grass Win Rate (13% weight)

```python
win_rates(grass)  →  grass_wr_all
```

Career win rate on grass across all surfaces and all available years (back to 1960s in some records). This rewards players who have proven long-term grass-court competence, independent of recent form.

```
grass_wr_all = total_grass_wins / total_grass_matches (all time)
```

---

#### Feature 5 — Best Wimbledon Run Since 2024 (8% weight)

```python
wimbledon_depth(mt[mt["tourney_date"] >= RECENT_CUT])  →  wimb_best_rec
```

The single best Wimbledon round reached since 1 January 2024 (i.e. Wimbledon 2024 and 2025). A player who reached the final last year scores highly here even if they have a modest historical average.

This feature rewards players who are at their peak Wimbledon form right now.

---

#### Feature 6 — Recent Overall Form (7% weight)

```python
win_rates(mt[mt["tourney_date"] >= FORM_CUT])  →  form_wr
```

Overall win rate across all surfaces in the last 14 months (from approximately April 2025 to June 2026). Surface is not filtered here — the intent is to capture general match fitness and momentum.

```
form_wr = wins_last_14_months / (wins + losses)_last_14_months
```

---

### Step 5: Filter candidates

Before scoring, the player pool is narrowed to candidates who meet at least one of:

- **Current rank ≤ 150** — ensures the player is ranked inside the main draw entry range
- **≥ 3 recent grass matches** — allows unranked or outside-150 players with significant recent grass activity to be included (e.g. returning players or those entering via wildcards)

---

### Step 6: Name deduplication

```python
name_key(name)
dedup_candidates(sc)
```

The data contains the same players under different name formats depending on which source the row came from:

| Source | Format | Example |
|--------|--------|---------|
| ATP schema | `"First Last"` | `"Carlos Alcaraz"` |
| Legacy odds | `"Last F."` | `"Alcaraz C."` |
| WTA schema | `"Last, First"` | `"Alcaraz, Carlota"` (parsed at load time) |

`name_key()` normalises both `"Carlos Alcaraz"` and `"Alcaraz C."` to the same key: `("alcaraz", "c")`.

The key is built as `(surname_last_word, first_initial)`, using only the **last word** of the surname portion. This correctly handles name particles:

| Player | Abbreviated | Key |
|--------|------------|-----|
| Alex de Minaur | De Minaur A. | `("minaur", "a")` |
| Jan-Lennard Struff | Struff J. | `("struff", "j")` |
| Stan Wawrinka | Wawrinka S. | `("wawrinka", "s")` |

`dedup_candidates()` groups rows by their key and keeps the one with the most non-null features. Ties are broken by name length (preferring the longer full name) then by rank (preferring the lower / better rank).

---

### Step 7: Composite scoring

All six features are **min-max normalised** to [0, 1] before combining:

```python
def norm(s):
    s = s.fillna(s.median())        # NaN → median before normalising
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo)    # scale to [0, 1]
```

Using the median (rather than 0) to fill missing values means a player with no data for a feature is treated as average, not as the worst — which is fair for features like `wimb_avg_all` where the absence of Wimbledon history doesn't mean the player is a first-rounder.

The composite score combines all six normalised features:

```
composite = 0.35 × rank_score
          + 0.22 × norm(grass_wr_recent)
          + 0.15 × norm(wimb_avg_all)
          + 0.13 × norm(grass_wr_all)
          + 0.08 × norm(wimb_best_rec)
          + 0.07 × norm(form_wr)
```

The top 8 composite scores per gender are the predictions.

---


## What the Model Does Not Account For

- **Injuries and withdrawals** — a player ranked #2 who is nursing a wrist injury gets the same rank score as a fully fit #2
- **Draw / bracket placement** — two top players meeting in the third round rather than the quarterfinal is not modelled
- **Head-to-head records** — no matchup-specific data
- **Playing style on grass** — serve-and-volley players or those with flat serves tend to do better on grass, but no point-level charting data is used
- **Qualifying performance is included** — grass win rates include qualifying rounds, which can slightly inflate rates for players who grind through qualifying but struggle against top seeds in the main draw
