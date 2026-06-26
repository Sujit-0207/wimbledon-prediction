# Wimbledon 2026 Quarterfinalist Predictor

Predicts the 8 men and 8 women most likely to reach the Wimbledon quarterfinals using historical match data, grass court performance, and current rankings.

---

## Folder Structure

```
wimbledon predictions/
│
├── README.md                          <- this file
├── wimbledon_predictor.py             <- main prediction script
│
├── predicted_quarterfinalists.csv     <- OUTPUT: 2026 QF predictions
│
│
└── preprocess_combine_datasets.py <- script that built the combined CSVs
```

---

## Datasets

### Match Data (`files/`)

| File | Rows | Content |
|------|------|---------|
| `men_matches_combined_part1-7.csv` | 639,790 total | All men's matches across all tiers |
| `women_matches_combined_part1-3.csv` | 240,324 total | All women's matches |

Both share a 52-column schema with tournament info, round, score, player names, rankings, serve stats, and bookmaker odds. Data spans from the 1960s through June 2026.

**Source schemas merged:**
- ATP results schema (130 yearly files, full player names, 2007–2026)
- Legacy bookmaker-odds format (`atp_tennis.csv`, `wta.csv`, abbreviated names, 2000–2026)
- Kaggle historical (`KaggleMatches.csv`, both genders, up to 2021)

**Key columns used by the model:**
`tourney_name`, `surface`, `tourney_date`, `tourney_level`, `winner_name`, `winner_rank`, `loser_name`, `loser_rank`, `round`, `event_type`

### Supporting Data (`files2/`)

| File | Content |
|------|---------|
| `women_rankings_current.csv` | Official WTA rankings snapshot dated 22 Jun 2026 |
| `men_players_bios.csv` | ATP biographical data (DOB, height, hand, country) |
| `women_players_bios.csv` | WTA biographical data (from Kaggle) |
| `men/women_charting_matches_index.csv` | Match Charting Project index (point-level data reference) |

> **Note:** Player bios and charting index are not loaded by the prediction model — they are reference files for future feature work.

---

## How to Run

**Requirements:** Python 3.8+, pandas, numpy

### Generate 2026 Predictions

```bash
python wimbledon_predictor.py
```

Outputs `predicted_quarterfinalists.csv` with 16 rows (8 men + 8 women).

### Validate Accuracy (Backtest 2022–2024)

```bash
python accuracy_validator.py
```

Outputs `accuracy_report.csv` with Precision@8 for each year and gender.

---

## Model

### Data Scope

All event types are included in feature calculations — main tour, qualifying tournaments, and challenger events all contribute to win rates and form signals. Only round-robin (RR) and bronze-medal (BR) matches are excluded as they are not comparable to single-elimination tennis.

### Features

| Feature | Weight | Description |
|---------|--------|-------------|
| Current ranking | **35%** | Most recent ATP/WTA rank from match data (or WTA rankings CSV for women) |
| Grass WR (recent) | **22%** | Win rate on grass surfaces, Jan 2024 onwards, all event tiers |
| Wimbledon avg depth | **15%** | Average round reached at Wimbledon per edition (all-time) |
| Grass WR (all-time) | **13%** | Career win rate on grass across all event tiers |
| Wimbledon best (2024+) | **8%** | Best single Wimbledon round reached since Jan 2024 |
| Recent form | **7%** | Overall win rate in the last 14 months across all surfaces |

### Scoring

Each feature is min-max normalised to [0, 1] (NaN filled with the median before normalising). Rankings are inverted so rank 1 = score 1.0.

**Composite = 0.35 × rank + 0.22 × grass_wr_recent + 0.15 × wimb_avg + 0.13 × grass_wr_all + 0.08 × wimb_best_rec + 0.07 × form_wr**

### Candidate Filtering

Players must meet at least one criterion to enter scoring:
- Current rank ≤ 150, **or**
- At least 3 recent grass matches (Jan 2024+)

### Name Deduplication

The data contains the same players under different name formats across sources (e.g. `"Carlos Alcaraz"` from ATP schema vs `"Alcaraz C."` from legacy odds). Names are normalised to `(surname_last_word, first_initial)` — this correctly handles particles like "de", "van", "le". The row with the most complete feature data is kept; ties broken by name length (full name preferred) then lowest rank.

---

## 2026 Predictions

Run date: **23 June 2026** (before Wimbledon 2026 starts)

### Men

| Seed | Player | Rank | Grass WR (recent) | Wimb Avg | Score |
|------|--------|------|-------------------|----------|-------|
| 1 | Carlos Alcaraz | 2 | 90.5% | 5.00 | 0.9924 |
| 2 | Novak Djokovic | 4 | 84.6% | 5.20 | 0.9568 |
| 3 | Jannik Sinner | 1 | 89.5% | 4.20 | 0.9206 |
| 4 | Taylor Fritz | 9 | 83.3% | 2.56 | 0.8102 |
| 5 | Daniil Medvedev | 7 | 68.2% | 2.86 | 0.7874 |
| 6 | Alex de Minaur | 6 | 75.0% | 2.43 | 0.7861 |
| 7 | Lorenzo Musetti | 10 | 75.0% | 2.00 | 0.7725 |
| 8 | Ben Shelton | 5 | 64.0% | 3.00 | 0.7718 |

### Women

| Seed | Player | Rank | Grass WR (recent) | Wimb Avg | Score |
|------|--------|------|-------------------|----------|-------|
| 1 | Swiatek I. | 3 | 85.7% | 0.83 | 0.9377 |
| 2 | Rybakina E. | 2 | 64.7% | 1.00 | 0.8982 |
| 3 | Anisimova A. | 5 | 76.5% | 0.75 | 0.8837 |
| 4 | Sabalenka A. | 1 | 76.9% | 0.67 | 0.8828 |
| 5 | Krejcikova B. | 45 | 80.0% | 1.00 | 0.8418 |
| 6 | Bencic B. | 11 | 71.4% | 0.78 | 0.8081 |
| 7 | Svitolina E. | 8 | — | — | 0.7583 |
| 8 | Keys M. | 28 | 66.7% | 1.00 | 0.7556 |

> Women's names are abbreviated (e.g. `Swiatek I.`) because the WTA legacy-odds data source uses that format for all players. Rankings for women come from match data except where the `women_rankings_current.csv` snapshot provides a more recent value.

---

## Accuracy (Backtesting 2022–2024)

Accuracy is measured as **Precision@8**: how many of the 8 predicted quarterfinalists were actual quarterfinalists, drawn directly from the match dataset.

For each backtest year, all match data after Wimbledon's start date is hidden. The model is re-run on the capped data, then compared against who actually appeared in the quarterfinal matches in the dataset.

| Year | Men Correct | Men Precision | Women Correct | Women Precision |
|------|-------------|---------------|---------------|-----------------|
| 2022 | 2 / 8 | 25.0% | 2 / 8 | 25.0% |
| 2023 | 3 / 8 | 37.5% | 5 / 8 | 62.5% |
| 2024 | 5 / 8 | 62.5% | 1 / 8 | 12.5% |
| **Average** | **3.3 / 8** | **41.7%** | **2.7 / 8** | **33.3%** |

**Overall average: 3.0/8 correct (37.5%)**

### Notes on Accuracy

- **Men trend upward** (25% → 37.5% → 62.5%) because the model's recent-data signals become more informative as the current roster of top players (Alcaraz, Sinner, Djokovic) stabilises.
- **Women's 2023 was best** (62.5%) — the top ranked players mostly delivered; Rybakina (defending champion), Sabalenka, Swiatek, Jabeur, and Keys all made QF as predicted.
- **Women's 2024 was hardest** (12.5%) — Wimbledon 2024 saw major upsets: Krejcikova (rank 31) won the title; Sun, Vekic, Navarro, Paolini all reached QF as significant underdogs. Only Rybakina was correctly predicted.
- **Baseline comparison:** Picking 8 players randomly from a 128-draw main draw gives ~6% expected precision. This model averages 37.5%, roughly **6× better than random**.

---

## Limitations

- Player ID namespaces differ across sources (Sackmann-style alphanumeric vs Kaggle numeric IDs) — bios are not entity-resolved into the match data.
- Women's rankings snapshot (`women_rankings_current.csv`) only covers ranks 80+; top-ranked women fall back to ranks extracted from recent match data.
- Qualifying data is included (grass win rates include qualifying-round results), which may slightly inflate win rates for players who grind through qualifying but rarely trouble top seeds in the main draw.
- The model does not account for injuries, withdrawals, draws/bracket placement, or head-to-head records.
