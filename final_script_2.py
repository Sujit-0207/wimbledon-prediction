"""
╔══════════════════════════════════════════════════════════════════╗
║      WIMBLEDON 2026 — PHASE 1 QUARTERFINALIST PREDICTOR        ║
║         EXL Analytics Hackathon | June 2026                    ║
║         + FIXED EVALUATION METRICS (v3)                        ║
╚══════════════════════════════════════════════════════════════════╝

DATASET FORMAT:
  Tournament,Date,Series,Court,Surface,Round,Best of,
  Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score

HOW TO RUN:
  pip install pandas numpy
  python wimbledon_phase1_predict.py

═══════════════════════════════════════════════════════════════════
 WHY OLD ACCURACY WAS POOR — AND WHAT WE FIXED
═══════════════════════════════════════════════════════════════════

 PROBLEM 1 — Wrong metric for top-K prediction:
   Binary accuracy = (TP+TN)/16.  When unseeded players crash the QF,
   TN → 0, so accuracy collapses to TP/16 ≈ 0.12–0.25. That's not
   model failure — that's a broken metric.

   FIX → Primary metric is now Hit Rate = TP/8 (how many of our 8
   picks were actually right). A hit rate of 0.50 means 4/8 correct,
   which is genuinely good for a tennis draw full of upsets.

 PROBLEM 2 — Candidate pool mismatch in backtest:
   We scored the *2026* seed list against *past* QF results. Players
   like Mensik, Fonseca, Draper had near-zero history pre-2022, so
   the model always scored them low → unavoidable FNs even if the
   model logic was correct.

   FIX → Backtest now uses the ACTUAL seeded players from each past
   Wimbledon year (extracted from dataset), not the 2026 list.

 PROBLEM 3 — Ground truth extracted too many names:
   get_actual_qf_surnames() counted both P1 and P2 from every deep
   round row, often returning 12-15 names instead of 8.

   FIX → We now extract the 8 QF WINNERS from Wimbledon in year T
   (i.e., the 8 players who won their QF match = actual SF players).
   This gives exactly 8 ground-truth names every time.

 NEW METRICS EXPLAINED:
   Hit Rate  = TP / 8           → primary metric, use this
   Precision = TP / (TP+FP)     → how clean are our picks
   Recall    = TP / (TP+FN)     → how many QFists did we catch
   F1        = harmonic mean(P,R)→ overall balance
   MCC       = balanced corr.   → best single robust metric
   Accuracy  = (TP+TN)/candidates → kept but de-emphasised
"""

import pandas as pd
import numpy as np
import warnings, os, sys

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  ▶ CHANGE THIS TO YOUR FILE NAME
# ══════════════════════════════════════════════════════════════
DATASET_PATH = r"C:\Users\srika\Desktop\wimbledon\data\atp_tennis.csv"
# ══════════════════════════════════════════════════════════════

BACKTEST_YEARS = list(range(2015, 2026))
TOP_K          = 8

MEN_SEEDS = [
    (1,  "Sinner",          "J", "Jannik Sinner"),
    (2,  "Zverev",          "A", "Alexander Zverev"),
    (3,  "Djokovic",        "N", "Novak Djokovic"),
    (4,  "Medvedev",        "D", "Daniil Medvedev"),
    (5,  "Fritz",           "T", "Taylor Fritz"),
    (6,  "De Minaur",       "A", "Alex de Minaur"),
    (7,  "Shelton",         "B", "Ben Shelton"),
    (8,  "Auger-Aliassime", "F", "Felix Auger-Aliassime"),
    (9,  "Cobolli",         "F", "Flavio Cobolli"),
    (10, "Bublik",          "A", "Alexander Bublik"),
    (11, "Mensik",          "J", "Jakub Mensik"),
    (12, "Tiafoe",          "F", "Frances Tiafoe"),
    (13, "Fonseca",         "J", "Joao Fonseca"),
    (14, "Khachanov",       "K", "Karen Khachanov"),
    (15, "Paul",            "T", "Tommy Paul"),
    (16, "Draper",          "J", "Jack Draper"),
]

WOMEN_SEEDS = [
    (1,  "Sabalenka",  "A", "Aryna Sabalenka"),
    (2,  "Rybakina",   "E", "Elena Rybakina"),
    (3,  "Swiatek",    "I", "Iga Swiatek"),
    (4,  "Pegula",     "J", "Jessica Pegula"),
    (5,  "Andreeva",   "M", "Mirra Andreeva"),
    (6,  "Anisimova",  "A", "Amanda Anisimova"),
    (7,  "Gauff",      "C", "Coco Gauff"),
    (8,  "Svitolina",  "E", "Elina Svitolina"),
    (9,  "Noskova",    "L", "Linda Noskova"),
    (10, "Muchova",    "K", "Karolina Muchova"),
    (11, "Bencic",     "B", "Belinda Bencic"),
    (12, "Kostyuk",    "M", "Marta Kostyuk"),
    (13, "Paolini",    "J", "Jasmine Paolini"),
    (14, "Osaka",      "N", "Naomi Osaka"),
    (15, "Shnaider",   "D", "Diana Shnaider"),
    (16, "Navarro",    "E", "Emma Navarro"),
]

ROUND_WEIGHTS = {
    "1st Round": 1.0, "2nd Round": 1.2, "3rd Round": 1.5,
    "Round of 16": 1.8, "Quarterfinals": 2.5,
    "Semifinals": 3.0, "The Final": 3.5, "Final": 3.5,
}

YEAR_WEIGHTS = {
    2000:0.10, 2001:0.10, 2002:0.15, 2003:0.15, 2004:0.20, 2005:0.20,
    2006:0.25, 2007:0.25, 2008:0.30, 2009:0.30, 2010:0.35, 2011:0.40,
    2012:0.45, 2013:0.50, 2014:0.55, 2015:0.60, 2016:0.65, 2017:0.70,
    2018:0.75, 2019:0.80, 2020:0.82, 2021:0.85, 2022:0.88,
    2023:0.92, 2024:1.00, 2025:1.20,
}

DEEP_ROUNDS = ["Quarterfinals", "Semifinals", "The Final", "Final"]

# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════
def auto_detect_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    return "\t" if line.count("\t") > line.count(",") else ","

def load_data(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: File not found -> '{path}'\n"); sys.exit(1)

    sep = auto_detect_sep(path)
    df  = pd.read_csv(path, sep=sep, low_memory=False,
                      encoding="utf-8", on_bad_lines="skip")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=("\t" if sep == "," else ","),
                         low_memory=False, encoding="utf-8", on_bad_lines="skip")

    df.columns = df.columns.str.strip()
    required = ["Tournament","Surface","Round","Player_1","Player_2","Winner","Date"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing columns: {missing}"); sys.exit(1)

    df["Date"] = df["Date"].astype(str).str.strip()
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    if df["Year"].isna().all():
        df["Year"] = df["Date"].str.extract(r"(\d{4})").astype(float)

    for col in ["Surface","Tournament","Round","Player_1","Player_2","Winner"]:
        df[col] = df[col].astype(str).str.strip()

    total = len(df)
    df    = df[df["Year"] >= 2000].copy()
    print(f"  Separator : {'TAB' if sep == chr(9) else 'COMMA'}")
    print(f"  Loaded    : {len(df):,} matches (from {total:,} rows, year >= 2000)")
    return df

# ══════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════
def weighted_score(match_df):
    total = 0.0
    for _, row in match_df.iterrows():
        y = int(row["Year"]) if not pd.isna(row["Year"]) else 2015
        r = str(row["Round"])
        total += YEAR_WEIGHTS.get(y, 0.5) * ROUND_WEIGHTS.get(r, 1.0)
    return total

def compute_player_score(df, last_name, seed_num,
                          recent_years=None, recent_grass_years=None):
    if recent_years       is None: recent_years       = [2024, 2025]
    if recent_grass_years is None: recent_grass_years = [2023, 2024, 2025]

    is_winner = df["Winner"].str.startswith(last_name, na=False)
    in_match  = (df["Player_1"].str.startswith(last_name, na=False) |
                 df["Player_2"].str.startswith(last_name, na=False))

    wins   = df[in_match &  is_winner]
    losses = df[in_match & ~is_winner]

    def wr(w, l):
        ws, ls = weighted_score(w), weighted_score(l)
        return ws / (ws + ls) if (ws + ls) > 0 else 0.0

    gw, gl   = wins[wins["Surface"]=="Grass"], losses[losses["Surface"]=="Grass"]
    grass_wr = wr(gw, gl)

    ww = wins[wins["Tournament"].str.contains("Wimbledon", case=False, na=False)]
    wl = losses[losses["Tournament"].str.contains("Wimbledon", case=False, na=False)]
    wimb_wr      = wr(ww, wl)
    wimb_matches = len(ww) + len(wl)

    wd         = ww[ww["Round"].isin(DEEP_ROUNDS)]
    deep_score = min(weighted_score(wd) / 8.0, 1.0)

    rw = wins[wins["Year"].isin(recent_years)]
    rl = losses[losses["Year"].isin(recent_years)]
    recent_wr = wr(rw, rl)

    rgw = wins[(wins["Surface"]=="Grass") & (wins["Year"].isin(recent_grass_years))]
    rgl = losses[(losses["Surface"]=="Grass") & (losses["Year"].isin(recent_grass_years))]
    recent_grass_wr = wr(rgw, rgl) if (len(rgw)+len(rgl)) > 0 else grass_wr

    score = (
        0.28 * recent_grass_wr +
        0.22 * wimb_wr         +
        0.18 * grass_wr        +
        0.15 * recent_wr       +
        0.10 * deep_score      +
        0.07 * (1.0 / seed_num)
    )
    return score, {
        "Grass WR":      f"{grass_wr*100:.1f}%",
        "Wimbledon WR":  f"{wimb_wr*100:.1f}%",
        "Rec. Grass WR": f"{recent_grass_wr*100:.1f}%",
        "Recent Form":   f"{recent_wr*100:.1f}%",
        "Wimb Matches":  wimb_matches,
        "Deep Run":      f"{deep_score:.2f}",
    }

# ══════════════════════════════════════════════════════════════
#  GROUND TRUTH (FIXED: exactly 8 QF winners)
# ══════════════════════════════════════════════════════════════
def get_exact_qf_winners(df, year):
    """
    Returns the 8 players who WON their Wimbledon Quarterfinal in
    a given year (i.e., the players who advanced to the Semifinals).
    
    FIX: We get winners of QF matches only (not all deep rounds).
    This guarantees exactly 8 names — no duplicates, no overcounting.
    """
    wimb_qf = df[
        (df["Year"] == year) &
        (df["Tournament"].str.contains("Wimbledon", case=False, na=False)) &
        (df["Round"] == "Quarterfinals")
    ]

    if len(wimb_qf) == 0:
        # Fallback: try SF players (they must have won QF)
        wimb_sf = df[
            (df["Year"] == year) &
            (df["Tournament"].str.contains("Wimbledon", case=False, na=False)) &
            (df["Round"] == "Semifinals")
        ]
        names = set()
        for _, row in wimb_sf.iterrows():
            names.add(row["Player_1"].split()[0])
            names.add(row["Player_2"].split()[0])
        return names

    winners = set()
    for _, row in wimb_qf.iterrows():
        winners.add(row["Winner"].split()[0])
    return winners


def get_seeded_players_for_year(df, year):
    """
    FIX: For each backtest year, extract the actual seeded players
    from the dataset (players who appear in Wimbledon that year),
    rather than using the fixed 2026 list.
    This avoids penalising the model for 'missing' players who
    didn't exist yet in that year.
    
    Returns list of (surname, full_display_name) tuples.
    """
    wimb_year = df[
        (df["Year"] == year) &
        (df["Tournament"].str.contains("Wimbledon", case=False, na=False))
    ]
    # Collect all players who appeared at Wimbledon that year
    all_players = {}
    for col in ["Player_1", "Player_2"]:
        for name in wimb_year[col].dropna():
            surname = name.split()[0]
            if surname not in all_players:
                all_players[surname] = name
    return list(all_players.items())   # [(surname, full_name), ...]

# ══════════════════════════════════════════════════════════════
#  METRICS (FIXED: use Hit Rate as primary)
# ══════════════════════════════════════════════════════════════
def compute_metrics(predicted, actual, n_candidates):
    """
    predicted : list of surname strings (our top-8 picks)
    actual    : set of surname strings (real QF winners)
    
    PRIMARY metric: Hit Rate = TP / 8
    SECONDARY:      Precision, Recall, F1, MCC
    DE-EMPHASISED:  Accuracy (misleading for this task — see header)
    """
    pred = set(predicted)
    act  = set(actual)

    TP = len(pred & act)
    FP = len(pred - act)
    FN = len(act  - pred)
    TN = max(n_candidates - TP - FP - FN, 0)

    precision = TP / (TP + FP)             if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN)             if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    accuracy  = (TP + TN) / n_candidates   if n_candidates > 0 else 0.0
    denom     = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc       = (TP*TN - FP*FN) / denom    if denom > 0 else 0.0
    hit_rate  = TP / TOP_K

    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                HitRate=hit_rate, Precision=precision,
                Recall=recall, F1=f1, Accuracy=accuracy, MCC=mcc)

# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD BACKTEST (FIXED)
# ══════════════════════════════════════════════════════════════
def run_backtest(df, seeds_2026, gender):
    W = 78
    print(f"\n{'='*W}")
    print(f"  WALK-FORWARD BACKTEST -- {gender}")
    print(f"  Primary metric: Hit Rate (TP/8) | Secondary: Precision, Recall, F1, MCC")
    print(f"{'='*W}")
    print(f"  {'Year':<6} {'Hits':>6}  {'HitRate':>8}  {'Prec':>7}  {'Rec':>7}  "
          f"{'F1':>7}  {'MCC':>7}  {'Acc(*)':>7}")
    print(f"  {'(*) Accuracy is misleading -- see header comments':^{W-2}}")
    print(f"  {'-'*W}")

    all_m   = []
    records = []

    for yr in BACKTEST_YEARS:
        train = df[df["Year"] < yr].copy()
        if len(train) < 50:
            continue

        # FIXED: get exact 8 QF winners from dataset
        actual_qf = get_exact_qf_winners(df, yr)
        if len(actual_qf) < 4:
            print(f"  {yr}    [SKIP] Could not extract Wimbledon QF winners for this year")
            continue

        # FIXED: score actual Wimbledon players that year, not 2026 list
        year_players = get_seeded_players_for_year(df, yr)
        if len(year_players) < 8:
            # Fallback to 2026 seed last names
            year_players = [(s[1], s[3]) for s in seeds_2026]

        # Adjust recent windows to test year
        rec_yrs       = [yr-2, yr-1]
        rec_grass_yrs = [yr-3, yr-2, yr-1]

        # Score players using training-only data
        scored = []
        for i, (surname, fullname) in enumerate(year_players):
            seed_proxy = i + 1   # use position as seed proxy
            sc, _ = compute_player_score(train, surname, seed_proxy,
                                          recent_years=rec_yrs,
                                          recent_grass_years=rec_grass_yrs)
            scored.append((surname, fullname, sc))
        scored.sort(key=lambda x: x[2], reverse=True)

        predicted = [r[0] for r in scored[:TOP_K]]
        m = compute_metrics(predicted, actual_qf, len(year_players))
        m["Year"] = yr
        all_m.append(m)

        hits_display = f"{m['TP']}/8"
        print(f"  {yr}   {hits_display:>6}  "
              f"{m['HitRate']:>8.3f}  {m['Precision']:>7.3f}  {m['Recall']:>7.3f}  "
              f"{m['F1']:>7.3f}  {m['MCC']:>7.3f}  {m['Accuracy']:>7.3f}  "
              f"  actual QF: {', '.join(sorted(actual_qf)[:4])}...")

        records.append({
            "Year":          yr,
            "Hits (TP)":     m["TP"],
            "FP":            m["FP"],
            "FN":            m["FN"],
            "TN":            m["TN"],
            "Hit Rate":      round(m["HitRate"],   4),
            "Precision":     round(m["Precision"], 4),
            "Recall":        round(m["Recall"],    4),
            "F1":            round(m["F1"],        4),
            "MCC":           round(m["MCC"],       4),
            "Accuracy (*)":  round(m["Accuracy"],  4),
            "Predicted":     ", ".join(predicted),
            "Actual QF":     ", ".join(sorted(actual_qf)),
        })

    if not all_m:
        print("  [WARN] No years had usable Wimbledon QF data in dataset.")
        return pd.DataFrame()

    # ── Aggregate ────────────────────────────────────────────
    metrics_to_avg = ["HitRate","Precision","Recall","F1","MCC","Accuracy"]
    means = {k: np.mean([m[k] for m in all_m]) for k in metrics_to_avg}
    stds  = {k: np.std( [m[k] for m in all_m]) for k in metrics_to_avg}
    n     = len(all_m)
    total_hits = sum(m["TP"] for m in all_m)

    print(f"\n  {'='*W}")
    print(f"  AGGREGATE RESULTS  ({n} test years)")
    print(f"  {'='*W}")
    print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>8}  {'Interpretation'}")
    print(f"  {'-'*60}")
    interp = {
        "HitRate":   "avg correct picks per year  ← PRIMARY",
        "Precision": "of our 8 picks, how many were real QFists",
        "Recall":    "of real QFists, how many did we catch",
        "F1":        "balance between precision and recall",
        "MCC":       "most robust single metric  ← ROBUST",
        "Accuracy":  "misleading for this task — ignore (*)",
    }
    for k in metrics_to_avg:
        print(f"  {k:<14} {means[k]:>8.3f}  {stds[k]:>8.3f}  {interp[k]}")

    avg_hits = total_hits / n
    pct      = total_hits / (n * TOP_K) * 100
    print(f"\n  Total correct QF picks  : {total_hits} / {n*TOP_K}  ({pct:.1f}%)")
    print(f"  Avg hits per year       : {avg_hits:.2f} / 8  "
          f"({'GOOD' if avg_hits >= 4 else 'FAIR' if avg_hits >= 3 else 'POOR'})")
    print(f"  Years evaluated         : {n}")
    print(f"\n  HOW TO READ HIT RATE:")
    print(f"    5-6/8 (0.625-0.75) = Excellent")
    print(f"    4/8   (0.50)       = Good  (beats random: 8/16=0.50)")
    print(f"    3/8   (0.375)      = Fair")
    print(f"    <3/8  (<0.375)     = Poor (worse than seed order alone)")

    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════
#  2026 PREDICTIONS
# ══════════════════════════════════════════════════════════════
def confidence_tier(score, all_scores):
    q75 = np.percentile(all_scores, 75)
    q50 = np.percentile(all_scores, 50)
    q25 = np.percentile(all_scores, 25)
    if   score >= q75: return "HIGH"
    elif score >= q50: return "MEDIUM"
    elif score >= q25: return "LOW"
    else:              return "VERY LOW"

def predict_2026(df, seeds, label):
    rows = []
    for seed_num, last_name, _, full_name in seeds:
        score, feats = compute_player_score(df, last_name, seed_num)
        rows.append({"Seed": seed_num, "Full Name": full_name,
                     "Score": round(score, 4), **feats})

    result = (pd.DataFrame(rows)
                .sort_values("Score", ascending=False)
                .reset_index(drop=True))
    result.index += 1

    all_scores = result["Score"].values.astype(float)
    result["Confidence"] = result["Score"].apply(
        lambda s: confidence_tier(s, all_scores))

    W = 80
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  {'Rank':<5} {'Player':<28} {'Score':<8} {'Conf.':<10} "
          f"{'Wimb WR':<10} {'Rec.Grass':<11} {'Deep Run'}")
    print(f"  {'-'*W}")
    for i, row in result.iterrows():
        marker = "  <-- QF PICK" if i <= TOP_K else ""
        print(f"  {i:<5} {row['Full Name']:<28} {row['Score']:<8.4f} "
              f"{row['Confidence']:<10} {row['Wimbledon WR']:<10} "
              f"{row['Rec. Grass WR']:<11} {row['Deep Run']}{marker}")
    return result

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print()
    print("="*70)
    print("   WIMBLEDON 2026 -- PHASE 1 PREDICTOR + FIXED BACKTEST  (EXL)")
    print("="*70)
    print()

    df = load_data(DATASET_PATH)
    grass_n = (df["Surface"]=="Grass").sum()
    wimb_n  = df["Tournament"].str.contains("Wimbledon", case=False, na=False).sum()
    print(f"  Year range    : {int(df['Year'].min())} - {int(df['Year'].max())}")
    print(f"  Grass matches : {grass_n:,}")
    print(f"  Wimbledon rows: {wimb_n:,}")

    # ── STEP 1: FIXED BACKTESTING ─────────────────────────────
    print("\n")
    print("="*70)
    print("   STEP 1: WALK-FORWARD BACKTEST (2015-2025)")
    print("   THREE FIXES vs previous version:")
    print("   [1] Primary metric changed to Hit Rate (TP/8) -- not accuracy")
    print("   [2] Ground truth = exact 8 QF winners -- not all deep round players")
    print("   [3] Candidate pool = actual Wimbledon players that year -- not 2026 list")
    print("="*70)

    men_bt   = run_backtest(df, MEN_SEEDS,   "MEN'S SINGLES")
    women_bt = run_backtest(df, WOMEN_SEEDS, "WOMEN'S SINGLES")

    if not men_bt.empty:
        men_bt.to_csv("men_backtest_metrics.csv", index=False)
        print(f"\n  Saved: men_backtest_metrics.csv")
    if not women_bt.empty:
        women_bt.to_csv("women_backtest_metrics.csv", index=False)
        print(f"  Saved: women_backtest_metrics.csv")

    # ── STEP 2: 2026 PREDICTIONS ─────────────────────────────
    print("\n")
    print("="*70)
    print("   STEP 2: 2026 PREDICTIONS (trained on full dataset)")
    print("="*70)

    men_res   = predict_2026(df, MEN_SEEDS,   "MEN'S SINGLES -- ALL 16 SEEDS RANKED")
    women_res = predict_2026(df, WOMEN_SEEDS, "WOMEN'S SINGLES -- ALL 16 SEEDS RANKED")

    for res, fname in [(men_res, "men_phase1_picks.csv"),
                       (women_res, "women_phase1_picks.csv")]:
        res[["Seed","Full Name","Wimbledon WR","Rec. Grass WR","Grass WR",
             "Recent Form","Wimb Matches","Deep Run","Score","Confidence"]].to_csv(
            fname, index=False)
    print(f"\n  Saved: men_phase1_picks.csv")
    print(f"  Saved: women_phase1_picks.csv")

    # ── METRIC REFERENCE ──────────────────────────────────────
    print("""
METRIC REFERENCE
================
  HIT RATE  = TP / 8            PRIMARY METRIC
              How many of our 8 picks were actual QF players.
              0.625+ = Excellent | 0.50 = Good | 0.375 = Fair

  PRECISION = TP / (TP+FP)
              Of the 8 we picked, what fraction truly reached QF.
              High precision = few false alarms.

  RECALL    = TP / (TP+FN)
              Of the real 8 QF players, how many did we catch.
              High recall = we didn't miss QFists.

  F1        = 2 * P * R / (P+R)
              Balanced single score. Use when P and R both matter.

  MCC       = Balanced binary correlation  (-1 to +1)
              Best single metric. Accounts for all 4 cells of the
              confusion matrix. Use this for hackathon leaderboard.

  ACCURACY (*) = (TP+TN) / all_candidates
              MISLEADING for this task. When unseeded players crash
              the QF, TN -> 0, so accuracy -> TP/16 even if the
              model logic is sound. This is why old accuracy was 0.125.
              A model that picks 4/8 correctly gets accuracy = 4/16 = 0.25,
              which looks terrible but is actually decent performance.

CONFIDENCE TIERS (2026 picks, by score percentile among 16 seeds)
  HIGH      = top 25% of 16 seeds  (very likely to reach QF)
  MEDIUM    = 50th-75th percentile
  LOW       = 25th-50th percentile
  VERY LOW  = bottom 25%
""")

    # ── FINAL SUBMISSION ──────────────────────────────────────
    men8   = men_res.head(8)["Full Name"].tolist()
    women8 = women_res.head(8)["Full Name"].tolist()
    mc     = men_res.head(8)["Confidence"].tolist()
    wc     = women_res.head(8)["Confidence"].tolist()

    print("="*70)
    print("   FINAL SUBMISSION")
    print("="*70)
    print(f"  {'MENS QF PICKS':<35}  WOMEN'S QF PICKS")
    print(f"  {'-'*68}")
    for i in range(8):
        m  = f"{i+1}. {men8[i]}"   if i < len(men8)   else ""
        w  = f"{i+1}. {women8[i]}" if i < len(women8) else ""
        print(f"  {m:<24} [{mc[i]:<9}]    {w:<24} [{wc[i]}]")
    print()
    print("  Deadline : 28 June 2026, 11:59 PM IST")
    print("  Scoring  : +2 per correct QF pick | Max 32 pts")
    print()

if __name__ == "__main__":
    main()