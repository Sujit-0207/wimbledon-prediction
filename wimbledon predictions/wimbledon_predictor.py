"""
wimbledon_predictor.py
Predicts Wimbledon 2026 Quarterfinalists - 8 men, 8 women.

Data scope: ALL event types are included (main tour, qualifying tournaments,
challenger events) so that qualifying-round performances feed into grass win
rates and recent form. Only round-robin and bronze-medal matches are excluded.

Model features (composite score):
  35% current ranking          - overall player quality
  22% recent grass win rate    - surface-specific form (2024+), all event tiers
  15% Wimbledon historical avg - Wimbledon-specific ability (all-time)
  13% all-time grass win rate  - grass consistency across all event tiers
   8% best Wimbledon run 2024+ - recent Wimbledon ceiling
   7% recent overall form      - last 14 months win rate
"""
import re
import pandas as pd
import numpy as np
import os
import glob as gl

DATA_DIR   = r"D:\wimbledon predictions"
FILES_DIR  = os.path.join(DATA_DIR, "files")
FILES2_DIR = os.path.join(DATA_DIR, "files2")
OUTPUT     = os.path.join(DATA_DIR, "predicted_quarterfinalists.csv")

LOAD_COLS = ["tourney_name", "surface", "tourney_date", "tourney_level",
             "winner_name", "winner_rank", "loser_name", "loser_rank",
             "round", "event_type"]

REF_DATE   = pd.Timestamp("2026-06-23")
RECENT_CUT = pd.Timestamp("2024-01-01")   # grass & Wimbledon "recent" window
ACTIVE_CUT = pd.Timestamp("2025-01-01")   # must have played at least once since here
FORM_CUT   = REF_DATE - pd.DateOffset(months=14)

# All event tiers included - qualifying/challenger rounds are valid form signals.
ALL_EVENT_TYPES = {
    "main_tour", "main_tour_ongoing_2026",
    "legacy_odds_format", "kaggle_historical",
    "qualifying", "challenger", "challenger_ongoing_2026",
}
# Only skip round-robin (group stage) and bronze-medal matches - not meaningful
# for predicting single-elimination Wimbledon QF performance.
SKIP_ROUNDS = {"RR", "BR"}

# Depth points per round: loser gets base pts, winner gets base+1 (cap 7)
# Both abbreviated (ATP schema) and full-word (legacy-odds / WTA) forms are mapped.
# Keys are compared uppercase (the code applies .str.upper() before lookup).
ROUND_PTS = {
    # abbreviated
    "R128": 1, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6, "W": 7,
    # full-word (legacy odds / WTA format)
    "1ST ROUND": 1, "2ND ROUND": 1, "3RD ROUND": 2, "4TH ROUND": 3,
    "QUARTERFINALS": 4, "QUARTER-FINALS": 4,
    "SEMIFINALS": 5,    "SEMI-FINALS": 5,
    "FINAL": 6,         "THE FINAL": 6,
}

FEAT_COLS = ["grass_wr_recent", "grass_wr_all", "wimb_avg_all", "wimb_best_rec", "form_wr"]


# -- I/O helpers ---------------------------------------------------------------

def load_csvs(pattern):
    files = sorted(gl.glob(pattern))
    print(f"  reading {len(files)} part-files ... ", end="", flush=True)
    dfs = [pd.read_csv(f, usecols=LOAD_COLS, low_memory=False) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df["winner_rank"]  = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"]   = pd.to_numeric(df["loser_rank"],  errors="coerce")
    print(f"{len(df):,} rows")
    return df


def parse_wta_name(s):
    """'Last, First' -> 'First Last'"""
    parts = str(s).split(",", 1)
    return (parts[1].strip() + " " + parts[0].strip()) if len(parts) == 2 else s


# -- Name deduplication --------------------------------------------------------

def name_key(name):
    """
    Normalise both 'Last F.' and 'First Last' formats to (surname_word, initial)
    so that duplicates across ATP full-names and legacy-odds abbreviations collapse.
    Uses the final word of the surname to handle particles: 'De Minaur A.' == 'Alex de Minaur'.
    """
    name = str(name).strip()
    # Pattern: "Alcaraz C." or "De Minaur A."
    m = re.match(r'^(.+?)\s+([A-Z])\.$', name)
    if m:
        surname_word = m.group(1).split()[-1].lower()   # last word only, drops 'De'/'Van' etc.
        return (surname_word, m.group(2).lower())
    # Pattern: "Carlos Alcaraz" or "Alex de Minaur"
    parts = name.split()
    if len(parts) >= 2:
        return (parts[-1].lower(), parts[0][0].lower())
    return (name.lower(), "")


def dedup_candidates(sc):
    """
    Within a candidates DataFrame, collapse rows that represent the same player
    under different name formats (e.g. ATP full-name vs legacy-odds abbreviated).
    Keeps the row with the most non-null feature values; ties broken by lowest rank.
    """
    sc = sc.copy()
    sc["_key"]    = sc["player"].apply(name_key)
    sc["_n_feat"] = sc[FEAT_COLS].notna().sum(axis=1)
    sc["_name_len"] = sc["player"].str.len()
    sc = (sc.sort_values(["_n_feat", "_name_len", "current_rank"], ascending=[False, False, True])
            .drop_duplicates("_key")
            .drop(columns=["_key", "_n_feat", "_name_len"]))
    return sc.reset_index(drop=True)


# -- Feature builders ----------------------------------------------------------

def filter_matches(df):
    """Keep all competitive match rows; drop only round-robin and bronze-medal rounds."""
    df = df[df["event_type"].isin(ALL_EVENT_TYPES)].copy()
    rnd = df["round"].fillna("").str.upper().str.strip()
    return df[~rnd.isin(SKIP_ROUNDS)]


def active_players(df):
    """Players who appeared in at least one match since ACTIVE_CUT."""
    r = df[df["tourney_date"] >= ACTIVE_CUT]
    return set(r["winner_name"].dropna()) | set(r["loser_name"].dropna())


def rank_map(df):
    """Most recent ranking for each player (from RECENT_CUT onwards)."""
    sub = df[df["tourney_date"] >= RECENT_CUT]
    w = sub[["winner_name", "winner_rank", "tourney_date"]].rename(
        columns={"winner_name": "p", "winner_rank": "r"})
    l = sub[["loser_name", "loser_rank", "tourney_date"]].rename(
        columns={"loser_name": "p", "loser_rank": "r"})
    all_ = pd.concat([w, l]).dropna(subset=["p", "r"]).sort_values("tourney_date", ascending=False)
    return all_.drop_duplicates("p").set_index("p")["r"]


def win_rates(df):
    """Win rate per player for a (pre-filtered) match DataFrame."""
    w = df["winner_name"].value_counts().rename("W")
    l = df["loser_name"].value_counts().rename("L")
    s = pd.concat([w, l], axis=1).fillna(0)
    s["n"]  = s["W"] + s["L"]
    s["wr"] = s["W"] / s["n"]
    return s


def wimbledon_depth(df):
    """
    Per player x Wimbledon edition: deepest round reached (0-7 pts).
    Returns DataFrame indexed by player with columns wimb_avg, wimb_best.
    """
    sub = df[df["tourney_name"].str.contains("Wimbledon", case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["wimb_avg", "wimb_best"])

    sub["bp"] = (sub["round"].fillna("").str.upper().str.strip()
                 .map(ROUND_PTS).fillna(0).astype(int))
    sub["yr"] = sub["tourney_date"].dt.year.astype(str)

    w_rows = sub[["winner_name", "yr", "bp"]].rename(columns={"winner_name": "player"}).copy()
    w_rows["pts"] = (w_rows["bp"] + 1).clip(upper=7)

    l_rows = sub[["loser_name", "yr", "bp"]].rename(columns={"loser_name": "player"}).copy()
    l_rows["pts"] = l_rows["bp"]

    rdf = pd.concat([w_rows[["player", "yr", "pts"]],
                     l_rows[["player", "yr", "pts"]]]).dropna(subset=["player"])
    per_ed = rdf.groupby(["player", "yr"])["pts"].max().reset_index()
    agg = per_ed.groupby("player")["pts"].agg(wimb_avg="mean", wimb_best="max")
    return agg   # index = player name


def norm(s):
    """Min-max normalise a Series; NaN filled with median before normalising."""
    s = s.copy().fillna(s.median() if s.notna().any() else 0)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)


# -- Core predictor ------------------------------------------------------------

def predict(df, label, extra_ranks=None):
    """Build composite scores and return the top-8 quarterfinalist predictions."""
    print(f"\n[{label}]")
    mt = filter_matches(df)
    print(f"  Main-tour rows : {len(mt):,}")
    active = active_players(mt)
    print(f"  Active players : {len(active):,}")

    rmap_   = rank_map(mt)
    grass   = mt[mt["surface"] == "Grass"]
    gr_rec  = win_rates(grass[grass["tourney_date"] >= RECENT_CUT])
    gr_all  = win_rates(grass)
    form    = win_rates(mt[mt["tourney_date"] >= FORM_CUT])
    wd_all  = wimbledon_depth(mt)
    wd_rec  = wimbledon_depth(mt[mt["tourney_date"] >= RECENT_CUT])

    rows = []
    for p in active:
        rank = (extra_ranks or {}).get(p)
        if rank is None and p in rmap_.index:
            rank = rmap_[p]
        rows.append({
            "player"          : p,
            "gender"          : label,
            "current_rank"    : rank,
            "grass_wr_recent" : gr_rec.loc[p, "wr"] if p in gr_rec.index else np.nan,
            "grass_n_recent"  : gr_rec.loc[p, "n"]  if p in gr_rec.index else 0,
            "grass_wr_all"    : gr_all.loc[p, "wr"] if p in gr_all.index else np.nan,
            "wimb_avg_all"    : wd_all.loc[p, "wimb_avg"]  if p in wd_all.index else np.nan,
            "wimb_best_rec"   : wd_rec.loc[p, "wimb_best"] if p in wd_rec.index else np.nan,
            "form_wr"         : form.loc[p, "wr"] if p in form.index else np.nan,
            "form_n"          : form.loc[p, "n"]  if p in form.index else 0,
        })

    sc = pd.DataFrame(rows)

    # candidates: top-150 ranked OR at least 3 recent grass matches
    sc = sc[(sc["current_rank"].fillna(999) <= 150) | (sc["grass_n_recent"] >= 3)]

    # collapse same player appearing under different name formats
    sc = dedup_candidates(sc)
    print(f"  Candidates     : {len(sc):,} (after name dedup)")

    sc["rank_score"] = 1 - (sc["current_rank"].clip(upper=200).fillna(200) - 1) / 199

    sc["composite"] = (
        0.35 * sc["rank_score"]            +
        0.22 * norm(sc["grass_wr_recent"])  +
        0.13 * norm(sc["grass_wr_all"])     +
        0.15 * norm(sc["wimb_avg_all"])     +
        0.08 * norm(sc["wimb_best_rec"])    +
        0.07 * norm(sc["form_wr"])
    )

    top8 = sc.nlargest(8, "composite").reset_index(drop=True)
    top8.insert(0, "qf_seed", range(1, 9))
    return top8


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading men's matches ...")
    men = load_csvs(os.path.join(FILES_DIR, "men_matches_combined_part*.csv"))

    print("\nLoading women's matches ...")
    women = load_csvs(os.path.join(FILES_DIR, "women_matches_combined_part*.csv"))

    wta = pd.read_csv(os.path.join(FILES2_DIR, "women_rankings_current.csv"))
    wta["pname"] = wta["name"].apply(parse_wta_name)
    extra_ranks = dict(zip(wta["pname"], wta["rank"]))
    print(f"\nWTA ranking entries loaded : {len(extra_ranks)}")

    men_qf   = predict(men,   "Men")
    women_qf = predict(women, "Women", extra_ranks=extra_ranks)

    KEEP = ["qf_seed", "player", "gender", "current_rank",
            "grass_wr_recent", "wimb_avg_all", "form_wr", "composite"]
    out = pd.concat([men_qf[KEEP], women_qf[KEEP]], ignore_index=True)

    # save CSV
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")

    # print results
    print("\n" + "=" * 60)
    print("   WIMBLEDON 2026  -  PREDICTED QUARTERFINALISTS")
    print("=" * 60)
    for g in ["Men", "Women"]:
        print(f"\n  {g.upper()}")
        print(f"  {'#':<5}{'Player':<30}{'Rank':<7}{'Score'}")
        print("  " + "-" * 52)
        sub = out[out["gender"] == g].reset_index(drop=True)
        for i, row in sub.iterrows():
            rk = str(int(row["current_rank"])) if pd.notna(row["current_rank"]) else "N/A"
            print(f"  {i+1:<5}{row['player']:<30}{rk:<7}{row['composite']:.4f}")
    print()
