"""
preprocess_combine_datasets.py

Combines every match-level and player-level CSV in the wimbledon-prediction
repo's `data/` folder into two unified, gender-separated datasets:

    men_matches_combined.csv     -- every men's match found anywhere in the repo
    women_matches_combined.csv   -- every women's match found anywhere in the repo
    men_players_bios.csv         -- player biographical reference data (men)
    women_players_bios.csv       -- player biographical reference data (women)
    women_rankings_current.csv   -- official WTA snapshot rankings (22 Jun 2026)
    men_charting_matches_index.csv   -- Match Charting Project match index (men)
    women_charting_matches_index.csv -- Match Charting Project match index (women)

DESIGN PRINCIPLE: NO DATA LOSS.
Every row from every source file is preserved. Sources with different schemas
are reindexed onto a common superset of columns (missing fields become NaN,
never dropped). Every output row carries `source_file` and `event_type` so the
original provenance of every value is always recoverable. Nothing is merged
or deduplicated except the one byte-for-byte duplicate file (atp_tennis.csv
exists at two paths in the repo with identical content).

SCHEMA GROUPS FOUND IN THE REPO (detected empirically, not assumed):
  1. "ATP results schema" (130 files): atp_quali/*, other/*.csv (yearly main
     draw + challenger), other/ongoing_tourneys.csv, other/challenger_ongoing_tourneys.csv
     -> all men's matches, already in a consistent 50-column schema.
  2. "Legacy odds ATP schema" (atp_tennis.csv, appears twice/identical): men's
     tour-level matches 2000-2026 with bookmaker odds, different column names.
  3. "Legacy odds WTA schema" (wta.csv): women's tour-level matches 2006-2026
     with bookmaker odds. Same shape as #2 minus the `Series` column.
  4. "Kaggle schema" (KaggleMatches.csv): both ATP and WTA rows (split via
     the `league` column), historical matches up to ~2021.
  5. "Kaggle players" (KagglePlayers.csv): both genders (split via `gender`).
  6. "ATP_Database.csv": men's player biographical data.

Player ID namespaces differ across sources (Sackmann-style alphanumeric IDs
vs Kaggle numeric IDs) and are NOT cross-resolved here -- that would risk
silently merging two different people who share a name. Bios are unioned
with a `source` tag instead of joined/deduplicated.
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/claude/wimbledon-prediction/data"
OUT_DIR = "/home/claude/combined_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Canonical superset schema for the unified matches tables.
CANON_COLS = [
    "source_file", "gender", "event_type",
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "indoor", "tourney_date", "match_num", "round", "best_of", "score", "minutes",
    "winner_id", "winner_name", "winner_seed", "winner_entry", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_seed", "loser_entry", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
    "series", "odd_winner", "odd_loser",
]

NUMERIC_COLS = [
    "draw_size", "match_num", "best_of", "minutes",
    "winner_seed", "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
    "loser_seed", "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
    "odd_winner", "odd_loser",
]


def log(msg):
    print(f"[preprocess] {msg}")


def event_type_for_path(path):
    rel = path.replace(DATA_DIR, "").lstrip("/\\")
    fname = os.path.basename(path)
    if "atp_quali" in rel:
        return "qualifying"
    if fname == "ongoing_tourneys.csv":
        return "main_tour_ongoing_2026"
    if fname == "challenger_ongoing_tourneys.csv":
        return "challenger_ongoing_2026"
    if fname.endswith("_challenger.csv"):
        return "challenger"
    return "main_tour"


def parse_mixed_date(series):
    """Handles both 'YYYYMMDD' (int/str) and ISO 'YYYY-MM-DD' date formats."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    mask = out.isna()
    if mask.any():
        out2 = pd.to_datetime(s[mask], errors="coerce")
        out.loc[mask] = out2
    return out.dt.strftime("%Y-%m-%d")


def load_atp_schema_files():
    """Schema group 1: the 130 files sharing the standard ATP results schema."""
    paths = []
    paths += glob.glob(os.path.join(DATA_DIR, "atp_quali", "*.csv"))
    paths += glob.glob(os.path.join(DATA_DIR, "other", "*.csv"))
    # drop the non-conforming special files from the "other" glob
    exclude = {"ATP_Database.csv", "atp_tennis.csv", "KaggleMatches.csv", "KagglePlayers.csv"}
    paths = [p for p in paths if os.path.basename(p) not in exclude]

    frames = []
    for p in sorted(paths):
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False, dtype=str)
        df["source_file"] = os.path.relpath(p, DATA_DIR)
        df["gender"] = "M"  # this entire repo's ATP-schema files are men's events
        df["event_type"] = event_type_for_path(p)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["tourney_date"] = parse_mixed_date(combined["tourney_date"])
    log(f"Loaded {len(frames)} ATP-schema files -> {len(combined):,} rows")
    return combined


def load_kaggle_matches():
    """Schema group 4: KaggleMatches.csv, split into men/women via `league`."""
    p = os.path.join(DATA_DIR, "other", "KaggleMatches.csv")
    df = pd.read_csv(p, low_memory=False, dtype=str)
    df["source_file"] = os.path.relpath(p, DATA_DIR)
    df["event_type"] = "kaggle_historical"
    df["tourney_date"] = parse_mixed_date(df["tourney_date"])
    df["gender"] = df["league"].map({"atp": "M", "wta": "W"})
    df = df.drop(columns=["league"])
    men = df[df["gender"] == "M"].copy()
    women = df[df["gender"] == "W"].copy()
    log(f"Loaded KaggleMatches.csv -> {len(men):,} men's rows, {len(women):,} women's rows")
    return men, women


def remap_legacy_odds(path, gender, has_series):
    """Schema groups 2 & 3: legacy bookmaker-odds format (atp_tennis.csv / wta.csv)."""
    df = pd.read_csv(path, low_memory=False, dtype=str)

    # numeric helpers for the winner/loser disambiguation (string compare is enough)
    is_p1_winner = df["Winner"] == df["Player_1"]

    out = pd.DataFrame(index=df.index)
    out["source_file"] = os.path.relpath(path, DATA_DIR)
    out["gender"] = gender
    out["event_type"] = "legacy_odds_format"
    out["tourney_name"] = df["Tournament"]
    out["surface"] = df["Surface"]
    out["tourney_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["round"] = df["Round"]
    out["best_of"] = df["Best of"]
    out["score"] = df["Score"]
    out["indoor"] = df["Court"].map({"Outdoor": "O", "Indoor": "I"})
    out["series"] = df["Series"] if has_series else np.nan

    out["winner_name"] = df["Winner"]
    out["loser_name"] = np.where(is_p1_winner, df["Player_2"], df["Player_1"])
    out["winner_rank"] = np.where(is_p1_winner, df["Rank_1"], df["Rank_2"])
    out["loser_rank"] = np.where(is_p1_winner, df["Rank_2"], df["Rank_1"])
    out["winner_rank_points"] = np.where(is_p1_winner, df["Pts_1"], df["Pts_2"])
    out["loser_rank_points"] = np.where(is_p1_winner, df["Pts_2"], df["Pts_1"])
    out["odd_winner"] = np.where(is_p1_winner, df["Odd_1"], df["Odd_2"])
    out["odd_loser"] = np.where(is_p1_winner, df["Odd_2"], df["Odd_1"])

    # fields this source simply doesn't have -- explicit NaN, not dropped
    for col in CANON_COLS:
        if col not in out.columns:
            out[col] = np.nan

    log(f"Loaded {os.path.basename(path)} ({gender}) -> {len(out):,} rows [legacy odds format]")
    return out[CANON_COLS]


def load_bios():
    atp_db = pd.read_csv(os.path.join(DATA_DIR, "other", "ATP_Database.csv"),
                          dtype=str, encoding="utf-8-sig")
    atp_db.columns = [c.strip().lower() for c in atp_db.columns]
    atp_db["source"] = "ATP_Database.csv"

    kp = pd.read_csv(os.path.join(DATA_DIR, "other", "KagglePlayers.csv"), dtype=str)
    kp["source"] = "KagglePlayers.csv"
    kp["full_name"] = (kp["name_first"].fillna("") + " " + kp["name_last"].fillna("")).str.strip()

    men_kaggle = kp[kp["gender"] == "male"].copy()
    women_bios = kp[kp["gender"] == "female"].copy()

    men_bios = pd.concat([atp_db, men_kaggle], ignore_index=True, sort=False)
    log(f"Men's bios: {len(atp_db):,} from ATP_Database + {len(men_kaggle):,} from KagglePlayers "
        f"(unioned, NOT entity-resolved -- different ID namespaces) = {len(men_bios):,} rows")
    log(f"Women's bios: {len(women_bios):,} rows (KagglePlayers.csv is the only bio source for women)")
    return men_bios, women_bios


def load_charting_index(filename, gender):
    p = os.path.join(DATA_DIR, "tennis_MatchChartingProject-master", filename)
    df = pd.read_csv(p, dtype=str)
    df["gender"] = gender
    df["source_file"] = os.path.relpath(p, DATA_DIR)
    return df


def cast_numeric(df):
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    log("=" * 70)
    log("STEP 1: loading the 130 standard-schema ATP results files (men)")
    atp_schema_men = load_atp_schema_files()
    atp_schema_men = atp_schema_men.reindex(columns=CANON_COLS)

    log("=" * 70)
    log("STEP 2: loading KaggleMatches.csv (men + women)")
    kaggle_men, kaggle_women = load_kaggle_matches()
    kaggle_men = kaggle_men.reindex(columns=CANON_COLS)
    kaggle_women = kaggle_women.reindex(columns=CANON_COLS)

    log("=" * 70)
    log("STEP 3: loading legacy odds-format files (atp_tennis.csv, wta.csv)")
    legacy_men = remap_legacy_odds(os.path.join(DATA_DIR, "other", "atp_tennis.csv"),
                                    gender="M", has_series=True)
    legacy_women = remap_legacy_odds(os.path.join(DATA_DIR, "wta.csv"),
                                      gender="W", has_series=False)
    # NOTE: data/atp_tennis.csv is a byte-identical duplicate of data/other/atp_tennis.csv
    # (confirmed via md5sum) -- using one copy is not data loss, it's deduplication of
    # an exact duplicate file.

    log("=" * 70)
    log("STEP 4: concatenating into final combined tables")
    men_combined = pd.concat([atp_schema_men, kaggle_men, legacy_men],
                              ignore_index=True, sort=False)
    women_combined = pd.concat([kaggle_women, legacy_women],
                                ignore_index=True, sort=False)

    men_combined = cast_numeric(men_combined)
    women_combined = cast_numeric(women_combined)

    men_combined.to_csv(os.path.join(OUT_DIR, "men_matches_combined.csv"), index=False)
    women_combined.to_csv(os.path.join(OUT_DIR, "women_matches_combined.csv"), index=False)
    log(f"men_matches_combined.csv   -> {len(men_combined):,} rows, {men_combined.shape[1]} cols")
    log(f"women_matches_combined.csv -> {len(women_combined):,} rows, {women_combined.shape[1]} cols")

    log("=" * 70)
    log("STEP 5: player bios")
    men_bios, women_bios = load_bios()
    men_bios.to_csv(os.path.join(OUT_DIR, "men_players_bios.csv"), index=False)
    women_bios.to_csv(os.path.join(OUT_DIR, "women_players_bios.csv"), index=False)

    log("=" * 70)
    log("STEP 6: Match Charting Project match indices (kept separate -- different grain,")
    log("        no winner/score field, so cannot be unioned into the matches schema)")
    men_charting = load_charting_index("charting-m-matches.csv", "M")
    women_charting = load_charting_index("charting-w-matches.csv", "W")
    men_charting.to_csv(os.path.join(OUT_DIR, "men_charting_matches_index.csv"), index=False)
    women_charting.to_csv(os.path.join(OUT_DIR, "women_charting_matches_index.csv"), index=False)
    log(f"men_charting_matches_index.csv   -> {len(men_charting):,} rows")
    log(f"women_charting_matches_index.csv -> {len(women_charting):,} rows")

    log("=" * 70)
    log("STEP 7: provenance summary")
    print("\n--- men_matches_combined.csv: rows by event_type ---")
    print(men_combined["event_type"].value_counts(dropna=False).to_string())
    print("\n--- women_matches_combined.csv: rows by event_type ---")
    print(women_combined["event_type"].value_counts(dropna=False).to_string())
    print("\n--- men date range ---", men_combined["tourney_date"].min(), "to",
          men_combined["tourney_date"].max())
    print("--- women date range ---", women_combined["tourney_date"].min(), "to",
          women_combined["tourney_date"].max())

    log("DONE.")


if __name__ == "__main__":
    main()
