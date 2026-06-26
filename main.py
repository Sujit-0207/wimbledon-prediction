"""
Wimbledon 2026 — Top-8 Seed Prediction System
================================================
Computes a composite "grass form" score for the official top-8 seeds
(men's & women's) using historical ATP/WTA match data, and prints a
ranked prediction table + projected finalists.

Usage:
    python wimbledon_2026_prediction.py

Expects atp_tennis.csv and wta.csv (same folder, or edit paths below).
Each file must have columns: Tournament, Date, Surface, Player_1,
Player_2, Winner.
"""

import pandas as pd

# ----------------------------------------------------------------------
# 1. CONFIG — paths + official Wimbledon 2026 top-8 seeds
#    (seeding cut-off: ATP/WTA rankings as of 22 June 2026)
# ----------------------------------------------------------------------

ATP_CSV = "atp_tennis.csv"
WTA_CSV = "wta.csv"

MEN_SEEDS = {
    1: "Sinner J.",
    2: "Zverev A.",
    3: "Auger-Aliassime F.",
    4: "Shelton B.",
    5: "De Minaur A.",
    6: "Fritz T.",
    7: "Djokovic N.",
    8: "Medvedev D.",
}

WOMEN_SEEDS = {
    1: "Sabalenka A.",
    2: "Rybakina E.",
    3: "Swiatek I.",
    4: "Pegula J.",
    5: "Andreeva M.",
    6: "Anisimova A.",
    7: "Gauff C.",
    8: "Svitolina E.",
}

# Weighting of the three signals in the composite score. Must sum to 1.0
WEIGHTS = {
    "grass_pct": 0.40,   # win% on grass, last 2 years
    "recent_pct": 0.35,  # win% over last N matches (any surface) - current form
    "wimb_pct": 0.25,    # career win% specifically at Wimbledon
}

RECENT_N = 25                # how many recent matches define "current form"
GRASS_SINCE = "2024-01-01"   # lookback window for grass-court win%


# ----------------------------------------------------------------------
# 2. DATA LOADING
# ----------------------------------------------------------------------

def load_matches(path):
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Player_1"] = df["Player_1"].str.strip()
    df["Player_2"] = df["Player_2"].str.strip()
    df["Winner"] = df["Winner"].astype(str).str.strip()
    return df


# ----------------------------------------------------------------------
# 3. PER-PLAYER STATS
# ----------------------------------------------------------------------

def player_stats(df, player, surface="Grass", since=GRASS_SINCE, recent_n=RECENT_N):
    """Return grass win%, recent-form win%, and Wimbledon win% for a player."""
    involved = df[(df["Player_1"] == player) | (df["Player_2"] == player)]
    involved = involved[involved["Date"] >= pd.Timestamp(since)]

    # Grass win% (since `since`)
    grass = involved[involved["Surface"] == surface]
    grass_w, grass_t = (grass["Winner"] == player).sum(), len(grass)
    grass_pct = (grass_w / grass_t * 100) if grass_t else 0.0

    # Recent form (last N matches, any surface)
    recent = involved.sort_values("Date").tail(recent_n)
    recent_w, recent_t = (recent["Winner"] == player).sum(), len(recent)
    recent_pct = (recent_w / recent_t * 100) if recent_t else 0.0

    # Wimbledon career win%
    wimb = involved[involved["Tournament"].str.contains("Wimbledon", case=False, na=False)]
    wimb_w, wimb_t = (wimb["Winner"] == player).sum(), len(wimb)
    wimb_pct = (wimb_w / wimb_t * 100) if wimb_t else 0.0

    return {
        "grass_w": grass_w, "grass_t": grass_t, "grass_pct": grass_pct,
        "recent_w": recent_w, "recent_t": recent_t, "recent_pct": recent_pct,
        "wimb_w": wimb_w, "wimb_t": wimb_t, "wimb_pct": wimb_pct,
    }


def composite_score(stats, weights=WEIGHTS):
    return round(
        weights["grass_pct"] * stats["grass_pct"]
        + weights["recent_pct"] * stats["recent_pct"]
        + weights["wimb_pct"] * stats["wimb_pct"],
        1,
    )


# ----------------------------------------------------------------------
# 4. BUILD RANKING TABLE FOR A SEED LIST
# ----------------------------------------------------------------------

def build_ranking(df, seeds):
    rows = []
    for seed_num, player in seeds.items():
        s = player_stats(df, player)
        s["score"] = composite_score(s)
        s["seed"] = seed_num
        s["player"] = player
        rows.append(s)

    table = pd.DataFrame(rows)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def print_table(table, title):
    print(f"\n{title}")
    print("-" * len(title))
    cols = ["rank", "seed", "player", "grass_pct", "recent_pct", "wimb_pct", "score"]
    fmt = table[cols].copy()
    for c in ["grass_pct", "recent_pct", "wimb_pct"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.0f}%")
    fmt = fmt.rename(columns={
        "rank": "Model Rank", "seed": "Seed", "player": "Player",
        "grass_pct": "Grass W%", "recent_pct": "Form (L25)",
        "wimb_pct": "Wimbledon W%", "score": "Score",
    })
    print(fmt.to_string(index=False))


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

def main():
    atp = load_matches(ATP_CSV)
    wta = load_matches(WTA_CSV)

    men_table = build_ranking(atp, MEN_SEEDS)
    women_table = build_ranking(wta, WOMEN_SEEDS)

    print_table(men_table, "WIMBLEDON 2026 - MEN'S TOP-8 SEED PREDICTION")
    print_table(women_table, "WIMBLEDON 2026 - WOMEN'S TOP-8 SEED PREDICTION")

    print("\nProjected finalists (model's top 2 by composite score):")
    print(f"  Men:   {men_table.iloc[0]['player']} vs {men_table.iloc[1]['player']}")
    print(f"  Women: {women_table.iloc[0]['player']} vs {women_table.iloc[1]['player']}")

    print("\nProjected champions (highest score):")
    print(f"  Men's champion:   {men_table.iloc[0]['player']}  (score {men_table.iloc[0]['score']})")
    print(f"  Women's champion: {women_table.iloc[0]['player']}  (score {women_table.iloc[0]['score']})")

    men_table.to_csv("wimbledon_2026_men_prediction.csv", index=False)
    women_table.to_csv("wimbledon_2026_women_prediction.csv", index=False)
    print("\nSaved: wimbledon_2026_men_prediction.csv, wimbledon_2026_women_prediction.csv")


if __name__ == "__main__":
    main()