"""
Wimbledon 2026 — Elo-Based Monte Carlo Tournament Simulator
==============================================================
Builds a historical Elo rating (overall + grass-specific) for every
player from ATP/WTA match data, then simulates the Wimbledon 2026
32-seed bracket thousands of times to estimate each seed's probability
of reaching each round, and of winning the title.

Usage:
    python wimbledon_2026_elo_simulation.py

Requires atp_tennis.csv and wta.csv (same folder, or edit paths below).
Columns expected: Tournament, Date, Surface, Player_1, Player_2, Winner.
"""

import random
import pandas as pd
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

ATP_CSV = "atp_tennis.csv"
WTA_CSV = "wta.csv"

N_SIMULATIONS = 20000     # Monte Carlo trials per draw
K_OVERALL = 32            # Elo K-factor, all surfaces
K_GRASS = 24              # Elo K-factor, grass-only ratings
BASE_ELO = 1500
GRASS_WEIGHT = 0.55       # blend weight on grass-specific Elo vs overall Elo

# Official Wimbledon 2026 seeds (seeding cut-off: rankings as of 22 June 2026)
# Seed order also defines bracket position (seed 1 vs seed 32, etc. - the
# standard single-elimination "1-32" seeding pattern used by Grand Slams).

MEN_SEEDS = [
    "Sinner J.", "Zverev A.", "Auger-Aliassime F.", "Shelton B.",
    "De Minaur A.", "Fritz T.", "Djokovic N.", "Medvedev D.",
    "Cobolli F.", "Bublik A.", "Ruud C.", "Rublev A.",
    "Lehecka J.", "Darderi L.", "Mensik J.", "Tien L.",
    "Tiafoe F.", "Cerundolo F.", "Khachanov K.", "Fils A.",
    "Paul T.", "Davidovich Fokina A.", "Jodar R.", "Fonseca J.",
    "Rinderknech A.", "Norrie C.", "Humbert U.", "Nakashima B.",
    "Etcheverry T.", "Tabilo A.", "Buse I.", "Arnaldi M.",
]

WOMEN_SEEDS = [
    "Sabalenka A.", "Rybakina E.", "Swiatek I.", "Pegula J.",
    "Andreeva M.", "Anisimova A.", "Gauff C.", "Svitolina E.",
    "Noskova L.", "Muchova K.", "Bencic B.", "Kostyuk M.",
    "Paolini J.", "Osaka N.", "Shnaider D.", "Jovic I.",
    "Cirstea S.", "Alexandrova E.", "Kalinskaya A.", "Chwalinska M.",
    "Bouzkova M.", "Fernandez L.A.", "Navarro E.", "Tauson C.",
    "Mertens E.", "Keys M.", "Potapova A.", "Li A.",
    "Eala A.", "Raducanu E.", "Vekic D.", "Siniakova K.",
]


# ----------------------------------------------------------------------
# 2. ELO RATING BUILDER
# ----------------------------------------------------------------------

def build_elo(df, k=K_OVERALL, surface_k=K_GRASS, base=BASE_ELO):
    """Chronologically replay every match to build overall + grass Elo."""
    df = df.dropna(subset=["Date"]).sort_values("Date")
    elo, elo_grass = {}, {}

    for _, r in df.iterrows():
        p1, p2, w, surf = r["Player_1"], r["Player_2"], r["Winner"], r["Surface"]
        e1, e2 = elo.get(p1, base), elo.get(p2, base)
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        s1 = 1.0 if w == p1 else 0.0
        elo[p1] = e1 + k * (s1 - exp1)
        elo[p2] = e2 + k * ((1 - s1) - (1 - exp1))

        if surf == "Grass":
            g1, g2 = elo_grass.get(p1, base), elo_grass.get(p2, base)
            expg1 = 1 / (1 + 10 ** ((g2 - g1) / 400))
            elo_grass[p1] = g1 + surface_k * (s1 - expg1)
            elo_grass[p2] = g2 + surface_k * ((1 - s1) - (1 - expg1))

    return elo, elo_grass


def blended_rating(player, elo, elo_grass, grass_weight=GRASS_WEIGHT, base=BASE_ELO):
    overall = elo.get(player, base)
    grass = elo_grass.get(player, base)
    return grass_weight * grass + (1 - grass_weight) * overall


# ----------------------------------------------------------------------
# 3. BRACKET CONSTRUCTION (standard 1-32 Grand Slam seeding order)
# ----------------------------------------------------------------------

def seeding_bracket_order(n=32):
    """
    Returns the standard single-elimination seeding order, e.g. for 8:
    [1,8,4,5,2,7,3,6] -> 1v8, 4v5, 2v7, 3v6 in round 1.
    Generalizes recursively for any power of two.
    """
    order = [1]
    size = 1
    while size < n:
        new_order = []
        for s in order:
            new_order.append(s)
            new_order.append(size * 2 + 1 - s)
        order = new_order
        size *= 2
    return order


# ----------------------------------------------------------------------
# 4. SINGLE-ELIMINATION SIMULATION
# ----------------------------------------------------------------------

def win_prob(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def simulate_bracket(players_in_order, ratings):
    """
    Run a single simulated tournament.
    Returns dict: player -> number of rounds WON (0 = lost in round 1,
    total_rounds = champion).
    """
    rounds_won = {p: 0 for p in players_in_order}
    current_round = players_in_order[:]
    rnd = 1
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            a, b = current_round[i], current_round[i + 1]
            p_a = win_prob(ratings[a], ratings[b])
            winner = a if random.random() < p_a else b
            rounds_won[winner] = rnd
            next_round.append(winner)
        current_round = next_round
        rnd += 1
    return rounds_won


def monte_carlo(seed_list, ratings, n_sims=N_SIMULATIONS):
    n = len(seed_list)
    total_rounds = n.bit_length() - 1  # log2(n) for a power of two
    order = seeding_bracket_order(n)
    players_in_order = [seed_list[s - 1] for s in order]

    # thresholds, in "rounds won", for each milestone
    semifinal_threshold = total_rounds - 2  # won quarterfinal -> reached SF
    final_threshold = total_rounds - 1      # won semifinal -> reached final
    title_threshold = total_rounds          # won final -> champion

    reach_counts = defaultdict(lambda: defaultdict(int))
    title_counts = defaultdict(int)

    for _ in range(n_sims):
        result = simulate_bracket(players_in_order, ratings)
        champ = max(result, key=lambda p: result[p])
        title_counts[champ] += 1
        for p, rnd in result.items():
            reach_counts[p][rnd] += 1

    rows = []
    for p in seed_list:
        sf_plus = sum(v for r, v in reach_counts[p].items() if r >= semifinal_threshold) / n_sims * 100
        final_plus = sum(v for r, v in reach_counts[p].items() if r >= final_threshold) / n_sims * 100
        title_pct = title_counts[p] / n_sims * 100
        rows.append({
            "player": p,
            "rating": round(ratings[p], 1),
            "semifinal_prob_%": round(sf_plus, 1),
            "final_prob_%": round(final_plus, 1),
            "title_prob_%": round(title_pct, 1),
        })

    table = pd.DataFrame(rows).sort_values("title_prob_%", ascending=False).reset_index(drop=True)
    table.insert(0, "seed", [seed_list.index(p) + 1 for p in table["player"]])
    return table


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

def run_draw(csv_path, seed_list, label):
    df = pd.read_csv(csv_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Player_1"] = df["Player_1"].str.strip()
    df["Player_2"] = df["Player_2"].str.strip()
    df["Winner"] = df["Winner"].astype(str).str.strip()

    elo, elo_grass = build_elo(df)
    ratings = {p: blended_rating(p, elo, elo_grass) for p in seed_list}

    missing = [p for p in seed_list if p not in elo]
    if missing:
        print(f"  [warning] no match history found for: {missing} (defaulted to {BASE_ELO} Elo)")

    table = monte_carlo(seed_list, ratings)

    print(f"\nWIMBLEDON 2026 — {label} 32-SEED ELO SIMULATION  ({N_SIMULATIONS:,} trials)")
    print("-" * 70)
    print(table.to_string(index=False))

    print(f"\nTop pick to win {label.lower()}: "
          f"{table.iloc[0]['player']} ({table.iloc[0]['title_prob_%']}% of simulations)")
    return table


def main():
    random.seed(42)  # reproducible simulation results

    men_table = run_draw(ATP_CSV, MEN_SEEDS, "MEN'S")
    women_table = run_draw(WTA_CSV, WOMEN_SEEDS, "WOMEN'S")

    men_table.to_csv("wimbledon_2026_men_elo_simulation.csv", index=False)
    women_table.to_csv("wimbledon_2026_women_elo_simulation.csv", index=False)
    print("\nSaved: wimbledon_2026_men_elo_simulation.csv, wimbledon_2026_women_elo_simulation.csv")


if __name__ == "__main__":
    main()
