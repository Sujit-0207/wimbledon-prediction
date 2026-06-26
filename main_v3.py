"""
Wimbledon 2026 — Trained ML Model + Monte Carlo Bracket Simulation
=====================================================================
Instead of a hand-tuned Elo formula, this trains an actual binary
classifier (Gradient Boosting) on historical match data to predict
P(player_1 wins). Features: Elo diff, grass-Elo diff, rank diff,
points diff, recent-form diff, and surface.

The trained model is then used as the win-probability function inside
a Monte Carlo simulation of the real Wimbledon 32-seed bracket.

Usage:
    python wimbledon_2026_trained_model.py

Requires: atp_tennis.csv, wta.csv, scikit-learn, pandas, numpy
"""

import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

ATP_CSV = "atp_tennis.csv"
WTA_CSV = "wta.csv"

N_SIMULATIONS = 20000
K_OVERALL = 32
K_GRASS = 24
BASE_ELO = 1500
RECENT_N = 25
RANDOM_STATE = 42

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
# 2. LOAD + CLEAN
# ----------------------------------------------------------------------

def load_matches(path):
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Player_1"] = df["Player_1"].str.strip()
    df["Player_2"] = df["Player_2"].str.strip()
    df["Winner"] = df["Winner"].astype(str).str.strip()
    df["Rank_1"] = pd.to_numeric(df["Rank_1"], errors="coerce")
    df["Rank_2"] = pd.to_numeric(df["Rank_2"], errors="coerce")
    df["Pts_1"] = pd.to_numeric(df.get("Pts_1"), errors="coerce")
    df["Pts_2"] = pd.to_numeric(df.get("Pts_2"), errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING — replay history chronologically, building
#    Elo + recent-form state per player, and emitting one training row
#    per match using only information available BEFORE that match.
# ----------------------------------------------------------------------

def build_feature_table(df):
    elo, elo_grass = {}, {}
    last_results = defaultdict(list)  # rolling list of 1/0 results

    rows = []
    cols = ["Player_1", "Player_2", "Winner", "Surface", "Rank_1", "Rank_2", "Pts_1", "Pts_2"]
    for p1, p2, w, surf, rank1, rank2, pts1, pts2 in df[cols].itertuples(index=False, name=None):

        e1, e2 = elo.get(p1, BASE_ELO), elo.get(p2, BASE_ELO)
        g1, g2 = elo_grass.get(p1, BASE_ELO), elo_grass.get(p2, BASE_ELO)
        form1 = np.mean(last_results[p1][-RECENT_N:]) if last_results[p1] else 0.5
        form2 = np.mean(last_results[p2][-RECENT_N:]) if last_results[p2] else 0.5

        is_grass = 1 if surf == "Grass" else 0

        rows.append({
            "elo_diff": e1 - e2,
            "grass_elo_diff": g1 - g2,
            "rank_diff": (rank2 - rank1) if pd.notna(rank1) and pd.notna(rank2) else 0,
            "pts_diff": (pts1 - pts2) if pd.notna(pts1) and pd.notna(pts2) else 0,
            "form_diff": form1 - form2,
            "is_grass": is_grass,
            "label": 1 if w == p1 else 0,
        })

        # ---- update state AFTER recording features (no leakage) ----
        s1 = 1.0 if w == p1 else 0.0
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        elo[p1] = e1 + K_OVERALL * (s1 - exp1)
        elo[p2] = e2 + K_OVERALL * ((1 - s1) - (1 - exp1))
        if is_grass:
            expg1 = 1 / (1 + 10 ** ((g2 - g1) / 400))
            elo_grass[p1] = g1 + K_GRASS * (s1 - expg1)
            elo_grass[p2] = g2 + K_GRASS * ((1 - s1) - (1 - expg1))
        last_results[p1].append(s1)
        last_results[p2].append(1 - s1)

    feat_df = pd.DataFrame(rows)
    return feat_df, elo, elo_grass, last_results


def augment_symmetric(feat_df):
    """Double the dataset by swapping player1/player2 perspective,
    so the model doesn't learn a spurious 'player_1 bias'."""
    swapped = feat_df.copy()
    for col in ["elo_diff", "grass_elo_diff", "rank_diff", "pts_diff", "form_diff"]:
        swapped[col] = -swapped[col]
    swapped["label"] = 1 - swapped["label"]
    return pd.concat([feat_df, swapped], ignore_index=True)


# ----------------------------------------------------------------------
# 4. TRAIN MODEL
# ----------------------------------------------------------------------

FEATURES = ["elo_diff", "grass_elo_diff", "rank_diff", "pts_diff", "form_diff", "is_grass"]


def train_model(feat_df, label):
    data = augment_symmetric(feat_df)
    X, y = data[FEATURES], data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    print(f"\n[{label}] Model trained on {len(X_train):,} rows, tested on {len(X_test):,} rows")
    print(f"  Test accuracy : {acc:.3f}")
    print(f"  Test log-loss : {ll:.3f}")
    print(f"  Test ROC-AUC  : {auc:.3f}")
    corrs = X_train.assign(label=y_train.values).corr()["label"].drop("label").sort_values(key=abs, ascending=False)
    print("  Feature correlation with label:", ", ".join(f"{f}={v:.2f}" for f, v in corrs.items()))

    return model


# ----------------------------------------------------------------------
# 5. BRACKET SIMULATION USING TRAINED MODEL
# ----------------------------------------------------------------------

def seeding_bracket_order(n=32):
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


def player_feature_vector(p, elo, elo_grass, last_results):
    """Current-state feature snapshot for a player, used at prediction time."""
    return {
        "elo": elo.get(p, BASE_ELO),
        "grass_elo": elo_grass.get(p, BASE_ELO),
        "form": np.mean(last_results[p][-RECENT_N:]) if last_results.get(p) else 0.5,
    }


def precompute_pairwise_probs(model, seed_list, snap, is_grass=1):
    """Compute P(a beats b) for every pair once, vectorized, to avoid
    calling model.predict_proba thousands of times inside the simulation loop."""
    pairs = [(a, b) for a in seed_list for b in seed_list if a != b]
    rows = []
    for a, b in pairs:
        sa, sb = snap[a], snap[b]
        rows.append({
            "elo_diff": sa["elo"] - sb["elo"],
            "grass_elo_diff": sa["grass_elo"] - sb["grass_elo"],
            "rank_diff": 0,
            "pts_diff": 0,
            "form_diff": sa["form"] - sb["form"],
            "is_grass": is_grass,
        })
    X = pd.DataFrame(rows)[FEATURES]
    probs = model.predict_proba(X)[:, 1]
    lookup = {}
    for (a, b), p in zip(pairs, probs):
        lookup[(a, b)] = p
    return lookup


def match_win_prob_lookup(lookup, pa, pb):
    return lookup[(pa, pb)]


def simulate_bracket(players_in_order, lookup):
    rounds_won = {p: 0 for p in players_in_order}
    current_round = players_in_order[:]
    rnd = 1
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            a, b = current_round[i], current_round[i + 1]
            p_a = match_win_prob_lookup(lookup, a, b)
            winner = a if random.random() < p_a else b
            rounds_won[winner] = rnd
            next_round.append(winner)
        current_round = next_round
        rnd += 1
    return rounds_won


def monte_carlo(seed_list, model, snap, n_sims=N_SIMULATIONS):
    n = len(seed_list)
    total_rounds = n.bit_length() - 1
    order = seeding_bracket_order(n)
    players_in_order = [seed_list[s - 1] for s in order]
    lookup = precompute_pairwise_probs(model, seed_list, snap)

    semifinal_threshold = total_rounds - 2
    final_threshold = total_rounds - 1

    reach_counts = defaultdict(lambda: defaultdict(int))
    title_counts = defaultdict(int)

    for _ in range(n_sims):
        result = simulate_bracket(players_in_order, lookup)
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
            "elo": round(snap[p]["elo"], 1),
            "grass_elo": round(snap[p]["grass_elo"], 1),
            "semifinal_prob_%": round(sf_plus, 1),
            "final_prob_%": round(final_plus, 1),
            "title_prob_%": round(title_pct, 1),
        })

    table = pd.DataFrame(rows).sort_values("title_prob_%", ascending=False).reset_index(drop=True)
    table.insert(0, "seed", [seed_list.index(p) + 1 for p in table["player"]])
    return table


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------

def run_draw(csv_path, seed_list, label):
    df = load_matches(csv_path)
    feat_df, elo, elo_grass, last_results = build_feature_table(df)
    model = train_model(feat_df, label)

    missing = [p for p in seed_list if p not in elo]
    if missing:
        print(f"  [warning] no match history for: {missing} (defaulted to base Elo {BASE_ELO})")

    snap = {p: player_feature_vector(p, elo, elo_grass, last_results) for p in seed_list}
    table = monte_carlo(seed_list, model, snap)

    print(f"\nWIMBLEDON 2026 — {label} 32-SEED PREDICTION (trained ML model, {N_SIMULATIONS:,} sims)")
    print("-" * 75)
    print(table.to_string(index=False))
    print(f"\nModel's pick to win {label.lower()}: "
          f"{table.iloc[0]['player']} ({table.iloc[0]['title_prob_%']}% of simulations)")
    return table


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    men_table = run_draw(ATP_CSV, MEN_SEEDS, "MEN'S")
    women_table = run_draw(WTA_CSV, WOMEN_SEEDS, "WOMEN'S")

    men_table.to_csv("wimbledon_2026_men_ml_prediction.csv", index=False)
    women_table.to_csv("wimbledon_2026_women_ml_prediction.csv", index=False)
    print("\nSaved: wimbledon_2026_men_ml_prediction.csv, wimbledon_2026_women_ml_prediction.csv")


if __name__ == "__main__":
    main()
