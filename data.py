"""Schedule ingestion and feature generation without future-game leakage."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED = {"season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"}


def load_games(path: str) -> pd.DataFrame:
    """Load nflverse schedules CSV or a compatible local CSV.

    Historical results must have both scores. Upcoming games can have blank scores.
    """
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    for col in ("season", "week", "home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["gameday", "season", "week", "home_team", "away_team"]).reset_index(drop=True)
    return df


def make_features(games: pd.DataFrame, prior_points: float = 22.0) -> pd.DataFrame:
    """Pre-game rolling team features computed using completed *earlier dates* only.

    Games on the same date are evaluated against the same pre-date state.
    This intentionally excludes injury/market information not available in input.
    """
    state: dict[str, dict[str, float]] = {}
    rows = []
    def get(team: str) -> dict[str, float]:
        return state.get(team, {"n": 0.0, "for": 0.0, "against": 0.0, "wins": 0.0})
    for _, batch in games.groupby("gameday", sort=True):
        updates = []
        for _, game in batch.iterrows():
            h, a = str(game.home_team), str(game.away_team)
            hs, as_ = get(h), get(a)
            def avg(s, key):
                return (s[key] + 5 * prior_points) / (s["n"] + 5) if key != "wins" else (s["wins"] + 2.5) / (s["n"] + 5)
            row = game.to_dict()
            row.update({
                "home_pf": avg(hs, "for"), "home_pa": avg(hs, "against"),
                "away_pf": avg(as_, "for"), "away_pa": avg(as_, "against"),
                "home_win_rate": avg(hs, "wins"), "away_win_rate": avg(as_, "wins"),
                "home_games": hs["n"], "away_games": as_["n"],
            })
            rows.append(row)
            if pd.notna(game.home_score) and pd.notna(game.away_score):
                updates.append((h, a, float(game.home_score), float(game.away_score)))
        for h, a, hp, ap in updates:
            for team, pf, pa, win in ((h, hp, ap, float(hp > ap) + 0.5 * float(hp == ap)),
                                      (a, ap, hp, float(ap > hp) + 0.5 * float(ap == hp))):
                s = state.setdefault(team, {"n": 0.0, "for": 0.0, "against": 0.0, "wins": 0.0})
                s["n"] += 1
                s["for"] += pf
                s["against"] += pa
                s["wins"] += win
    return pd.DataFrame(rows)

FEATURES = ["home_pf", "home_pa", "away_pf", "away_pa", "home_win_rate", "away_win_rate", "home_games", "away_games"]
