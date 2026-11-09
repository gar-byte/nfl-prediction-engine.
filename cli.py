"""Command-line entry point."""
import argparse
import json
from .data import load_games, make_features
from .model import ScoreModel, completed, walk_forward


def main():
    p = argparse.ArgumentParser(description="NFL schedule-based prediction research engine")
    p.add_argument("--csv", required=True, help="Path to historical/upcoming schedule CSV")
    p.add_argument("--mode", choices=["forecast", "backtest"], default="forecast")
    p.add_argument("--home", help="Upcoming home team abbreviation")
    p.add_argument("--away", help="Upcoming away team abbreviation")
    p.add_argument("--spread", type=float, help="Home team spread (e.g. -3.5)")
    p.add_argument("--total", type=float, help="Game total points line")
    p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train", type=int, default=128)
    args = p.parse_args()
    features = make_features(load_games(args.csv))
    if args.mode == "backtest":
        results = walk_forward(features, min_train=args.min_train)
        print(f"Test games: {len(results)} | Margin MAE: {results.abs_margin_error.mean():.2f}")
        print(results.tail(10).to_string(index=False))
        return
    if not args.home or not args.away:
        p.error("--home and --away required for forecast")
    pending = features.loc[features.home_team.eq(args.home) & features.away_team.eq(args.away) &
                           (features.home_score.isna() | features.away_score.isna())]
    if pending.empty:
        p.error("No unplayed matching game in CSV; add an upcoming schedule row with blank scores")
    game = pending.sort_values("gameday").iloc[[0]]
    training = completed(features.loc[features.gameday < game.iloc[0].gameday])
    model = ScoreModel().fit(training)
    print(json.dumps(model.simulate(game, n=args.sims, seed=args.seed, spread=args.spread, total=args.total), indent=2))

if __name__ == "__main__":
    main()
