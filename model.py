"""Ridge score model with chronological backtesting and Monte Carlo probabilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES


def completed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.home_score.notna() & df.away_score.notna()].copy()


class ScoreModel:
    def __init__(self, alpha: float = 50.0):
        self.home = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.away = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.residuals: np.ndarray | None = None

    def fit(self, features: pd.DataFrame) -> "ScoreModel":
        train = completed(features)
        if len(train) < 32:
            raise ValueError("At least 32 completed historical games are required for training")
        x = train[FEATURES].to_numpy(dtype=float)
        self.home.fit(x, train.home_score.to_numpy(dtype=float))
        self.away.fit(x, train.away_score.to_numpy(dtype=float))
        pred = np.column_stack([self.home.predict(x), self.away.predict(x)])
        self.residuals = train[["home_score", "away_score"]].to_numpy(dtype=float) - pred
        return self

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if self.residuals is None:
            raise ValueError("Fit the model before predicting")
        x = features[FEATURES].to_numpy(dtype=float)
        return np.clip(np.column_stack([self.home.predict(x), self.away.predict(x)]), 0, None)

    def simulate(self, features: pd.DataFrame, n: int = 20000, seed: int = 42,
                 spread: float | None = None, total: float | None = None) -> dict:
        if len(features) != 1:
            raise ValueError("simulate expects exactly one game")
        if n < 100:
            raise ValueError("n must be at least 100")
        rng = np.random.default_rng(seed)
        means = self.predict_scores(features)[0]
        assert self.residuals is not None
        sampled = self.residuals[rng.integers(0, len(self.residuals), n)]
        scores = np.maximum(0, np.rint(means + sampled)).astype(int)
        margin = scores[:, 0] - scores[:, 1]
        totals = scores.sum(axis=1)
        result = {
            "home_team": str(features.iloc[0].home_team),
            "away_team": str(features.iloc[0].away_team),
            "expected_home_points": round(float(scores[:, 0].mean()), 2),
            "expected_away_points": round(float(scores[:, 1].mean()), 2),
            "home_win_probability": round(float(np.mean(margin > 0)), 4),
            "away_win_probability": round(float(np.mean(margin < 0)), 4),
            "tie_probability": round(float(np.mean(margin == 0)), 4),
            "simulations": n,
            "note": "Empirical in-sample residual bootstrap; uncertainty may be miscalibrated.",
        }
        if spread is not None:
            result["home_cover_probability"] = round(float(np.mean(margin + spread > 0)), 4)
            result["spread_push_probability"] = round(float(np.mean(margin + spread == 0)), 4)
        if total is not None:
            result["over_probability"] = round(float(np.mean(totals > total)), 4)
            result["total_push_probability"] = round(float(np.mean(totals == total)), 4)
        return result


def walk_forward(features: pd.DataFrame, min_train: int = 128, step: int = 16) -> pd.DataFrame:
    """Fit only on games strictly before each test date; no lookahead.

    Predictions are generated for all games in the next `step` distinct dates.
    """
    games = completed(features).sort_values("gameday").reset_index(drop=True)
    dates = sorted(games.gameday.unique())
    out = []
    start = 0
    while start < len(dates):
        cutoff = dates[start]
        train = games.loc[games.gameday < cutoff]
        if len(train) < min_train:
            start += 1
            continue
        test_dates = dates[start:start + step]
        test = games.loc[games.gameday.isin(test_dates)].copy()
        model = ScoreModel().fit(train)
        pred = model.predict_scores(test)
        test["pred_home"] = pred[:, 0]
        test["pred_away"] = pred[:, 1]
        test["pred_margin"] = pred[:, 0] - pred[:, 1]
        test["actual_margin"] = test.home_score - test.away_score
        test["abs_margin_error"] = (test.pred_margin - test.actual_margin).abs()
        out.append(test[["gameday", "home_team", "away_team", "pred_home", "pred_away", "pred_margin", "actual_margin", "abs_margin_error"]])
        start += step
    if not out:
        raise ValueError("Insufficient historical data for walk-forward evaluation")
    result = pd.concat(out, ignore_index=True)
    return result.drop_duplicates(["gameday", "home_team", "away_team"])


def american_to_decimal(odds: int) -> float:
    if odds == 0 or -100 < odds < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def expected_value(probability: float, american_odds: int, push_probability: float = 0) -> float:
    """Expected profit per $1 staked, including pushes (returned stake)."""
    if not (0 <= probability <= 1 and 0 <= push_probability <= 1 and probability + push_probability <= 1):
        raise ValueError("Invalid win/push probabilities")
    return probability * (american_to_decimal(american_odds) - 1) - (1 - probability - push_probability)
