"""Local Streamlit dashboard. Does not send your data to a model provider."""
import streamlit as st
from nfl_engine.data import load_games, make_features
from nfl_engine.model import ScoreModel, completed, walk_forward, expected_value

st.set_page_config(page_title="NFL Prediction Engine", layout="wide")
st.title("NFL Prediction Engine")
st.caption("Research prototype • local schedule data • no profitability claims")
upload = st.file_uploader("Upload nflverse schedules CSV", type="csv")
if upload is None:
    st.info("Upload a schedules CSV containing season, week, gameday, home_team, away_team, home_score, away_score.")
    st.stop()
try:
    data = make_features(load_games(upload))
except Exception as exc:
    st.error(str(exc))
    st.stop()
mode = st.radio("Analysis", ["Upcoming game forecast", "Walk-forward backtest"], horizontal=True)
if mode == "Walk-forward backtest":
    min_train = st.number_input("Minimum training games", min_value=32, value=128)
    try:
        backtest = walk_forward(data, min_train=int(min_train))
        st.metric("Out-of-sample margin MAE", f"{backtest.abs_margin_error.mean():.2f} points")
        st.metric("Evaluated games", len(backtest))
        st.dataframe(backtest, use_container_width=True)
    except ValueError as exc:
        st.warning(str(exc))
else:
    pending = data.loc[data.home_score.isna() | data.away_score.isna()].copy()
    if pending.empty:
        st.warning("No upcoming games found. Add schedule rows with blank scores.")
        st.stop()
    pending["label"] = pending.gameday.dt.strftime("%Y-%m-%d") + " | " + pending.away_team + " @ " + pending.home_team
    selection = st.selectbox("Game", pending.index.tolist(), format_func=lambda i: pending.loc[i, "label"])
    c1, c2, c3 = st.columns(3)
    spread = c1.number_input("Home spread", value=-3.5, step=0.5)
    total = c2.number_input("Game total", value=44.5, step=0.5)
    odds = c3.number_input("American odds for home spread", value=-110, step=5)
    n = st.slider("Monte Carlo simulations", 1000, 100000, 20000, step=1000)
    game = data.loc[[selection]]
    train = completed(data.loc[data.gameday < game.iloc[0].gameday])
    try:
        forecast = ScoreModel().fit(train).simulate(game, n=n, spread=spread, total=total)
        st.json(forecast)
        p = forecast["home_cover_probability"]
        push = forecast["spread_push_probability"]
        st.metric("Home spread EV per $1 (model-estimated)", f"${expected_value(p, int(odds), push):+.3f}")
        st.caption("These probabilities are uncalibrated; an apparent edge is not proof of an advantage.")
    except ValueError as exc:
        st.warning(str(exc))
