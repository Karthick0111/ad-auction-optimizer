"""
Ad Auction Bid/CTR Optimization Simulator dashboard.

Two views:
  1. CTR Model Performance - reads the pretrained model + metrics bundled in
     the repo (model/ctr_model.txt, model/metrics.json, model/holdout.parquet).
     No AWS calls needed - this is about the static, already-trained model.
  2. Auction Simulation - the live piece: triggers a new run via the
     run_trigger Lambda (boto3, using narrowly-scoped read-only IAM
     credentials stored as Streamlit secrets) and polls DynamoDB for live
     results as Kinesis -> Lambda -> DynamoDB processes the run.

Run locally with: streamlit run dashboard/app.py
"""
import json
import sys
import time
from pathlib import Path

import boto3
import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard.theme import ARM_COLOR, apply_theme
from simulation.schema import CATEGORICAL_COLS, LABEL_COL, NUMERIC_COLS

st.set_page_config(page_title="Ad Auction Optimizer", layout="wide")
apply_theme()

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def _secret(key: str, default=None):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return default


@st.cache_resource
def _aws_session():
    return boto3.Session(
        aws_access_key_id=_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_secret("AWS_SECRET_ACCESS_KEY"),
        region_name=_secret("AWS_REGION", "us-east-1"),
    )


@st.cache_resource
def _load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_DIR / "ctr_model.txt"))


@st.cache_data
def _load_metrics() -> dict:
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def _load_category_mappings() -> dict:
    with open(MODEL_DIR / "category_mappings.json") as f:
        return json.load(f)


@st.cache_data
def _load_holdout_sample(n: int = 5000) -> pd.DataFrame:
    df = pd.read_parquet(MODEL_DIR / "holdout.parquet")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
    return df.sample(n=min(n, len(df)), random_state=42)


def ctr_model_view():
    st.header("CTR Model Performance")
    metrics = _load_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LightGBM AUC", f"{metrics['lightgbm']['auc']:.4f}")
    col2.metric("LightGBM LogLoss", f"{metrics['lightgbm']['logloss']:.4f}")
    col3.metric("Baseline AUC (logistic regression)", f"{metrics['logistic_regression_baseline']['auc']:.4f}")
    col4.metric("Baseline LogLoss", f"{metrics['logistic_regression_baseline']['logloss']:.4f}")
    st.caption(
        f"Trained on {metrics['n_train']:,} rows of real Criteo CTR data, evaluated on "
        f"{metrics['n_holdout']:,} holdout rows. Overall click rate: {metrics['click_rate']:.2%}."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("LightGBM vs. baseline")
        fig = go.Figure()
        fig.add_bar(
            x=["LightGBM", "Logistic Regression"],
            y=[metrics["lightgbm"]["auc"], metrics["logistic_regression_baseline"]["auc"]],
            text=[f"{metrics['lightgbm']['auc']:.3f}", f"{metrics['logistic_regression_baseline']['auc']:.3f}"],
            textposition="outside",
        )
        fig.update_layout(yaxis_title="AUC-ROC (holdout)", yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    model = _load_model()
    with col_b:
        st.subheader("Top 15 features by gain")
        importance = pd.DataFrame({
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).head(15)
        fig2 = go.Figure(go.Bar(x=importance["importance"], y=importance["feature"], orientation="h"))
        fig2.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Gain")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Calibration: predicted CTR vs. actual click rate")
    sample = _load_holdout_sample()
    mappings = _load_category_mappings()
    # Same explicit category->code encoding used at training time (and by
    # the bid_consumer Lambda) - passing a freshly-loaded category-dtype
    # DataFrame straight to model.predict() would use pandas' own
    # locally-assigned codes, which don't match what the model was trained
    # on (see simulation/schema.py's encode_categoricals docstring).
    encoded = sample.copy()
    for col in CATEGORICAL_COLS:
        encoded[col] = sample[col].map(mappings[col]).fillna(-1).astype(int)
    preds = model.predict(encoded[NUMERIC_COLS + CATEGORICAL_COLS])
    sample = sample.assign(predicted_ctr=preds)
    sample["bucket"] = pd.qcut(sample["predicted_ctr"], 10, duplicates="drop")
    calibration = (
        sample.groupby("bucket", observed=True)
        .agg(predicted=("predicted_ctr", "mean"), actual=(LABEL_COL, "mean"))
        .reset_index()
    )
    fig3 = go.Figure()
    fig3.add_scatter(x=calibration["predicted"], y=calibration["actual"], mode="lines+markers", name="Model")
    max_val = float(calibration["predicted"].max())
    fig3.add_scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Perfect calibration", line=dict(dash="dash"))
    fig3.update_layout(xaxis_title="Predicted CTR (bucket mean)", yaxis_title="Actual click rate")
    st.plotly_chart(fig3, use_container_width=True)


def _invoke_run_trigger(session, config: dict) -> str:
    client = session.client("lambda")
    response = client.invoke(
        FunctionName=_secret("RUN_TRIGGER_FUNCTION_NAME", "ad-auction-optimizer-run-trigger"),
        InvocationType="RequestResponse",
        Payload=json.dumps(config),
    )
    payload = json.loads(response["Payload"].read())
    return json.loads(payload["body"])["run_id"]


def auction_simulation_view():
    st.header("Auction Simulation")
    st.caption(
        "Real historical impressions (from the CTR model's holdout set) are replayed onto a "
        "Kinesis stream and processed live by a Lambda that scores CTR, picks a bid multiplier "
        "via Thompson Sampling, and settles a simulated second-price auction against N competitors."
    )

    session = _aws_session()

    with st.sidebar:
        st.subheader("Run configuration")
        budget = st.number_input("Budget ($)", min_value=10.0, value=500.0, step=50.0)
        value_per_click = st.number_input("Value per click ($)", min_value=0.01, value=2.0, step=0.1)
        n_competitors = st.slider("Number of competitors", 1, 20, 5)
        sigma = st.slider("Competitiveness (higher = pricier auctions)", 0.1, 1.5, 0.4, 0.1)
        n_impressions = st.slider("Impressions to simulate", 100, 5000, 2000, 100)
        seed = st.number_input("Random seed", value=42, step=1)
        run_clicked = st.button("Run simulation", type="primary")

    if run_clicked:
        with st.spinner("Starting run..."):
            run_id = _invoke_run_trigger(session, {
                "budget": budget,
                "value_per_click": value_per_click,
                "n_competitors": n_competitors,
                "mean_log_bid": 0.0,
                "sigma": sigma,
                "n_impressions": n_impressions,
                "seed": seed,
            })
        st.session_state["run_id"] = run_id

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Configure a run in the sidebar and click **Run simulation** to start.")
        return

    dynamodb = session.resource("dynamodb")
    runs_table = dynamodb.Table(_secret("DYNAMODB_RUNS_TABLE", "ad-auction-optimizer-simulation-runs"))
    events_table = dynamodb.Table(_secret("DYNAMODB_EVENTS_TABLE", "ad-auction-optimizer-simulation-events"))

    run = runs_table.get_item(Key={"run_id": run_id}).get("Item")
    if not run:
        st.warning("Run not found yet - it may still be initializing. Refreshing shortly...")
        time.sleep(2)
        st.rerun()
        return

    status = run.get("status", "running")
    rows_processed = int(run.get("rows_processed", 0))
    run_budget = float(run["config"]["budget"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", status)
    col2.metric("Impressions processed", f"{rows_processed:,} / {run['config']['n_impressions']:,}")
    col3.metric("Cumulative spend", f"${float(run.get('cumulative_spend', 0)):.2f} / ${run_budget:.2f}")
    win_rate = int(run.get("wins", 0)) / max(rows_processed, 1)
    col4.metric("Win rate", f"{win_rate:.1%}")

    response = events_table.query(
        KeyConditionExpression="run_id = :r",
        ExpressionAttributeValues={":r": run_id},
    )
    events = response.get("Items", [])

    if events:
        events_df = pd.DataFrame(events).sort_values("sequence")
        for col in ["price_paid", "reward", "arm_multiplier", "predicted_ctr", "bid"]:
            events_df[col] = events_df[col].astype(float)
        events_df["cumulative_spend"] = events_df["price_paid"].cumsum()
        events_df["cumulative_reward"] = events_df["reward"].cumsum()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Cumulative spend vs. budget")
            fig = go.Figure()
            fig.add_scatter(x=events_df["sequence"], y=events_df["cumulative_spend"], name="Spend", fill="tozeroy")
            fig.add_hline(y=run_budget, line_dash="dash", annotation_text="Budget")
            fig.update_layout(xaxis_title="Impression #", yaxis_title="Cumulative spend ($)")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Cumulative reward")
            fig2 = go.Figure()
            fig2.add_scatter(x=events_df["sequence"], y=events_df["cumulative_reward"], name="Reward")
            fig2.update_layout(xaxis_title="Impression #", yaxis_title="Cumulative net reward ($)")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Bandit arm selection over time")
        arm_counts = events_df.groupby("arm_multiplier").size().reset_index(name="count")
        arm_counts["arm_multiplier"] = arm_counts["arm_multiplier"].astype(float)
        fig3 = go.Figure(go.Bar(
            x=[f"{m}x" for m in arm_counts["arm_multiplier"]],
            y=arm_counts["count"],
            marker_color=[ARM_COLOR.get(m, "#888") for m in arm_counts["arm_multiplier"]],
        ))
        fig3.update_layout(xaxis_title="Bid multiplier", yaxis_title="Times selected")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Recent impressions")
        st.dataframe(
            events_df[["sequence", "arm_multiplier", "bid", "won", "price_paid", "clicked", "reward"]]
            .sort_values("sequence", ascending=False)
            .head(20),
            use_container_width=True,
        )

    if status == "running":
        st.caption("Live - refreshing automatically while the simulation runs.")
        time.sleep(2)
        st.rerun()


def main():
    st.title("Ad Auction Bid/CTR Optimization Simulator")
    st.caption(
        "Real Criteo CTR data feeding a simulated real-time second-price auction, streamed "
        "through Kinesis → Lambda → DynamoDB on AWS. A Thompson Sampling bandit learns "
        "the best bid multiplier under a budget constraint. See the GitHub repo for the "
        "architecture, Terraform, and Lambda code."
    )
    view = st.radio(
        "View", ["CTR Model Performance", "Auction Simulation"],
        horizontal=True, label_visibility="collapsed",
    )
    st.divider()
    if view == "CTR Model Performance":
        ctr_model_view()
    else:
        auction_simulation_view()


if __name__ == "__main__":
    main()
