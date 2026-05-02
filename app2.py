import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_IMPORT_ERROR = ""
except Exception as exc:
    AgentExecutor = None
    create_tool_calling_agent = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    ChatGroq = None
    tool = None
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = str(exc)

LANGCHAIN_RUNTIME_ERROR = ""

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "model.h5"
UPDATE_CSV_PATH = BASE_DIR / "logs" / "update_history.csv"
PARAMETERS_CSV_PATH = BASE_DIR / "logs" / "parameters.csv"

ACTIONABLE_FEATURES = [
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
LEARNING_FEATURES = ACTIONABLE_FEATURES
OFFER_LIBRARY = {
    "cashback_bonus": "cashback bonus on card and debit spends",
    "fee_waiver": "temporary waiver on service and transfer fees",
    "rate_bonus": "higher savings return for a limited retention window",
    "credit_limit_review": "priority credit review with card upgrade support",
    "product_bundle": "bundled second product with loyalty benefits",
    "advisor_callback": "dedicated advisor callback with tailored account guidance", #-? Should be done if credit score is low
}
OFFER_FEATURES = list(OFFER_LIBRARY.keys())
OFFER_EFFECTS = { #-? Dense heuristic priors so every offer can influence every actionable feature from the start
    "cashback_bonus": {"Balance": 0.45, "NumOfProducts": 0.15, "HasCrCard": 0.10, "IsActiveMember": 0.30},
    "fee_waiver": {"Balance": 0.35, "NumOfProducts": 0.15, "HasCrCard": 0.20, "IsActiveMember": 0.30},
    "rate_bonus": {"Balance": 0.50, "NumOfProducts": 0.10, "HasCrCard": 0.10, "IsActiveMember": 0.30},
    "credit_limit_review": {"Balance": 0.10, "NumOfProducts": 0.10, "HasCrCard": 0.55, "IsActiveMember": 0.25},
    "product_bundle": {"Balance": 0.10, "NumOfProducts": 0.50, "HasCrCard": 0.15, "IsActiveMember": 0.25},
    "advisor_callback": {"Balance": 0.20, "NumOfProducts": 0.15, "HasCrCard": 0.15, "IsActiveMember": 0.50},
}
RNG = np.random.default_rng()
BATCH_SIZE = 5

st.set_page_config(page_title="Bank Customer Retention Agent", layout="wide")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(BASE_DIR / ".env")


@st.cache_resource
def load_assets() -> tuple[Any, Any, Any, Any]:
    model = tf.keras.models.load_model(MODEL_PATH)
    with (BASE_DIR / "label_encoder_gender.pkl").open("rb") as file:
        label_encoder_gender = pickle.load(file)
    with (BASE_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
        onehot_encoder_geo = pickle.load(file)
    with (BASE_DIR / "Scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler


@st.cache_data
def load_customer_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["CustomerId"] = df["CustomerId"].astype(str)
    return df


model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
customer_df = load_customer_dataset()


@st.cache_data
def build_learning_feature_dataset() -> tuple[pd.DataFrame, pd.Series]:
    base_features = customer_df[LEARNING_FEATURES].astype(float).copy()
    target = customer_df["Exited"].astype(int).copy()
    return base_features, target


def scale_learning_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    scaler_mean = pd.Series(getattr(scaler, "mean_", []), index=scaler_features, dtype=float)
    scaler_scale = pd.Series(getattr(scaler, "scale_", []), index=scaler_features, dtype=float)
    learning_mean = scaler_mean.reindex(LEARNING_FEATURES)
    learning_scale = scaler_scale.reindex(LEARNING_FEATURES).replace(0.0, 1.0)
    return (feature_frame - learning_mean) / learning_scale


def customer_to_learning_vector(customer_data: dict[str, Any]) -> pd.Series:
    return pd.Series(
        {feature: float(customer_data[feature]) for feature in LEARNING_FEATURES},
        dtype=float,
    )


def serialize_series(series: pd.Series) -> dict[str, float]:
    return {key: float(value) for key, value in series.items()}


#-? Rounded conversion is still required here because these fields are binary/count inputs to the churn model
def predict_churn_from_learning_vector(customer_data: dict[str, Any], learning_vector: pd.Series) -> float:
    updated_customer = dict(customer_data)
    for feature, value in learning_vector.items():
        if feature in {"NumOfProducts", "HasCrCard", "IsActiveMember"}:
            updated_customer[feature] = int(round(float(value)))
        else:
            updated_customer[feature] = float(value)
    return predict_churn(updated_customer)


def initialize_learning_state() -> dict[str, Any]:
    feature_frame, target = build_learning_feature_dataset()
    scaled_feature_frame = scale_learning_features(feature_frame)
    retained_mean = feature_frame.loc[target == 0].mean()
    retained_mean_scaled = scaled_feature_frame.loc[target == 0].mean()
    churned_mean_scaled = scaled_feature_frame.loc[target == 1].mean()
    feature_weights = (retained_mean_scaled - churned_mean_scaled).fillna(0.0)
    offer_weights = {
        feature: {
            offer_name: float(OFFER_EFFECTS.get(offer_name, {}).get(feature, 0.0))
            for offer_name in OFFER_LIBRARY
        }
        for feature in LEARNING_FEATURES
    }
    return {
        "retained_mean": serialize_series(retained_mean),
        "feature_weights": serialize_series(feature_weights),
        "offer_weights": offer_weights,
    }


def append_parameters_snapshot(
    state: dict[str, Any],
    reason: str,
    customer_id: str = "",
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "customer_id": customer_id,
        "feature_weights": json.dumps(state["feature_weights"]),
        "offer_weights": json.dumps(state["offer_weights"]),
        "retained_mean": json.dumps(state["retained_mean"]),
    }
    PARAMETERS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot_frame = pd.DataFrame([payload])
    if PARAMETERS_CSV_PATH.exists():
        snapshot_frame.to_csv(PARAMETERS_CSV_PATH, mode="a", header=False, index=False)
    else:
        snapshot_frame.to_csv(PARAMETERS_CSV_PATH, index=False)


def load_learning_state() -> dict[str, Any]:
    if not PARAMETERS_CSV_PATH.exists():
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="initialize")
        return state

    state_frame = pd.read_csv(PARAMETERS_CSV_PATH)
    if state_frame.empty:
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="initialize")
        return state

    latest = state_frame.iloc[-1]
    try:
        state = {
            "feature_weights": json.loads(latest["feature_weights"]),
            "offer_weights": json.loads(latest["offer_weights"]),
            "retained_mean": json.loads(latest["retained_mean"]),
        }
    except Exception:
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    if set(state.get("feature_weights", {})) != set(LEARNING_FEATURES):
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    if set(state.get("offer_weights", {})) != set(LEARNING_FEATURES):
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    return state


def get_observation_store() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault("observations", {})


def softmax_dict(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    labels = list(score_map)
    scores = np.array([score_map[label] for label in labels], dtype=float)
    scores = scores - float(scores.max())
    weights = np.exp(scores)
    probabilities = weights / weights.sum()
    return {label: float(probability) for label, probability in zip(labels, probabilities)}

def safe_sample(
    items: list[str],
    scores: dict[str, float],
    k: int,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, float]]:
    """
    Returns (chosen_items, chosen_probs) with robust handling of zeros/degeneracy.
    - Softmax over scores
    - Smoothing
    - Fallback to uniform if needed
    - No replacement sampling (size clipped safely)
    """
    if not items:
        return [], {}

    # build probability array via softmax on provided scores
    raw = np.array([float(scores.get(it, 0.0)) for it in items], dtype=float)
    raw = raw - raw.max()  # stability
    probs = np.exp(raw)

    # smoothing to avoid exact zeros
    probs = probs + 1e-8

    # normalize (guard)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        probs = np.ones(len(items)) / len(items)
    else:
        probs = probs / s

    # filter near-zero (optional but safer)
    mask = probs > 1e-10
    filtered_items = np.array(items)[mask]
    filtered_probs = probs[mask]

    # fallback if everything filtered out
    if len(filtered_items) == 0:
        filtered_items = np.array(items)
        filtered_probs = np.ones(len(items)) / len(items)
    else:
        # renormalize after filtering
        filtered_probs = filtered_probs / filtered_probs.sum()

    # safe size
    k = int(min(k, len(filtered_items)))
    if k <= 0:
        return [], {}

    chosen = rng.choice(
        filtered_items,
        size=k,
        replace=False,
        p=filtered_probs
    )

    # map chosen → their probabilities correctly
    prob_map = {
        it: float(filtered_probs[np.where(filtered_items == it)[0][0]])
        for it in chosen
    }

    return list(chosen), prob_map

def build_feature_frame(customer_data: dict[str, Any]) -> pd.DataFrame:
    encoded_gender = label_encoder_gender.transform([customer_data["Gender"]])[0]
    base_frame = pd.DataFrame(
        {
            "CreditScore": [customer_data["CreditScore"]],
            "Gender": [encoded_gender],
            "Age": [customer_data["Age"]],
            "Tenure": [customer_data["Tenure"]],
            "Balance": [customer_data["Balance"]],
            "NumOfProducts": [customer_data["NumOfProducts"]],
            "HasCrCard": [customer_data["HasCrCard"]],
            "IsActiveMember": [customer_data["IsActiveMember"]],
            "EstimatedSalary": [customer_data["EstimatedSalary"]],
        }
    )
    geography_encoded = onehot_encoder_geo.transform([[customer_data["Geography"]]]).toarray()
    geography_frame = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )
    final_frame = pd.concat([base_frame.reset_index(drop=True), geography_frame], axis=1)
    return final_frame


def predict_churn(customer_data: dict[str, Any]) -> float:
    feature_frame = build_feature_frame(customer_data)
    scaled_data = scaler.transform(feature_frame)
    prediction = model.predict(scaled_data, verbose=0)
    return float(prediction[0][0])


def _search_update_df(df: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    df_rev = df.iloc[::-1]
    record = df_rev[df_rev["customer_id"].astype(str) == str(customer_id)]
    if record.empty:
        return None
    row = record.iloc[0]
    cur_rec = json.loads(row["cur_rec"])
    return {str(k): v for k, v in cur_rec.items()}


def _search_customer_df(df: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    df_rev = df.iloc[::-1]
    record = df_rev[df_rev["CustomerId"].astype(str) == str(customer_id)]
    if record.empty:
        return None
    row = record.iloc[0].to_dict()
    row.pop("RowNumber", None)
    return {str(k): v for k, v in row.items()}


#-? Remove transaction flow; latest actionable profile comes from update_history.csv merged onto the base customer row
def fetch_customer_record(customer_id: str) -> dict[str, Any] | None:
    base_record = _search_customer_df(customer_df, customer_id)
    if base_record is None:
        return None
    if UPDATE_CSV_PATH.exists():
        update_df = pd.read_csv(UPDATE_CSV_PATH)
        updated_fields = _search_update_df(update_df, customer_id)
        if updated_fields is not None:
            base_record.update(updated_fields)
    return base_record


#-? ADD_INTERPRETABILITY
def build_insights(customer_data: dict[str, Any], probability: float) -> list[str]:
    insights: list[str] = []
    if float(customer_data["Balance"]) < 50000:
        insights.append("low retained balance compared with typical banking relationship value")
    if int(customer_data["IsActiveMember"]) == 0:
        insights.append("customer is inactive, which is a strong churn signal")
    if int(customer_data["NumOfProducts"]) <= 1:
        insights.append("limited product holding suggests weaker bank stickiness")
    if int(customer_data["Age"]) >= 55:
        insights.append("senior segment may need tailored servicing and wealth support")
    if int(customer_data["Tenure"]) <= 2:
        insights.append("short tenure indicates an early-stage relationship with lower loyalty")
    if probability >= 0.8:
        insights.append("model predicts very high churn risk requiring immediate intervention")
    elif probability >= 0.6:
        insights.append("model predicts moderate churn risk and targeted outreach is justified")
    else:
        insights.append("current profile looks comparatively stable")
    return insights[:4]

def create_offer_text(
    customer_data: dict[str, Any],
    churn_rate: float,
    offer_importance: dict[str, float],
    previous_one_rejected: int = 0,
) -> str:
    credit_score = float(customer_data["CreditScore"])
    top_offers = sorted(offer_importance.items(), key=lambda item: item[1], reverse=True)
    selected_labels = [OFFER_LIBRARY[offer_name] for offer_name, _ in top_offers]
    if credit_score >= 750:
        benefit_band = "premium"
    elif credit_score >= 650:
        benefit_band = "balanced"
    else:
        benefit_band = "supportive"
    urgency = "immediate" if churn_rate >= 0.8 else "targeted" if churn_rate >= 0.6 else "preventive"
    prefix = "Alternative offer" if previous_one_rejected else "Primary offer"
    return (
        f"{prefix}: propose a {urgency} {benefit_band} retention package centered on "
        f"{', '.join(selected_labels)} for customer {customer_data['CustomerId']}."
        f" Credit score {int(credit_score)} and churn risk {churn_rate:.2f}."
    )


def extract_llm_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text", "")
            else:
                text_value = str(item)
            text_value = str(text_value).strip()
            if text_value:
                parts.append(text_value)
        return " ".join(parts).strip()
    return str(content).strip()


@st.cache_resource
def get_groq_llm() -> Any:
    if not LANGCHAIN_AVAILABLE or not os.getenv("GROQ_API_KEY"):
        return None
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def generate_offer_text(
    customer_data: dict[str, Any],
    churn_rate: float,
    offer_importance: dict[str, float],
    offer_scores: dict[str, float],
    selected_features: list[str],
    previous_one_rejected: int = 0,
) -> str:
    global LANGCHAIN_RUNTIME_ERROR

    fallback_text = create_offer_text(
        customer_data=customer_data,
        churn_rate=churn_rate,
        offer_importance=offer_importance,
        previous_one_rejected=previous_one_rejected,
    )
    llm = get_groq_llm()
    if llm is None:
        return fallback_text

    ranked_offers = sorted(offer_importance.items(), key=lambda item: item[1], reverse=True)
    offer_lines = [
        (
            f"- {offer_name}: label={OFFER_LIBRARY[offer_name]}, "
            f"probability={probability:.4f}, raw_score={float(offer_scores.get(offer_name, 0.0)):.4f}"
        )
        for offer_name, probability in ranked_offers
    ]
    prompt = "\n".join(
        [
            "You are a bank retention strategist.",
            "Write a concise retention offer in at most 3 sentences.",
            "Use the offer importance and customer data to produce realistic offer",
            "You can include schemes,coupon,perks,relaxation but should avoid risky offers based on customer data",
            "Do not mention probabilities, raw scores, models, JSON, or Groq.",
            "Do not invent offer types outside the candidate list.",
            f"Customer ID: {customer_data['CustomerId']}",
            f"Churn risk: {churn_rate:.4f}",
            f"Credit score: {float(customer_data['CreditScore']):.0f}",
            f"Selected features to improve: {', '.join(selected_features)}",
            "Candidate offers:",
            *offer_lines,
            "Return only the final recommendation text.",
        ]
    )

    try:
        response = llm.invoke(prompt)
        offer_text = extract_llm_text(response)
        if offer_text:
            LANGCHAIN_RUNTIME_ERROR = ""
            return offer_text
        LANGCHAIN_RUNTIME_ERROR = "Groq returned an empty offer response."
    except Exception as exc:
        LANGCHAIN_RUNTIME_ERROR = str(exc)
    return fallback_text


#-? Update history is the customer outcome log; parameters are stored separately in parameters.csv
def add_to_csv(
    pre_rec: pd.Series,
    cur_rec: pd.Series,
    pre_churn: float,
    current_churn: float,
    offer_given: dict[str, float],
    offer_string: str,
    customer_id: str,
    accepted_offer: bool,
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "pre_rec": json.dumps(serialize_series(pre_rec)),
        "cur_rec": json.dumps(serialize_series(cur_rec)),
        "pre_churn": float(pre_churn),
        "current_churn": float(current_churn),
        "offer_given": json.dumps({k: float(v) for k, v in offer_given.items()}),
        "offer_string": offer_string,
        "accepted_offer": int(accepted_offer),
    }
    update_frame = pd.DataFrame([payload])
    UPDATE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if UPDATE_CSV_PATH.exists():
        update_frame.to_csv(UPDATE_CSV_PATH, mode="a", header=False, index=False)
    else:
        update_frame.to_csv(UPDATE_CSV_PATH, index=False)


def remove_from_observation(customer_id: str) -> dict[str, Any] | None:
    return get_observation_store().pop(customer_id, None)


def simulate_post_offer_record(
    pre_rec: pd.Series,
    retained_mean: pd.Series,
    offer_given: dict[str, float],
    accepted_offer: bool,
    state: dict[str, Any],
) -> pd.Series:
    if not accepted_offer:
        return pre_rec.copy()
    current = pre_rec.copy()
    for offer_name, importance in offer_given.items():
        for feature in ACTIONABLE_FEATURES:
            learned_effect = float(state["offer_weights"][feature].get(offer_name, 0.0))
            gap = float(retained_mean[feature] - current[feature])
            delta = gap * float(importance) * learned_effect
            if feature == "Balance":
                current[feature] = max(current[feature] + delta, 0.0)
            elif feature == "NumOfProducts":
                current[feature] = min(max(current[feature] + delta, 1.0), 4.0)
            elif feature in {"HasCrCard", "IsActiveMember"}:
                current[feature] = min(max(current[feature] + delta, 0.0), 1.0)
    return current

def compute_reward(update: dict[str, Any]) -> float:
    alpha = 0.8
    beta = 0.2

    accepted = int(update.get("accepted_offer", 0))
    pre_churn = float(update["pre_churn"])
    current_churn = float(update["current_churn"])

    acceptance_signal = 1.0 if accepted == 1 else -1.0
    churn_signal = 1.0 if current_churn <= pre_churn else -1.0

    reward = alpha * acceptance_signal + beta * churn_signal

    # ---------- structured feedback influence ----------
    reason = update.get("feedback_reason", "")

    if reason == "Liked benefits":
        reward += 0.2
    elif reason == "Too expensive":
        reward -= 0.2
    elif reason == "Not relevant":
        reward -= 0.2
    elif reason == "Prefers other offer":
        reward -= 0.3

    return float(np.clip(reward, -1.0, 1.0))

def update_offer_weights(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    pre_rec = pd.Series(update["pre_rec"], dtype=float)
    cur_rec = pd.Series(update["cur_rec"], dtype=float)
    offer_given = {
        offer_name: float(weight)
        for offer_name, weight in update["offer_given"].items()
        if float(weight) != 0.0
    }
    if not offer_given:
        return state

    d_rec = cur_rec - pre_rec
    reward = compute_reward(update)
    normalization_base = (retained_mean - pre_rec).abs().replace(0.0, 1.0)
    relative_change = (d_rec.abs() / normalization_base).clip(lower=0.0, upper=1.0)

    for offer_name, offer_strength in offer_given.items():
        for feature in ACTIONABLE_FEATURES:
            alpha = float(relative_change.get(feature, 0.0))
            if alpha <= 0:
                continue
            current_weight = float(state["offer_weights"][feature][offer_name])
            learning_rate = 0.1
            updated_weight = current_weight + (learning_rate * reward * alpha * offer_strength)
            state["offer_weights"][feature][offer_name] = float(np.clip(updated_weight, -1.0, 1.0))
    return state

def update_feature_weights(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    pre_rec = pd.Series(update["pre_rec"], dtype=float)
    cur_rec = pd.Series(update["cur_rec"], dtype=float)

    d_rec = cur_rec - pre_rec
    reward = compute_reward(update)
    direction = reward

    for feature in ACTIONABLE_FEATURES:
        change = abs(float(d_rec.get(feature, 0.0)))

        if change == 0:
            continue

        current_weight = float(state["feature_weights"][feature])

        updated_weight = current_weight + direction * change * 0.1

        state["feature_weights"][feature] = float(
            np.clip(updated_weight, -1.0, 1.0)
        )

    return state

def get_customer_offer_history(customer_id: str) -> list[dict[str, Any]]:
    if not UPDATE_CSV_PATH.exists():
        return []

    df = pd.read_csv(UPDATE_CSV_PATH)
    history = df[df["customer_id"].astype(str) == str(customer_id)]
    if history.empty:
        return []
    return history.tail(5).to_dict("records")


def store_in_observation(customer_id: str, observation_payload: dict[str, Any]) -> None:
    get_observation_store()[customer_id] = observation_payload


#-? excluded_features is kept so the selector can skip features if you add that constraint later
def select_offer_features(
    required_gap: pd.Series,
    feature_weights: pd.Series,
    excluded_features: set[str] | None = None,
) -> list[str]:
    excluded = excluded_features or set()
    candidate_scores: dict[str, float] = {}
    for feature in ACTIONABLE_FEATURES:
        if feature in excluded:
            continue
        gap = float(required_gap.get(feature, 0.0))
        score = gap * float(feature_weights.get(feature, 0.0))
        candidate_scores[feature] = score
    if not candidate_scores:
        fallback = (required_gap[ACTIONABLE_FEATURES] * feature_weights.reindex(ACTIONABLE_FEATURES).fillna(0.0)).sort_values(ascending=False)
        return [str(feature) for feature in fallback.index[:2]]
    sample_size = min(3, len(candidate_scores))
    features = list(candidate_scores)
    probabilities = softmax_dict(candidate_scores)
    
    selected, _ = safe_sample(
    items=features,
    scores=candidate_scores,
    k=sample_size,
    rng=RNG)

    return [str(feature) for feature in selected]


def give_offer(
    customer_data: dict[str, Any],
    churn_rate: float,
    state: dict[str, Any],
    excluded_offers: set[str] | None = None,
    previous_one_rejected: int = 0,
) -> dict[str, Any]:
    current_rec = customer_to_learning_vector(customer_data)
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    feature_weights = pd.Series(state["feature_weights"], dtype=float)
    required_gap = retained_mean - current_rec
    selected_features = select_offer_features(required_gap, feature_weights)
    if not selected_features:
        selected_features = ACTIONABLE_FEATURES[:2]

    selected_feature_scores = {
        feature: float(required_gap.get(feature, 0.0)) * float(feature_weights.get(feature, 0.0))
        for feature in selected_features
    }
    history = get_customer_offer_history(customer_data["CustomerId"])
    history_bonus = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}
    history_penalty = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}

    for record in history:
        past_offers = json.loads(record["offer_given"])
        for offer_name in past_offers:
            if int(record["accepted_offer"]) == 1:
                history_bonus[offer_name] += 0.2
            else:
                history_penalty[offer_name] += 0.3

    offer_scores = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}
    for feature in selected_features:
        gap = float(required_gap.get(feature, 0.0))
        feature_weight = float(feature_weights.get(feature, 0.0))
        for offer_name, weight in state["offer_weights"][feature].items():
            base = gap * feature_weight * float(weight)
            offer_scores[offer_name] += (
                base
                + history_bonus.get(offer_name, 0.0)
                - history_penalty.get(offer_name, 0.0)
            )

    if float(customer_data["CreditScore"]) >= 650:
        offer_scores.pop("advisor_callback", None)

    blocked_offers = excluded_offers or set()
    available_offer = {
        offer_name: score
        for offer_name, score in offer_scores.items()
        if offer_name not in blocked_offers
    }
    if not available_offer:
        available_offer = offer_scores

    available_offer_probabilities = softmax_dict(available_offer)
    ranked_offers = sorted(
        available_offer_probabilities,
        key=lambda offer_name: available_offer_probabilities[offer_name],
        reverse=True,
    )
    top_offer_pool = ranked_offers[: min(3, len(ranked_offers))]
    selection_size = min(2, len(top_offer_pool))
    top_offer_probabilities = softmax_dict(
        {offer_name: float(available_offer[offer_name]) for offer_name in top_offer_pool}
    )
    
    chosen_offers, offers_importance = safe_sample(
    items=top_offer_pool,
    scores=available_offer,   # use scores for softmax
    k=selection_size,
    rng=RNG)
    chosen_offer_scores = {offer_name: float(available_offer[offer_name]) for offer_name in chosen_offers}
    offer_rankings = [
        {
            "offer_name": offer_name,
            "label": OFFER_LIBRARY[offer_name],
            "score": float(available_offer[offer_name]),
            "probability": float(available_offer_probabilities[offer_name]),
        }
        for offer_name in ranked_offers
    ]
    offer_string = generate_offer_text(
        customer_data=customer_data,
        churn_rate=churn_rate,
        offer_importance=offers_importance,
        offer_scores=chosen_offer_scores,
        selected_features=selected_features,
        previous_one_rejected=previous_one_rejected,
    )
    return {
        "selected_features": selected_features,
        "selected_feature_scores": selected_feature_scores,
        "offers_importance": offers_importance,
        "offer_scores": chosen_offer_scores,
        "offer_rankings": offer_rankings,
        "offer_string": offer_string,
        "offer_labels": [OFFER_LIBRARY[offer_name] for offer_name in offers_importance],
        "current_rec": serialize_series(current_rec),
    }


langchain_executor = None


def analyze_customer(customer_id: str, customer_query: str) -> dict[str, Any]:
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    probability = predict_churn(customer_data)
    insights = build_insights(customer_data, probability)
    state = load_learning_state()
    offer_payload = give_offer(customer_data, probability, state)

    store_in_observation(
        customer_id,
        {
            "customer_id": customer_id,
            "pre_rec": offer_payload["current_rec"],
            "pre_churn": float(probability),
            "offer_given": offer_payload["offers_importance"],
            "offer_string": offer_payload["offer_string"],
            "selected_features": offer_payload["selected_features"],
            "customer_query": customer_query,
            "created_at": datetime.utcnow().isoformat(),
        },
    )

    recommended_actions = [offer_payload["offer_string"]]
    recommended_actions.extend(
        f"Prioritize {feature} improvement for this customer." for feature in offer_payload["selected_features"]
    )
    return {
        "customer_id": customer_id,
        "customer_query": customer_query,
        "churn_risk": round(probability, 4),
        "insights": insights,
        "recommended_actions": recommended_actions,
        "critic": {"score": None, "status": "pending"},
        "offer_context": {
            "selected_features": offer_payload["selected_features"],
            "selected_feature_scores": offer_payload["selected_feature_scores"],
            "offers_importance": offer_payload["offers_importance"],
            "offer_scores": offer_payload["offer_scores"],
            "offer_rankings": offer_payload["offer_rankings"],
            "offer_labels": offer_payload["offer_labels"],
        },
    }


def process_feedback(
    latest_result: dict[str, Any],
    accepted_offer: bool,
    feedback_reason: str
) -> dict[str, Any]:
    customer_id = latest_result["customer_id"]
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    state = load_learning_state()
    observation = remove_from_observation(customer_id)
    if observation is None:
        raise ValueError("No active observation found for this customer. Analyze the customer again before submitting feedback.")

    pre_rec = pd.Series(observation["pre_rec"], dtype=float)
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    pre_churn = float(observation["pre_churn"])
    offer_given = {key: float(value) for key, value in observation["offer_given"].items()}
    cur_rec = simulate_post_offer_record(pre_rec, retained_mean, offer_given, accepted_offer, state)
    current_churn = predict_churn_from_learning_vector(customer_data, cur_rec)

    add_to_csv(
        pre_rec=pre_rec,
        cur_rec=cur_rec,
        pre_churn=pre_churn,
        current_churn=current_churn,
        offer_given=offer_given,
        offer_string=observation["offer_string"],
        customer_id=customer_id,
        accepted_offer=accepted_offer,
    )

    batch_store = st.session_state.setdefault("batch_updates", [])
    batch_store.append(
        {
            "pre_rec": serialize_series(pre_rec),
            "cur_rec": serialize_series(cur_rec),
            "pre_churn": pre_churn,
            "current_churn": current_churn,
            "offer_given": offer_given,
            "accepted_offer": int(accepted_offer),
            "feedback_reason": feedback_reason
        }
    )

    if len(batch_store) >= BATCH_SIZE:
        for update in batch_store:
            state = update_offer_weights(state, update)
            state = update_feature_weights(state, update)   # ✅ NEW

        append_parameters_snapshot(state, reason="batch_update", customer_id=customer_id)
        batch_store.clear()

    latest_result["critic"] = {"score": None,"status": "removed (using reward-based learning)"}
    latest_result["post_offer_churn"] = round(current_churn, 4)

    if not accepted_offer:
        replacement_offer = give_offer(
            customer_data,
            current_churn,
            state,
            excluded_offers=set(offer_given),
            previous_one_rejected=1,
        )
        store_in_observation(
            customer_id,
            {
                "customer_id": customer_id,
                "pre_rec": replacement_offer["current_rec"],
                "pre_churn": float(current_churn),
                "offer_given": replacement_offer["offers_importance"],
                "offer_string": replacement_offer["offer_string"],
                "selected_features": replacement_offer["selected_features"],
                "customer_query": latest_result.get("customer_query", ""),
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        latest_result["recommended_actions"] = [replacement_offer["offer_string"]]
        latest_result["recommended_actions"].extend(
            f"Follow up through {label}." for label in replacement_offer["offer_labels"]
        )
        latest_result["offer_context"] = {
            "selected_features": replacement_offer["selected_features"],
            "selected_feature_scores": replacement_offer["selected_feature_scores"],
            "offers_importance": replacement_offer["offers_importance"],
            "offer_scores": replacement_offer["offer_scores"],
            "offer_rankings": replacement_offer["offer_rankings"],
            "offer_labels": replacement_offer["offer_labels"],
            "previous_one_rejected": 1,
        }

    return latest_result


st.title("Bank Customer Retention Learning Agent")
st.caption("Performance element + learning element with reward-based feedback on top of the churn model.")

with st.expander("Project Structure", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- `app.py`: Streamlit UI, churn model integration, learning loop, offer updates",
                "- `Churn_Modelling.csv`: customer profile dataset used as the prototype customer store",
                "- `model.h5`: trained TensorFlow churn model",
                "- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `Scaler.pkl`: preprocessing assets",
                "- `logs/parameters.csv`: append-only parameter snapshots for feature weights and offer weights",
                "- `logs/update_history.csv`: recorded customer offer outcomes",
            ]
        )
    )
manual_tab, agent_tab = st.tabs(["Manual Prediction", "Learning Agent"])

with manual_tab:
    st.subheader("Direct churn prediction")
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)

    age = st.slider("Age", 18, 92)
    tenure = st.slider("Tenure", 0, 10)
    balance = st.number_input("Balance", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
    num_of_products = st.slider("Number of Products", 1, 4)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])

    if st.button("Predict Churn"):
        manual_customer = {
            "CustomerId": "manual_input",
            "CreditScore": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography": geography,
        }
        probability = predict_churn(manual_customer)
        manual_offer_payload = give_offer(manual_customer, probability, load_learning_state())
        manual_actions = [manual_offer_payload["offer_string"]]
        manual_actions.extend(
            f"Prioritize {feature} improvement for this customer."
            for feature in manual_offer_payload["selected_features"]
        )
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {probability:.2f}")
        st.progress(float(probability))
        st.json(
            {
                "customer_id": "manual_input",
                "churn_risk": round(probability, 4),
                "insights": build_insights(manual_customer, probability),
                "recommended_actions": manual_actions,
                "offer_context": {
                    "selected_features": manual_offer_payload["selected_features"],
                    "selected_feature_scores": manual_offer_payload["selected_feature_scores"],
                    "offers_importance": manual_offer_payload["offers_importance"],
                    "offer_scores": manual_offer_payload["offer_scores"],
                    "offer_rankings": manual_offer_payload["offer_rankings"],
                    "offer_labels": manual_offer_payload["offer_labels"],
                },
            }
        )

with agent_tab: #-? This tab still needs UX review if you want to change the agent-side interaction further
    st.subheader("Analyze an existing customer")
    st.write("Use a `CustomerId` from `Churn_Modelling.csv`, for example `15634602`.")

    customer_ids = sorted(customer_df["CustomerId"].astype(str).unique())
    customer_id = st.selectbox("Customer ID", customer_ids, index=0)
with st.expander("Parameter Visualizer", expanded=False):
    if PARAMETERS_CSV_PATH.exists():
        params_df = pd.read_csv(PARAMETERS_CSV_PATH)

        if not params_df.empty:
            params_df["feature_weights"] = params_df["feature_weights"].apply(json.loads)
            params_df["offer_weights"] = params_df["offer_weights"].apply(json.loads)

            # ---------- Create batch index (X-axis fix) ----------
            params_df["batch_step"] = range(1, len(params_df) + 1)

            # ---------- Feature weights trend ----------
            fw_df = pd.json_normalize(params_df["feature_weights"])
            fw_df["batch_step"] = params_df["batch_step"]

            st.subheader("Feature Weights Evolution")
            st.line_chart(fw_df.set_index("batch_step"))

            # ---------- Offer weights (latest heatmap style table) ----------
            st.subheader("Current Offer Strategy")
            latest_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])
            st.dataframe(latest_offer.style.format("{:.2f}"))

            # ---------- Delta (change in last batch) ----------
            if len(params_df) > 1:
                prev_offer = pd.DataFrame(params_df.iloc[-2]["offer_weights"])
                curr_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])

                delta_offer = (curr_offer - prev_offer).round(3)

                st.subheader("Recent Learning Update (Δ weights)")
                st.dataframe(delta_offer)

                # ---------- Feature delta ----------
                prev_feature = params_df.iloc[-2]["feature_weights"]
                curr_feature = params_df.iloc[-1]["feature_weights"]

                delta_feature = {
                    k: round(float(curr_feature[k]) - float(prev_feature[k]), 4)
                    for k in curr_feature
                }

                st.subheader("Feature Importance Change")
                st.json(delta_feature)

            # ---------- Quick insights ----------
            if len(params_df) > 1:
                st.subheader("Key Insights")

                # strongest feature
                latest_fw = params_df.iloc[-1]["feature_weights"]
                top_feature = max(latest_fw, key=lambda x: abs(latest_fw[x]))

                # most changed offer weight
                if len(params_df) > 1:
                    change_abs = delta_offer.abs()
                    max_change = change_abs.stack().idxmax()

                    st.write(f"• Most influential feature: **{top_feature}**")
                    st.write(f"• Largest update: **{max_change[1]} → {max_change[0]}**")

    customer_query = st.text_area(
        "Agent Query",
        value="Analyze the customer, estimate churn risk, and recommend retention actions.",
        height=100,
    )

    st.caption("Feature selection and offer scoring are computed locally; Groq is used to turn the scored offers into natural-language recommendations when available.")
    if LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY"):
        st.caption("Groq offer generation is enabled.")
        if LANGCHAIN_RUNTIME_ERROR:
            st.info(f"Groq status: {LANGCHAIN_RUNTIME_ERROR}")
    elif LANGCHAIN_AVAILABLE:
        st.caption("LangChain packages are available, but `GROQ_API_KEY` is missing. Offer text falls back to the local template.")
    else:
        st.caption(f"LangChain packages are unavailable: {LANGCHAIN_IMPORT_ERROR}. Offer text falls back to the local template.")
    if st.button("Analyze Customer"):
        try:
            result = analyze_customer(customer_id.strip(), customer_query.strip())
            st.session_state["latest_agent_result"] = result
            st.subheader("Churn Risk")
            st.write(result["churn_risk"])

            st.subheader("Insights")
            for insight in result["insights"]:
                st.write("•", insight)

            st.subheader("Recommended Offer")
            st.write(result["recommended_actions"][0])   # main offer

            st.subheader("Suggested Improvements")
            for action in result["recommended_actions"][1:]:
                st.write("•", action)
        except Exception as exc:
            st.error(str(exc))

    latest_result = st.session_state.get("latest_agent_result")
    if latest_result:
        st.subheader("Feedback")

        # ---------- FEEDBACK UI ----------
        st.subheader("Customer Feedback")

        col1, col2 = st.columns([2, 3])

        # ---------- Acceptance (clean UX) ----------
        with col1:
            response = st.radio(
                "Customer Response",
                ["Accepted", "Rejected"],
                horizontal=True
            )
            accepted_offer = (response == "Accepted")

        # ---------- Structured feedback ----------
        with col2:
            feedback_reason = st.selectbox(
                "Feedback Reason",
                [
                    "Liked benefits",
                    "Too expensive",
                    "Not relevant",
                    "Prefers other offer",
                    "No clear reason"
                ]
            )


        # ---------- Single action button ----------
        if st.button("Update & Learn", use_container_width=True):
            try:
                updated_result = process_feedback(latest_result,accepted_offer,feedback_reason)

                st.session_state["latest_agent_result"] = updated_result

                st.success("Feedback recorded and learning updated.")

                if not accepted_offer:
                    st.info("A replacement offer has been generated.")

                st.subheader("Updated Customer Status")

                # ---------- 1. Churn ----------
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Churn Risk (Before)",
                        f"{updated_result['churn_risk']:.3f}"
                    )

                with col2:
                    st.metric(
                        "Churn Risk (After)",
                        f"{updated_result['post_offer_churn']:.3f}",
                        delta=round(updated_result['post_offer_churn'] - updated_result['churn_risk'], 4)
                    )

                # ---------- 2. Insights ----------
                st.subheader("Key Insights")
                for insight in updated_result["insights"]:
                    st.write("•", insight)

                # ---------- 3. Main Offer ----------
                st.subheader("Recommended Offer")
                st.success(updated_result["recommended_actions"][0])

                # ---------- 4. Improvements ----------
                st.subheader("Focus Areas")
                for action in updated_result["recommended_actions"][1:]:
                    st.write("•", action)

                # ---------- 5. Offer Breakdown ----------
                st.subheader("Offer Strategy Breakdown")

                offer_df = pd.DataFrame([
                    {
                        "Offer": o["label"],
                        "Score": round(o["score"], 2),
                        "Probability": round(o["probability"], 4)
                    }
                    for o in updated_result["offer_context"]["offer_rankings"]
                ])

                st.dataframe(offer_df, use_container_width=True)

                # ---------- 6. Selected Features ----------
                st.subheader("Key Drivers")

                feature_df = pd.DataFrame({
                    "Feature": list(updated_result["offer_context"]["selected_feature_scores"].keys()),
                    "Importance": list(updated_result["offer_context"]["selected_feature_scores"].values())
                })

                st.bar_chart(feature_df.set_index("Feature"))

                # ---------- 7. Learning Feedback ----------
                st.subheader("Learning Update")

                st.info("System updated using customer feedback.")

                if updated_result["post_offer_churn"] < updated_result["churn_risk"]:
                    st.success("Retention strategy improved churn risk.")
                else:
                    st.warning("No improvement observed. System will adapt next iteration.")
            except Exception as exc:
                st.error(str(exc))
st.markdown("Built with Streamlit, TensorFlow, and LangChain.")


