import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any,cast,Hashable

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
LOG_PATH = BASE_DIR / "logs" / "agent_interactions.jsonl" #-?
LEARNING_STATE_PATH = BASE_DIR / "logs" / "learning_state.json" #-?
UPDATE_CSV_PATH = BASE_DIR / "logs" / "update_history.csv"
BATCH_UPDATE_THRESHOLD = 20

ACTIONABLE_FEATURES = [
    "CreditScore",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
LEARNING_FEATURES=ACTIONABLE_FEATURES
OFFER_LIBRARY = {
    "cashback_bonus": "cashback bonus on card and debit spends",
    "fee_waiver": "temporary waiver on service and transfer fees",
    "rate_bonus": "higher savings return for a limited retention window",
    "credit_limit_review": "priority credit review with card upgrade support",
    "product_bundle": "bundled second product with loyalty benefits",
    "advisor_callback": "dedicated advisor callback with tailored account guidance", #-?Should be done if credit score is low
}
OFFER_FEATURES = list(OFFER_LIBRARY.keys())
OFFER_EFFECTS = { #-?In initial state we can't determine which offer increases which feature how much
    "cashback_bonus": {"Balance": 0.35, "IsActiveMember": 0.65},
    "fee_waiver": {"Balance": 0.45, "IsActiveMember": 0.55},
    "rate_bonus": {"Balance": 0.8, "IsActiveMember": 0.2},
    "credit_limit_review": {"CreditScore": 0.7, "HasCrCard": 0.3},
    "product_bundle": {"NumOfProducts": 0.75, "HasCrCard": 0.25},
    "advisor_callback": {"IsActiveMember": 0.7, "Balance": 0.3},
}
RNG = np.random.default_rng()

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


def append_learning_log(event: dict[str, Any]) -> None: #-?These is not required if sates are stored in update log(with id)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


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

# ======================================================================================================================================= 
def min_max_standardize(values: pd.Series | dict[str, float] | list[float] | np.ndarray) -> pd.Series:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return series
    min_value = float(series.min())
    max_value = float(series.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - min_value) / (max_value - min_value)

# ??
@st.cache_data
def build_learning_feature_dataset() -> tuple[pd.DataFrame, pd.Series]:
    base_features = customer_df[LEARNING_FEATURES].astype(float).copy()
    target = customer_df["Exited"].astype(int).copy()
    return base_features, target

#-?Why both above and below functions are required

def customer_to_learning_vector(customer_data: dict[str, Any]) -> pd.Series:
    return pd.Series(
        {feature: float(customer_data[feature]) for feature in LEARNING_FEATURES},
        dtype=float,
    )


def serialize_series(series: pd.Series) -> dict[str, float]:
    return {key: float(value) for key, value in series.items()}

#-? Why this is here, who said to round off values 
def predict_churn_from_learning_vector(customer_data: dict[str, Any], learning_vector: pd.Series) -> float:
    updated_customer = dict(customer_data)
    for feature, value in learning_vector.items():
        if feature in {"NumOfProducts", "HasCrCard", "IsActiveMember"}:
            updated_customer[feature] = int(round(float(value)))
        else:
            updated_customer[feature] = float(value)
    return predict_churn(updated_customer)

# =======================================================================================================================================
def initialize_learning_state() -> dict[str, Any]:
    feature_frame, target = build_learning_feature_dataset()
    retained_mean = feature_frame.loc[target == 0].mean()
    churned_mean = feature_frame.loc[target == 1].mean() #-? This is also not required
    feature_diff = retained_mean - churned_mean #-? Need difference between current feature and tar=0 feaature
    feature_weight_magnitude = min_max_standardize(feature_diff.abs()) #-? required?
    feature_weights = (feature_weight_magnitude * np.sign(feature_diff)).fillna(0.0)
    feature_weights = (feature_diff).fillna(0.0)

    offer_weights = {
        feature: {offer_name: 0.5 for offer_name in OFFER_LIBRARY}
        for feature in LEARNING_FEATURES
    }
    return {
        "retained_mean": serialize_series(retained_mean),
        "feature_weights": serialize_series(feature_weights),
        "offer_weights": offer_weights,
        "observations": {},
        "pending_updates": [],
        "added_non_updated_count": 0,
    }

#-? When I am storing in update.csv then why this
def save_learning_state(state: dict[str, Any]) -> None:
    LEARNING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEARNING_STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def load_learning_state() -> dict[str, Any]:
    if not LEARNING_STATE_PATH.exists():
        state = initialize_learning_state()
        save_learning_state(state)
        return state
    with LEARNING_STATE_PATH.open("r", encoding="utf-8") as file:
        state = json.load(file)
    if set(state.get("offer_weights", {})) != set(LEARNING_FEATURES):
        state = initialize_learning_state()
        save_learning_state(state)
    return state

#===================================================================================================================================
def softmax_dict(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    labels = list(score_map)
    scores = np.array([score_map[label] for label in labels], dtype=float)
    scores = scores - float(scores.max())
    weights = np.exp(scores)
    probabilities = weights / weights.sum()
    return {label: float(probability) for label, probability in zip(labels, probabilities)}

# ==============================================================================================================================
def get_offer_history(offer_importance: dict[str, Any],limit: int = 8) -> list[dict[str, Any]]:

    if not UPDATE_CSV_PATH.exists():
        return []

    history_frame = pd.read_csv(UPDATE_CSV_PATH)
    if history_frame.empty:
        return []

    target = np.array(
        [float(offer_importance.get(k, 0.0)) for k in OFFER_FEATURES],
        dtype=float
    )

    def row_distance(row):
        vec = np.array(
            [float(row.get(f"offer_{k}", 0.0)) for k in OFFER_FEATURES],
            dtype=float
        )
        return np.linalg.norm(vec - target)

    history_frame["__distance"] = history_frame.apply(row_distance, axis=1)

    history_frame = history_frame.sort_values("__distance")

    limit = min(limit, len(history_frame))

    return cast(
        list[dict[str, Any]],
        history_frame.head(limit)
        .drop(columns="__distance")
        .to_dict("records")
    )


# def risk_level(probability: float) -> str:
#     if probability >= 0.8:
#         return "high"
#     if probability >= 0.55:
#         return "medium"
#     return "low"

#================================================================================================================
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

#==================================================================================================================================
def predict_churn(customer_data: dict[str, Any]) -> float:
    feature_frame = build_feature_frame(customer_data)
    scaled_data = scaler.transform(feature_frame)
    prediction = model.predict(scaled_data, verbose=0)
    return float(prediction[0][0])

#-?
update_df = pd.read_csv(UPDATE_CSV_PATH)

def _search_update_df(df: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    df_rev = df.iloc[::-1]

    record = df_rev[df_rev["customer_id"].astype(str) == str(customer_id)]
    if record.empty:
        return None

    row = record.iloc[0]

    # parse cur_rec (string → dict)
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


def fetch_customer_record(customer_id: str) -> dict[str, Any] | None:
    # 1. search update history (latest state)
    result = _search_update_df(update_df, customer_id)
    if result is not None:
        return result

    # 2. fallback to original dataset
    return _search_customer_df(customer_df, customer_id)

## ----------------------------ADD_INTERPRETABILITY--------------------------
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


## Change according to feature change ,offer value , bank loss risk, churn change
#-? Pass change details to this
def critic_score(probability: float, accepted_offer: bool, feedback_text: str) -> dict[str, Any]:
    base_score = 0.2 if probability >= 0.8 else 0.5 if probability >= 0.6 else 0.7
    if accepted_offer:
        base_score += 0.25
    if feedback_text.strip():
        base_score += 0.1
    final_score = min(round(base_score, 2), 1.0)
    return {
        "score": final_score,
        "status": "success" if final_score >= 0.7 else "failure",
    }

#========================================================================================================================================
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
        f" Credit score {int(credit_score)} and churn risk {churn_rate:.2f}"
        f"{customer_data['CustomerId']['estimated_salary']} is its estimated salary"
    )


def add_to_csv(
    pre_rec: pd.Series,
    cur_rec: pd.Series,
    pre_churn: float,
    current_churn: float,
    offer_given: dict[str, float],
    offer_string: str,
    customer_id: str,
    accepted_offer: bool,
) -> dict[str, Any]:
    offer_flat = {
        f"offer_{k}": float(offer_given.get(k, 0.0))
        for k in OFFER_FEATURES
    }
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "pre_rec": json.dumps(serialize_series(pre_rec)),
        "cur_rec": json.dumps(serialize_series(cur_rec)),
        "pre_churn": float(pre_churn),
        "current_churn": float(current_churn),
        "offer_string": offer_string,
        "accepted_offer": int(accepted_offer),
        **offer_flat
    }
    update_frame = pd.DataFrame([payload])

    UPDATE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if UPDATE_CSV_PATH.exists():
        update_frame.to_csv(UPDATE_CSV_PATH, mode="a", header=False, index=False)
    else:
        update_frame.to_csv(UPDATE_CSV_PATH, index=False)
    return payload
#-? Why it is returning payload back

def remove_from_observation(state: dict[str, Any], customer_id: str) -> dict[str, Any] | None:
    return state.get("observations", {}).pop(customer_id, None)

#-? Below one will be done at customer side also Remove Credit score as I think it can't be improved by offers
def simulate_post_offer_record(
    pre_rec: pd.Series,
    retained_mean: pd.Series,
    offer_given: dict[str, float],
    accepted_offer: bool,
) -> pd.Series:
    if not accepted_offer:
        return pre_rec.copy()
    current = pre_rec.copy()
    for offer_name, importance in offer_given.items():
        for feature, effect in OFFER_EFFECTS.get(offer_name, {}).items():
            gap = max(float(retained_mean[feature] - current[feature]), 0.0)
            delta = gap * float(importance) * float(effect)
            if feature == "CreditScore":
                current[feature] = min(current[feature] + delta, 900.0)
            elif feature == "Balance":
                current[feature] = max(current[feature] + delta, 0.0)
            elif feature == "NumOfProducts":
                current[feature] = min(current[feature] + delta, 4.0)
            elif feature in {"HasCrCard", "IsActiveMember"}:
                current[feature] = min(current[feature] + delta, 1.0)
    return current


def update_offer_weights(state: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
    reference_diff = pd.Series(state["feature_reference_diff"], dtype=float)
    #-? In below codes alpha calculation need to be revised
    normalization_base = reference_diff.abs().replace(0.0, 1.0)
    for update in updates:
        pre_rec = pd.Series(update["pre_rec"], dtype=float)
        cur_rec = pd.Series(update["cur_rec"], dtype=float)
        offer_given = update["offer_given"]
        d_rec = cur_rec - pre_rec
        d_churn = float(update["current_churn"]) - float(update["pre_churn"])
        relative_change = (d_rec.abs() / normalization_base).clip(lower=0.0)
        alpha_series = min_max_standardize(relative_change).clip(lower=0.0, upper=1.0)
        for feature, alpha in alpha_series.items(): #-? Change offer value of those only which were selected don't update for the ones which were 0
            if float(alpha) <= 0:
                continue
            alignment = np.sign(d_rec[feature]) * np.sign(reference_diff[feature]) #-? Why reference_diff is used here
            change_sign = 1.0 if d_churn <= 0 and alignment >= 0 else -1.0 #-? No need to consider how feature affects churn that is done while weighing feature importance in initialize_state()
            for offer_name, offer_strength in offer_given.items():
                current_weight = float(state["offer_weights"][feature][offer_name])
                updated_weight = ((1 - float(alpha)) * current_weight) + (
                    change_sign * float(alpha) * float(offer_strength)
                )
                state["offer_weights"][feature][offer_name] = float(max(updated_weight, 0.0)) #-? No need to remove negative value, because while selecting softmax will be applied on the weights
    return state


def store_in_observation(state: dict[str, Any], customer_id: str, observation_payload: dict[str, Any]) -> None:
    state.setdefault("observations", {})[customer_id] = observation_payload

#=========================================================================================================================================
def select_offer_features(
    required_gap: pd.Series,
    feature_weights: pd.Series,
    excluded_features: set[str] | None = None,
) -> list[str]:
    excluded = excluded_features or set()
    #-? Where excluded feature will be used
    candidate_scores = {}
    for feature in ACTIONABLE_FEATURES:
        if feature in excluded:
            continue
        gap = float(required_gap.get(feature, 0.0))
        score = gap *float(feature_weights.get(feature, 0.0))
        if score > 0:
            candidate_scores[feature] = score
    if not candidate_scores:
        fallback = required_gap[ACTIONABLE_FEATURES].clip(lower=0.0).sort_values(ascending=False)
        return [feature for feature in fallback.index[:2] if float(fallback[feature]) > 0]
    sample_size = min(3, len(candidate_scores))
    features = list(candidate_scores)
    probabilities = softmax_dict({features: float(candidate_scores) for features, candidate_scores in zip(features, candidate_scores)})
    selected = RNG.choice(features, size=sample_size, replace=False, p=np.array(list(probabilities.values()),dtype=float))
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

    offer_scores = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}
    for feature in selected_features:
        gap = float(required_gap.get(feature, 0.0))
        for offer_name, weight in state["offer_weights"][feature].items():
            offer_scores[offer_name] += float(weight) * gap
    blocked_offers = excluded_offers or set()
    available_offer= {
        offer_name: probability
        for offer_name, weight in offer_scores.items()
        if offer_name not in blocked_offers
    }
    if not available_offer:
        available_offer = offer_scores
    available_offer_probabilities = softmax_dict(available_offer)

    selection_size = min(2, len(available_offer_probabilities))
    offer_names = list(available_offer_probabilities)
    offer_probs = np.array([available_offer_probabilities[name] for name in offer_names], dtype=float)
    chosen_offers = RNG.choice(offer_names, size=selection_size, replace=False, p=offer_probs)
    chosen_offer_map = {offer_name: available_offer[offer_name] for offer_name in chosen_offers}
    offers_importance = {
        offer_name:weight
        for offer_name, weight in chosen_offer_map.items()
    }
    history = get_offer_history(offers_importance,limit=5)
    offer_string = create_offer_text(
        customer_data=customer_data,
        churn_rate=churn_rate,
        offer_importance=offers_importance,
        previous_one_rejected=previous_one_rejected,
    )
    return {
        "selected_features": selected_features,
        # "required_gap": serialize_series(required_gap), already compensated in importance
        "offers_importance": offers_importance,
        "offer_string": offer_string,
        "offer_labels": [OFFER_LIBRARY[offer_name] for offer_name in offers_importance],
        "current_rec": serialize_series(current_rec),
    }


langchain_executor = None

def fallback_analysis(customer_id: str, customer_query: str) -> dict[str, Any]:
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    probability = predict_churn(customer_data)
    insights = build_insights(customer_data, probability)
    state = load_learning_state()
    offer_payload = give_offer(customer_data, probability, state)

    store_in_observation(
        state,
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
    save_learning_state(state)

    recommended_actions = [offer_payload["offer_string"]]
    recommended_actions.extend(
        f"Prioritize {feature} improvement for this customer." for feature in offer_payload["selected_features"]
    )
    result = {
        "customer_id": customer_id,
        "customer_query": customer_query,
        "churn_risk": round(probability, 4),
        # "risk_level": risk_level(probability),
        "insights": insights,
        "recommended_actions": recommended_actions,
        "critic": {"score": None, "status": "pending"},
        "offer_context": {
            "selected_features": offer_payload["selected_features"],
            "offers_importance": offer_payload["offers_importance"],
            "offer_labels": offer_payload["offer_labels"],
            "required_gap": offer_payload["required_gap"],
        },
    }
    append_learning_log(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "analysis",
            "mode": "learning_logic",
            "payload": result,
        }
    )
    return result


def run_learning_agent(customer_id: str, customer_query: str) -> dict[str, Any]:
    return fallback_analysis(customer_id, customer_query)


def process_feedback(
    latest_result: dict[str, Any],
    accepted_offer: bool,
    feedback_text: str,
) -> dict[str, Any]:
    customer_id = latest_result["customer_id"]
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    state = load_learning_state()
    observation = remove_from_observation(state, customer_id)
    if observation is None:
        raise ValueError("No active observation found for this customer. Analyze the customer again before submitting feedback.")

    pre_rec = pd.Series(observation["pre_rec"], dtype=float)
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    pre_churn = float(observation["pre_churn"])
    offer_given = {key: float(value) for key, value in observation["offer_given"].items()}
    cur_rec = simulate_post_offer_record(pre_rec, retained_mean, offer_given, accepted_offer)
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

    state.setdefault("pending_updates", []).append(
        {
            "pre_rec": serialize_series(pre_rec),
            "cur_rec": serialize_series(cur_rec),
            "pre_churn": pre_churn,
            "current_churn": current_churn,
            "offer_given": offer_given,
            "offer_string": observation["offer_string"],
        }
    )
    state["added_non_updated_count"] = int(state.get("added_non_updated_count", 0)) + 1

    if state["added_non_updated_count"] > BATCH_UPDATE_THRESHOLD and state["pending_updates"]:
        state = update_offer_weights(state, state["pending_updates"])
        state["pending_updates"] = []
        state["added_non_updated_count"] = 0

    assessment = critic_score(float(latest_result["churn_risk"]), accepted_offer, feedback_text)
    latest_result["critic"] = assessment
    latest_result["post_offer_churn"] = round(current_churn, 4)
    latest_result["feedback_notes"] = feedback_text

    if not accepted_offer:
        replacement_offer = give_offer(
            customer_data,
            current_churn,
            state,
            excluded_offers=set(offer_given),
            previous_one_rejected=1,
        )
        store_in_observation(
            state,
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
            "offers_importance": replacement_offer["offers_importance"],
            "offer_labels": replacement_offer["offer_labels"],
            "required_gap": replacement_offer["required_gap"],
            "previous_one_rejected": 1,
        }

    append_learning_log(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "feedback",
            "customer_id": customer_id,
            "accepted_offer": accepted_offer,
            "feedback": feedback_text,
            "critic": assessment,
            "post_offer_churn": round(current_churn, 4),
        }
    )
    save_learning_state(state)
    return latest_result

st.title("Bank Customer Retention Learning Agent")
st.caption("Performance element + learning element + critic on top of the existing churn model.")

with st.expander("Project Structure", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- `app.py`: Streamlit UI, churn model integration, learning loop, offer updates",
                "- `Churn_Modelling.csv`: customer profile dataset used as the prototype customer store",
                "- `model.h5`: trained TensorFlow churn model",
                "- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `Scaler.pkl`: preprocessing assets",
                "- `logs/agent_interactions.jsonl`: runtime learning log written by the app",
                "- `logs/learning_state.json`: persisted feature and offer weights plus active observations",
                "- `logs/update_history.csv`: recorded observation outcomes used for batch offer-weight updates",
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
        # level = risk_level(probability)
        manual_offer_payload = give_offer(manual_customer, probability, load_learning_state())
        manual_actions = [manual_offer_payload["offer_string"]]
        manual_actions.extend(
            f"Prioritize {feature} improvement for this customer."
            for feature in manual_offer_payload["selected_features"]
        )
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {probability:.2f}")
        st.progress(float(probability))
        # st.write(f"Risk Level: {level.upper()}")
        st.json(
            {
                "customer_id": "manual_input",
                "churn_risk": round(probability, 4),
                # "risk_level": level,
                "insights": build_insights(manual_customer, probability),
                "recommended_actions": manual_actions,
                "offer_context": {
                    "selected_features": manual_offer_payload["selected_features"],
                    "offers_importance": manual_offer_payload["offers_importance"],
                    "offer_labels": manual_offer_payload["offer_labels"],
                    "required_gap": manual_offer_payload["required_gap"],
                },
            }
        )

with agent_tab: #-? This need to be checked
    st.subheader("Analyze an existing customer")
    st.write("Use a `CustomerId` from `Churn_Modelling.csv`, for example `15634602`.")

    customer_id = st.text_input("Customer ID", value="15634602")
    customer_query = st.text_area(
        "Agent Query",
        value="Analyze the customer, estimate churn risk, and recommend retention actions.",
        height=100,
    )

    st.caption("The active path uses the local learning-agent logic with persisted feature weights, offer weights, observation memory, and batched updates.")
    if LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY") and langchain_executor is not None:
        st.caption("LangChain and Groq are available, but retention decisions are currently driven by the local learning loop.")
    elif LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY"):
        st.caption("Groq-backed initialization failed. The local learning loop remains active.")
        if LANGCHAIN_RUNTIME_ERROR:
            st.info(f"Groq status: {LANGCHAIN_RUNTIME_ERROR}")
    elif LANGCHAIN_AVAILABLE:
        st.caption("LangChain packages are available, but `GROQ_API_KEY` is missing. The local learning loop remains active.")
    else:
        st.caption(f"LangChain packages are unavailable: {LANGCHAIN_IMPORT_ERROR}. The local learning loop remains active.")
    if st.button("Analyze Customer"):
        try:
            result = run_learning_agent(customer_id.strip(), customer_query.strip())
            st.session_state["latest_agent_result"] = result
            st.json(result)
        except Exception as exc:
            st.error(str(exc))

    latest_result = st.session_state.get("latest_agent_result")
    if latest_result:
        st.subheader("Critic Feedback")
        accepted_offer = st.checkbox("Customer accepted the proposed offer", value=False)
        feedback_text = st.text_area("Feedback Notes", height=80)

        if st.button("Submit Feedback"):
            try:
                updated_result = process_feedback(latest_result, accepted_offer, feedback_text)
                st.session_state["latest_agent_result"] = updated_result
                status_text = updated_result["critic"]["status"]
                score_text = updated_result["critic"]["score"]
                st.success(f"Critic recorded feedback with status `{status_text}` and score `{score_text}`.")
                if not accepted_offer:
                    st.info("A replacement offer has been generated and stored back into observation for continued learning.")
                st.json(updated_result)
            except Exception as exc:
                st.error(str(exc))
st.markdown("Built with Streamlit, TensorFlow, and LangChain.")
























