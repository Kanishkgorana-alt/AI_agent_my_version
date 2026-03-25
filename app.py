import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
LOG_PATH = BASE_DIR / "logs" / "agent_interactions.jsonl"

st.set_page_config(page_title="Bank Retention Learning Agent", layout="wide")


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


def append_learning_log(event: dict[str, Any]) -> None:
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


def risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "high"
    if probability >= 0.6:
        return "medium"
    return "low"


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

# We can create a csv table of updates in customer details
# offer_provided,offer_accepted,change will be extra columns in it
# Whenever user will pass customer detail
# Its history will be retrieved from the updated .csv and if not found then from original .csv on which model is trained

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


def fetch_customer_record(customer_id: str) -> dict[str, Any] | None:
    record = customer_df.loc[customer_df["CustomerId"] == str(customer_id)]
    if record.empty:
        return None
    return record.iloc[0].to_dict()


## XXXXX
def derive_transaction_history(customer_data: dict[str, Any]) -> list[dict[str, Any]]:
    balance = float(customer_data["Balance"])
    salary = float(customer_data["EstimatedSalary"])
    salary_unit = round(salary / 12, 2)
    base_date = datetime.utcnow().date()
    transactions = [
        {
            "date": str(base_date - timedelta(days=28)),
            "type": "salary_credit",
            "amount": salary_unit,
            "source": "derived_demo",
        },
        {
            "date": str(base_date - timedelta(days=16)),
            "type": "bill_payment",
            "amount": round(max(balance * 0.08, 1500), 2),
            "source": "derived_demo",
        },
        {
            "date": str(base_date - timedelta(days=5)),
            "type": "atm_withdrawal",
            "amount": round(max(balance * 0.12, 1000), 2),
            "source": "derived_demo",
        },
    ]
    return transactions

## To be modify later to use llm
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


# I think one RNN model is also required which outputs embedded vector for provided updates and action causing updates

# Define scope of suggestions like certain set of updations advice is only in our hand
# based on embedded vector create set of suggestions with attention to be taken provide the whole update blueprint to LLM it will create meaningful string for the same.
# Can also think of each element of vector as attention weight for suggestion element
# Provide feedback strings to LLM with the update Blueprint

def heuristic_retention_actions(customer_data: dict[str, Any], probability: float) -> list[str]:
    actions: list[str] = []
    if int(customer_data["IsActiveMember"]) == 0:
        actions.append("launch a relationship manager callback with digital reactivation support")
    if float(customer_data["Balance"]) > 100000:
        actions.append("offer premium savings or wealth advisory benefits to protect high-value balance")
    else:
        actions.append("offer cashback or fee-waiver incentives to increase everyday engagement")
    if int(customer_data["NumOfProducts"]) <= 1:
        actions.append("bundle a second product such as a card upgrade or savings add-on")
    if probability >= 0.8:
        actions.append("prioritize the customer for immediate retention outreach within 24 hours")
    return actions[:4]


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

# Problem generator will fluctuate any value of attention. And it will fluctuate with the probability of temprature which itself will be calculated based on customer churn value(if high take chance)
# ?? How to consider profit and betterment of Bank.(Adverserial search)
# Use as Min agent
def generate_problem_generator_actions(probability: float, customer_data: dict[str, Any]) -> list[str]:
    proposals: list[str] = []
    if probability >= 0.8:
        proposals.append("experiment with a stronger retention package combining fee reversal and advisor outreach")
    if int(customer_data["NumOfProducts"]) <= 1:
        proposals.append("test a cross-sell offer to increase product stickiness")
    if int(customer_data["IsActiveMember"]) == 0:
        proposals.append("try a dormant-account reactivation campaign with a time-boxed incentive")
    if not proposals:
        proposals.append("continue monitoring and test lightweight loyalty nudges instead of costly interventions")
    return proposals[:3]


def fallback_analysis(customer_id: str, customer_query: str) -> dict[str, Any]:
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    probability = predict_churn(customer_data)
    insights = build_insights(customer_data, probability)
    transactions = derive_transaction_history(customer_data)
    recommended_actions = heuristic_retention_actions(customer_data, probability)
    result = {
        "customer_id": customer_id,
        "customer_query": customer_query,
        "churn_risk": round(probability, 4),
        "risk_level": risk_level(probability),
        "insights": insights,
        "recommended_actions": recommended_actions,# here recommended action will be given by LLM 
        "transactions": transactions,
        "critic": {"score": None, "status": "pending"},
        "problem_generator": generate_problem_generator_actions(probability, customer_data),
        # Should add updated values due to action
    }
    append_learning_log(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "analysis",
            "mode": "fallback",
            "payload": result,
        }
    )
    return result


@st.cache_resource
def build_langchain_executor() -> Any:
    global LANGCHAIN_RUNTIME_ERROR

    if not LANGCHAIN_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
    except Exception as exc:
        LANGCHAIN_RUNTIME_ERROR = str(exc)
        return None

    @tool
    def get_customer_data(customer_id: str) -> str:
        """Fetch structured customer profile data for a bank customer by customer id."""
        customer_data = fetch_customer_record(customer_id)
        if customer_data is None:
            return json.dumps({"error": f"CustomerId {customer_id} not found"})
        serializable = {key: (value.item() if hasattr(value, "item") else value) for key, value in customer_data.items()}
        return json.dumps(serializable)

    @tool
    def get_transactions(customer_id: str) -> str:
        """Fetch recent customer transactions. In this prototype, transactions are derived from profile-level data."""
        customer_data = fetch_customer_record(customer_id)
        if customer_data is None:
            return json.dumps({"error": f"CustomerId {customer_id} not found"})
        return json.dumps(derive_transaction_history(customer_data))

    @tool
    def predict_churn_tool(customer_id: str) -> str:
        """Predict churn probability for a customer id using the trained TensorFlow churn model."""
        customer_data = fetch_customer_record(customer_id)
        if customer_data is None:
            return json.dumps({"error": f"CustomerId {customer_id} not found"})
        probability = predict_churn(customer_data)
        return json.dumps(
            {
                "customer_id": customer_id,
                "churn_risk": round(probability, 4),
                "risk_level": risk_level(probability),
            }
        )

    @tool
    def generate_retention_strategy(customer_id: str) -> str:
        """Generate personalized retention actions for a customer using profile and churn indicators."""
        customer_data = fetch_customer_record(customer_id)
        if customer_data is None:
            return json.dumps({"error": f"CustomerId {customer_id} not found"})
        probability = predict_churn(customer_data)
        payload = {
            "recommended_actions": heuristic_retention_actions(customer_data, probability),
            "insights": build_insights(customer_data, probability),
            "problem_generator": generate_problem_generator_actions(probability, customer_data),
        }
        return json.dumps(payload)

    tools = [get_customer_data, get_transactions, predict_churn_tool, generate_retention_strategy]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the performance element of a bank customer retention learning agent. "
                "Always use the available tools before finalizing your answer. "
                "Return valid JSON only with keys customer_id, churn_risk, risk_level, insights, recommended_actions. "
                "Use lower-case risk_level values: low, medium, high. "
                "Do not include markdown fences or extra commentary.",
            ),
            ("human", "Customer ID: {customer_id}\nUser query: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
        LANGCHAIN_RUNTIME_ERROR = ""
        return AgentExecutor(agent=agent, tools=tools, verbose=False)
    except Exception as exc:
        LANGCHAIN_RUNTIME_ERROR = str(exc)
        return None


langchain_executor = build_langchain_executor()


# So, there are 2 possible suggestion agents one is fallback_analysis(which is hardcoded) another is LLM(langchain implemented)
def run_learning_agent(customer_id: str, customer_query: str) -> dict[str, Any]:
    global LANGCHAIN_RUNTIME_ERROR

    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    if langchain_executor is None:
        return fallback_analysis(customer_id, customer_query)

    try:
        response = langchain_executor.invoke({"customer_id": customer_id, "input": customer_query})
    except Exception as exc:
        LANGCHAIN_RUNTIME_ERROR = str(exc)
        append_learning_log(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "langchain_error",
                "payload": {"customer_id": customer_id, "error": LANGCHAIN_RUNTIME_ERROR},
            }
        )
        return fallback_analysis(customer_id, customer_query)

    raw_output = response.get("output", "{}")
    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError:
        LANGCHAIN_RUNTIME_ERROR = "LangChain returned non-JSON output."
        return fallback_analysis(customer_id, customer_query)
    probability = float(parsed_output["churn_risk"])
    parsed_output["customer_query"] = customer_query
    parsed_output["critic"] = {"score": None, "status": "pending"}
    parsed_output["problem_generator"] = generate_problem_generator_actions(probability, customer_data)
    append_learning_log(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "analysis",
            "mode": "langchain",
            "payload": parsed_output,
        }
    )
    return parsed_output


st.title("Bank Customer Retention Learning Agent")
st.caption("Performance element + learning element + critic + problem generator on top of the existing churn model.")

with st.expander("Project Structure", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- `app.py`: Streamlit UI, churn model integration, LangChain agent, learning loop",
                "- `Churn_Modelling.csv`: customer profile dataset used as the prototype customer store",
                "- `model.h5`: trained TensorFlow churn model",
                "- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `Scaler.pkl`: preprocessing assets",
                "- `logs/agent_interactions.jsonl`: runtime learning log written by the app",
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
        level = risk_level(probability)

        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {probability:.2f}")
        st.progress(float(probability))
        st.write(f"Risk Level: {level.upper()}")
        st.json(
            {
                "customer_id": "manual_input",
                "churn_risk": round(probability, 4),
                "risk_level": level,
                "insights": build_insights(manual_customer, probability),
                "recommended_actions": heuristic_retention_actions(manual_customer, probability),
            }
        )

with agent_tab:
    st.subheader("Analyze an existing customer")
    st.write("Use a `CustomerId` from `Churn_Modelling.csv`, for example `15634602`.")

    customer_id = st.text_input("Customer ID", value="15634602")
    customer_query = st.text_area(
        "Agent Query",
        value="Analyze the customer, estimate churn risk, and recommend retention actions.",
        height=100,
    )

    if LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY") and langchain_executor is not None:
        st.caption("LangChain agent mode is active with Groq tool-calling.")
    elif LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY"):
        st.caption(
            "Groq-backed agent initialization failed. Falling back to deterministic analysis."
        )
        if LANGCHAIN_RUNTIME_ERROR:
            st.info(f"Groq fallback reason: {LANGCHAIN_RUNTIME_ERROR}")
    elif LANGCHAIN_AVAILABLE:
        st.caption("LangChain packages are available, but `GROQ_API_KEY` is missing. Falling back to deterministic analysis.")
    else:
        st.caption(f"LangChain packages are unavailable: {LANGCHAIN_IMPORT_ERROR}. Falling back to deterministic analysis.")

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
            assessment = critic_score(
                float(latest_result["churn_risk"]),
                accepted_offer,
                feedback_text,
            )
            latest_result["critic"] = assessment
            append_learning_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "feedback",
                    "customer_id": latest_result["customer_id"],
                    "accepted_offer": accepted_offer,
                    "feedback": feedback_text,
                    "critic": assessment,
                }
            )
            st.success(f"Critic recorded feedback with status `{assessment['status']}` and score `{assessment['score']}`.")
            st.json(latest_result)

st.markdown("Built with Streamlit, TensorFlow, and LangChain.")

