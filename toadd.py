customer_id = st.text_input("Customer ID", value="15634602")

########################################################################
with st.expander("Parameter Visualizer", expanded=False):
    if PARAMETERS_CSV_PATH.exists():
        params_df = pd.read_csv(PARAMETERS_CSV_PATH)

        params_df["feature_weights"] = params_df["feature_weights"].apply(json.loads)
        params_df["offer_weights"] = params_df["offer_weights"].apply(json.loads)

        # -------- Feature weights over time --------
        fw_df = pd.json_normalize(params_df["feature_weights"])
        st.subheader("Feature Weights Trend")
        st.line_chart(fw_df)

        # -------- Offer heatmap --------
        st.subheader("Latest Offer Weights Heatmap")
        latest_offer = params_df.iloc[-1]["offer_weights"]
        heatmap_df = pd.DataFrame(latest_offer)

        st.dataframe(heatmap_df)

        # -------- Change detection --------
        if len(params_df) > 1:
            prev = pd.DataFrame(params_df.iloc[-2]["offer_weights"])
            curr = pd.DataFrame(params_df.iloc[-1]["offer_weights"])

            delta = curr - prev
            st.subheader("Change in Offer Weights")
            st.dataframe(delta)

########################################################################
BATCH_SIZE = 5
########################################################################
#Remove
state = update_offer_weights(...)
append_parameters_snapshot(...)

# Store batch updates
batch_store = st.session_state.setdefault("batch_updates", [])
batch_store.append({
    "pre_rec": serialize_series(pre_rec),
    "cur_rec": serialize_series(cur_rec),
    "pre_churn": pre_churn,
    "current_churn": current_churn,
    "offer_given": offer_given,
})

# Apply batch update
if len(batch_store) >= BATCH_SIZE:
    for upd in batch_store:
        state = update_offer_weights(state, upd)

    append_parameters_snapshot(state, reason="batch_update", customer_id=customer_id)
    st.success(f"Batch update applied on {len(batch_store)} records")

    batch_store.clear()
########################################################################
if PARAMETERS_CSV_PATH.exists():
    df = pd.read_csv(PARAMETERS_CSV_PATH)

    if len(df) > 1:
        prev = json.loads(df.iloc[-2]["feature_weights"])
        curr = json.loads(df.iloc[-1]["feature_weights"])

        delta = {
            k: curr[k] - prev[k]
            for k in curr
        }

        st.subheader("Feature Weight Change")
        st.json(delta)
########################################################################
def get_customer_offer_history(customer_id: str):
    if not UPDATE_CSV_PATH.exists():
        return []

    df = pd.read_csv(UPDATE_CSV_PATH)
    history = df[df["customer_id"].astype(str) == str(customer_id)]

    if history.empty:
        return []

    return history.tail(5).to_dict("records")  # last 5 interactions
########################################################################
history = get_customer_offer_history(customer_data["CustomerId"])

history_bonus = {offer: 0.0 for offer in OFFER_LIBRARY}
history_penalty = {offer: 0.0 for offer in OFFER_LIBRARY}

for record in history:
    past_offers = json.loads(record["offer_given"])

    for offer in past_offers:
        if int(record["accepted_offer"]) == 1:
            history_bonus[offer] += 0.2
        else:
            history_penalty[offer] += 0.3
########################################################################
#Replace
offer_scores[offer_name] += gap * feature_weight * float(weight)

#with
base = gap * feature_weight * float(weight)

offer_scores[offer_name] += (
    base
    + history_bonus.get(offer_name, 0)
    - history_penalty.get(offer_name, 0)
)
########################################################################

########################################################################

########################################################################