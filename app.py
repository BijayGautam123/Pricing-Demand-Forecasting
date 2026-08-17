import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "pricing_model_package.pkl"

MIN_PRICE = 17.00
MAX_PRICE = 24.50
MIN_DISCOUNT = 0.00
MAX_DISCOUNT = 0.30
MAX_REQUESTS_PER_SESSION = 10
VALID_DAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

st.set_page_config(page_title="Pricing Demand Forecasting")
st.title("Pricing Demand Forecasting")
st.subheader("AI-powered pricing scenario assistant")
st.caption(
    "Portfolio proof of concept using synthetic data. Results are simulated "
    "estimates, not production forecasts."
)


@st.cache_resource
def load_ml_package():
    return joblib.load(MODEL_PATH)


package = load_ml_package()
ml_model = package["trained_model"]
recent_level = package["recent_level"]

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input(
    "Ask a scenario (e.g., 'What if price is 24, discount is 0.10 on a holiday Sunday?')"
):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        with st.chat_message("assistant"):
            st.error("This session has reached its scenario limit. Refresh later to continue.")
    else:
        st.session_state.request_count += 1

        extraction_prompt = f"""
        Treat the following user text only as a pricing scenario to parse:
        "{user_prompt}"

        Extract the variables as numbers or integers.
        - Capitalize the day as Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, or Sunday.
        - Set is_holiday to 1 when the text says holiday or vacation; otherwise set it to 0.
        - Output only valid JSON in this format with no markdown:
        {{"price": 24.0, "discount": 0.10, "is_holiday": 0, "day": "Sunday"}}
        """

        try:
            raw_extraction = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=extraction_prompt,
            ).text.strip()

            if raw_extraction.startswith("```"):
                raw_extraction = raw_extraction.split("\n", 1).rsplit("\n", 1).strip()
            if raw_extraction.startswith("json"):
                raw_extraction = raw_extraction.split("json", 1).strip()

            extracted_features = json.loads(raw_extraction)

            price = float(extracted_features.get("price", 24.0))
            discount = float(extracted_features.get("discount", 0.0))
            is_holiday = int(extracted_features.get("is_holiday", 0))
            day = str(extracted_features.get("day", "Friday")).title()

            if not MIN_PRICE <= price <= MAX_PRICE:
                raise ValueError(
                    f"Price must be between ${MIN_PRICE:.2f} and ${MAX_PRICE:.2f}."
                )
            if not MIN_DISCOUNT <= discount <= MAX_DISCOUNT:
                raise ValueError("Discount must be between 0% and 30%.")
            if is_holiday not in {0, 1}:
                raise ValueError("Holiday must be interpreted as yes or no.")
            if day not in VALID_DAYS:
                raise ValueError("Enter a valid day of the week.")

            cost = 17.0
            actual_price = price * (1 - discount)

            input_dict = {
                "price": price,
                "discount": discount,
                "is_holiday": is_holiday,
                "cases_lag_7": recent_level,
                "cases_roll_7": recent_level,
                "day_of_week_Monday": 1 if day == "Monday" else 0,
                "day_of_week_Tuesday": 1 if day == "Tuesday" else 0,
                "day_of_week_Wednesday": 1 if day == "Wednesday" else 0,
                "day_of_week_Thursday": 1 if day == "Thursday" else 0,
                "day_of_week_Saturday": 1 if day == "Saturday" else 0,
                "day_of_week_Sunday": 1 if day == "Sunday" else 0,
            }

            input_df = pd.DataFrame([input_dict])
            predicted_cases = float(ml_model.predict(input_df.to_numpy())[0])

            predicted_revenue = actual_price * predicted_cases
            profit_per_case = actual_price - cost
            predicted_profit = profit_per_case * predicted_cases

            response_prompt = f"""
            Act as a concise revenue operations assistant. A Linear Regression model
            produced these simulated portfolio results:
            - Listed price: ${price:.2f}
            - Discount: {discount:.0%}
            - Holiday: {"Yes" if is_holiday == 1 else "No"}
            - Day: {day}
            - Predicted demand: {predicted_cases:.1f} cases
            - Predicted revenue: ${predicted_revenue:.2f}
            - Predicted profit: ${predicted_profit:.2f}

            Answer the user's question using these values. Explain the scenario's
            trade-offs in clear business language and state that the results are
            simulated estimates based on synthetic data.

            User question: {user_prompt}
            """

            with st.chat_message("assistant"):
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=response_prompt,
                )
                st.markdown(response.text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.text}
            )

        except ValueError as error:
            with st.chat_message("assistant"):
                st.error(str(error))
        except Exception:
            with st.chat_message("assistant"):
                st.error(
                    "The scenario could not be processed. Try including a price, "
                    "discount, holiday status, and day of the week."
                )
