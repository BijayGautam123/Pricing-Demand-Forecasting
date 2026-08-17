import streamlit as st
import pandas as pd
import joblib
from google import genai
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "pricing_model_package.pkl"

# 1. SETUP THE DASHBOARD UI WITH A BRAND NEW DISTINCT TITLE
st.set_page_config(page_title="ML Pricing Core", page_icon="🚀")
st.title("🚀 ML-Powered Executive Pricing Core")
st.subheader("Your exact Linear Regression pipeline is now running live calculations")

# 2. LOAD YOUR REAL MACHINE LEARNING PACKAGE
@st.cache_resource
def load_ml_package():
    # Pulls your saved package dictionary file back into memory
    return joblib.load(MODEL_PATH)

package = load_ml_package()
ml_model = package["trained_model"]
recent_level = package["recent_level"]

# 3. INITIALIZE THE GOOGLE AI CLIENT
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat logs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. STREAMLIT INTERACTIVE INPUT
if user_prompt := st.chat_input("Ask a scenario (e.g., 'What if price is 24, discount is 0.10 on a holiday Sunday?')"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 5. DATA EXTRACTION: Gemini reads the user's sentence and separates the raw numbers
    extraction_prompt = f"""
    You are a precise data extraction tool. Read this user prompt: "{user_prompt}"
    Extract the variables as numbers or integers. 
    - Convert day names into standard capitalized format: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday".
    - Look for terms like "holiday" or "vacation" to set is_holiday to 1, otherwise set it to 0.
    Output ONLY valid JSON matching this format precisely with no markdown wrapping:
    {{"price": 24.0, "discount": 0.10, "is_holiday": 0, "day": "Sunday"}}
    """
    
    try:
        raw_extraction = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=extraction_prompt
        ).text.strip()
        
        # Strip structural markdown block wraps if Gemini accidentally appends them
        if raw_extraction.startswith("```"):
            raw_extraction = raw_extraction.split("\n", 1).rsplit("\n", 1).strip()
        if raw_extraction.startswith("json"):
            raw_extraction = raw_extraction.split("json", 1).strip()
            
        extracted_features = json.loads(raw_extraction)
        
        price = extracted_features.get("price", 24.0)
        discount = extracted_features.get("discount", 0.0)
        is_holiday = extracted_features.get("is_holiday", 0)
        day = extracted_features.get("day", "Friday")
        
        # 6. REPLICATE YOUR NOTEBOOK'S FEATURE ROW PIPELINE EXACTLY
        cost = 17.0
        actual_price = price * (1 - discount)
        
        # Match your exact X1.columns dictionary mapping configuration
        input_dict = {
            "price": price,
            "discount": discount,
            "is_holiday": is_holiday,
            "cases_lag_7": recent_level,  # Uses your notebook's real tail(14) calculation!
            "cases_roll_7": recent_level, # Uses your notebook's real tail(14) calculation!
            "day_of_week_Monday": 1 if day == "Monday" else 0,
            "day_of_week_Tuesday": 1 if day == "Tuesday" else 0,
            "day_of_week_Wednesday": 1 if day == "Wednesday" else 0,
            "day_of_week_Thursday": 1 if day == "Thursday" else 0,
            "day_of_week_Saturday": 1 if day == "Saturday" else 0,
            "day_of_week_Sunday": 1 if day == "Sunday" else 0
        }
        
        # Convert dictionary to single-row DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # RUN THE LIVE MODEL FORECAST MATHEMATICS
        live_cases_prediction = float(ml_model.predict(input_df))
        
        # Calculate real financial metrics using your backend model outputs
        predicted_revenue = actual_price * live_cases_prediction
        profit_per_case = actual_price - cost
        predicted_net_profit = profit_per_case * live_cases_prediction
        
        # 7. CONTEXT INJECTION: Hand the fresh live math results straight to Gemini
        system_instructions = f"""
        You are an elite corporate Revenue Operations assistant. 
        The user asked a custom scenario. Our backend Linear Regression machine learning model just calculated the exact math results:
        - Input Selected Strategy Price: ${price:.2f}
        - Input Discount Applied: {discount:.0%}
        - Is Holiday Option Selected: {"Yes" if is_holiday == 1 else "No"}
        - Selected Operational Day: {day}
        - LIVE MODEL DEMAND FORECAST: {live_cases_prediction:.1f} cases sold
        - FORECASTED TOTAL NET PROFIT: ${predicted_net_profit:.2f}
        - FORECASTED TOTAL REVENUE: ${predicted_revenue:.2f}
        
        Answer the user's question dynamically using these exact numbers. Explain the trade-offs of this specific scenario in professional business terms.
        """
        
        with st.chat_message("assistant"):
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=f"{system_instructions}\n\nUser Question: {user_prompt}"
            )
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error processing your custom machine learning model input: {e}")
