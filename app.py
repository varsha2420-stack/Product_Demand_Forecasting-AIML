# ==========================================================
# Retail Demand Forecasting System
# AI Powered Sales Prediction Dashboard
# ==========================================================

# ----------------------------------------------------------
# Import Libraries
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(

    page_title="Retail Demand Forecasting",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="collapsed"

)

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

model = joblib.load("xgboost_model.pkl")

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

st.markdown("""

<style>

.main{

    padding-top:25px;

}

.block-container{

    padding-top:2rem;

    padding-bottom:2rem;

    padding-left:5rem;

    padding-right:5rem;

}

h1{

    text-align:center;

    color:#1E3A8A;

    font-weight:700;

}

h3{

    color:#2563EB;

}

div[data-testid="stMetric"]{

    background-color:#F8FAFC;

    border-radius:12px;

    padding:15px;

    border:1px solid #E5E7EB;

}

.stButton>button{

    width:100%;

    height:55px;

    font-size:20px;

    font-weight:bold;

    border-radius:12px;

    background-color:#2563EB;

    color:white;

}

.stButton>button:hover{

    background-color:#1D4ED8;

}

</style>

""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Dashboard Header
# ----------------------------------------------------------

st.markdown("""

<h1>📈 Retail Demand Forecasting</h1>

<h4 style='text-align:center;color:gray;'>

AI Powered Sales Prediction Dashboard

</h4>

""", unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------
# Welcome Card
# ----------------------------------------------------------

st.info(

"""
### Welcome

Enter the product and date information below to forecast retail demand.

This dashboard uses an **XGBoost Machine Learning Model**
to generate accurate sales predictions.

"""

)

st.divider()

# ==========================================================
# Input Section
# ==========================================================

st.subheader("📝 Enter Product Details")

left_col, right_col = st.columns(2)

# ----------------------------------------------------------
# Left Column
# ----------------------------------------------------------

with left_col:

    st.markdown("### 🏪 Store & Product Information")

    store_id = st.number_input(

        "Store ID",

        min_value=0,

        value=0,

        step=1

    )

    item_id = st.number_input(

        "Item ID",

        min_value=0,

        value=0,

        step=1

    )

    price = st.number_input(

        "Price (₹)",

        min_value=0.0,

        value=20.00,

        step=0.01

    )

    promotion = st.selectbox(

        "Promotion",

        [

            "No Promotion",

            "Promotion Available"

        ]

    )

# ----------------------------------------------------------
# Right Column
# ----------------------------------------------------------

with right_col:

    st.markdown("### 📅 Date Information")

    weekday_name = st.selectbox(

        "Weekday",

        [

            "Monday",

            "Tuesday",

            "Wednesday",

            "Thursday",

            "Friday",

            "Saturday",

            "Sunday"

        ]

    )

    month_name = st.selectbox(

        "Month",

        [

            "January",

            "February",

            "March",

            "April",

            "May",

            "June",

            "July",

            "August",

            "September",

            "October",

            "November",

            "December"

        ]

    )

    year = st.number_input(

        "Year",

        min_value=2019,

        max_value=2035,

        value=2019,

        step=1

    )

    day = st.slider(

        "Day",

        min_value=1,

        max_value=31,

        value=1

    )

st.divider()

# ==========================================================
# Convert Inputs for Model
# ==========================================================

promo = 1 if promotion == "Promotion Available" else 0

weekday_map = {

    "Monday":1,

    "Tuesday":2,

    "Wednesday":3,

    "Thursday":4,

    "Friday":5,

    "Saturday":6,

    "Sunday":7

}

month_map = {

    "January":1,

    "February":2,

    "March":3,

    "April":4,

    "May":5,

    "June":6,

    "July":7,

    "August":8,

    "September":9,

    "October":10,

    "November":11,

    "December":12

}

weekday = weekday_map[weekday_name]

month = month_map[month_name]

# ==========================================================
# Prediction Section
# ==========================================================

st.markdown("## 🔮 Sales Prediction")

if st.button("Predict Sales", use_container_width=True):

    # Create Input DataFrame
    input_data = pd.DataFrame({

        "store_id": [store_id],

        "item_id": [item_id],

        "price": [price],

        "promo": [promo],

        "weekday": [weekday],

        "month": [month],

        "year": [year],

        "day": [day]

    })

    # Make Prediction
    prediction = model.predict(input_data)

    predicted_sales = float(prediction[0])

    st.divider()

    # ======================================================
    # Prediction Result
    # ======================================================

    st.markdown("## 📊 Prediction Result")

    result_col1, result_col2 = st.columns([2,1])

    with result_col1:

        st.metric(

            label="Predicted Sales",

            value=f"{predicted_sales:.2f} Units"

        )

    with result_col2:

        if predicted_sales >= 50:

            st.success("🟢 High Demand")

        elif predicted_sales >= 20:

            st.warning("🟡 Medium Demand")

        else:

            st.error("🔴 Low Demand")

    st.divider()

    # ======================================================
    # Prediction Summary
    # ======================================================

    st.markdown("## 📋 Prediction Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Store ID",

            "Item ID",

            "Price",

            "Promotion",

            "Weekday",

            "Month",

            "Year",

            "Day"

        ],

        "Value":[

            store_id,

            item_id,

            f"₹ {price:.2f}",

            promotion,

            weekday_name,

            month_name,

            year,

            day

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ======================================================
    # Download Prediction
    # ======================================================

    output = pd.DataFrame({

        "Store ID":[store_id],

        "Item ID":[item_id],

        "Predicted Sales":[round(predicted_sales,2)]

    })

    csv = output.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="⬇ Download Prediction",

        data=csv,

        file_name="retail_prediction.csv",

        mime="text/csv",

        use_container_width=True

    )

    # ==========================================================
# Model Information
# ==========================================================

st.divider()

st.markdown("## 🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div style="
        background:#F8FAFC;
        padding:20px;
        border-radius:15px;
        text-align:center;
        border:1px solid #E5E7EB;
    ">
    <h4>Algorithm</h4>
    <h3 style="color:#2563EB;">XGBoost</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div style="
        background:#F8FAFC;
        padding:20px;
        border-radius:15px;
        text-align:center;
        border:1px solid #E5E7EB;
    ">
    <h4>Task</h4>
    <h3 style="color:#2563EB;">Demand Forecasting</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div style="
        background:#F8FAFC;
        padding:20px;
        border-radius:15px;
        text-align:center;
        border:1px solid #E5E7EB;
    ">
    <h4>Framework</h4>
    <h3 style="color:#2563EB;">Streamlit</h3>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
"""
<div style="text-align:center;color:gray;padding:15px;">

<b>Retail Demand Forecasting System</b><br>

Powered by Python • Streamlit • XGBoost

</div>
""",
unsafe_allow_html=True
)