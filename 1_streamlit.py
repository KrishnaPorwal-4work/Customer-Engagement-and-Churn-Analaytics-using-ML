import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
)

# Premium Organic Earthy Palette Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdfbf7 0%, #f4efe4 100%);
        color: #362417;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #123d2e 0%, #2d7f48 42%, #d9eb9d 100%);
        color: white;
        border-radius: 18px;
        padding: 22px 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(18, 61, 46, 0.14);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #fdfbf7;
    }
    .hero p {
        margin: 0;
        font-size: 1.05rem;
        color: #e2d9c8;
        font-weight: 400;
    }

    .risk-badge {
        padding: 16px 20px;
        border-radius: 14px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .high-risk {
        background: #fbeae9;
        color: #90241b;
        border: 1.5px solid #f3b0a9;
    }
    .medium-risk {
        background: #fef7e7;
        color: #8c5b08;
        border: 1.5px solid #fce0a6;
    }
    .low-risk {
        background: #ebf5ee;
        color: #1c522f;
        border: 1.5px solid #b7e1c1;
    }

    div[data-testid="stFormSubmitButton"] > button, .stButton > button {
        background: linear-gradient(180deg, #2c5e3b 0%, #21472c 100%);
        color: #fdfbf7 !important;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(44, 94, 59, 0.25);
    }
    div[data-testid="stFormSubmitButton"] > button:hover, .stButton > button:hover {
        background: linear-gradient(180deg, #e5a93b 0%, #d48b28 100%);
        color: #362417 !important;
        box-shadow: 0 6px 18px rgba(229, 169, 59, 0.35);
    }

    [data-testid="stSidebar"] {
        background-color: #f5f0e6 !important;
        border-right: 1px solid #e6ded0;
    }
    .stSidebar label {
        color: #4a3525 !important;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stSidebar h3 {
        color: #2c5e3b !important;
        border-bottom: 2px solid #dcd3c1;
        padding-bottom: 4px;
        margin-top: 1.2rem;
        margin-bottom: 0.75rem;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .stNumberInput input, .stSelectbox select {
        background-color: #faf6ee !important;
        color: #362417 !important;
        border-radius: 8px !important;
        border: 1px solid #dcd3c1 !important;
    }

    [data-testid="stMetric"] {
        background-color: #faf6ee;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e8e0d0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stMetricLabel"] {
        color: #6b5544 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #2c5e3b !important;
        font-weight: 800;
        font-size: 1.7rem !important;
    }

    .stAlert {
        background-color: #f7ede0 !important;
        color: #4a3525 !important;
        border: 1px solid #e3d2bd !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    try:
        return joblib.load("1_Customer_churn_pipeline.pkl")
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None


artifacts = load_artifacts()

st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Risk Intelligence</h1>
        <p>Analyze engagement dynamics, predict risk thresholds, and deploy targeted retention actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts:
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
ohe = artifacts["ohe"]
threshold = artifacts.get("optimal_threshold", 0.35)
feature_names = artifacts["feature_names"]

with st.sidebar:
    st.header("👤 Customer Parameters")
    st.caption("Adjust inputs below to simulate customer behavioral profile.")

    with st.container():
        st.subheader("Demographics")
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=42, key="Age"
        )
        gender = st.selectbox(
            "Gender", ["Male", "Female", "Other"], index=0, key="Gender"
        )
        country = st.selectbox(
            "Country",
            [
                "USA",
                "UK",
                "Germany",
                "Canada",
                "India",
                "Japan",
                "France",
                "Australia",
            ],
            index=0,
            key="Country",
        )
        membership_years = st.slider(
            "Membership Years", 0.0, 10.0, value=1.2, step=0.1, key="Membership_Years"
        )

    with st.container():
        st.subheader("Platform Engagement")
        login_freq = st.slider(
            "Login Frequency / month", 0, 31, value=3, key="Login_Frequency"
        )
        session_dur = st.number_input(
            "Avg Session Duration (mins)",
            1.0,
            75.6,
            value=8.5,
            key="Session_Duration_Avg",
        )
        pages_per_session = st.number_input(
            "Pages / Session", 1.0, 24.1, value=3.2, key="Pages_Per_Session"
        )

        cart_abandonment_pct = st.slider(
            "Cart Abandonment Rate (%)",
            0.0,
            100.0,
            value=38.0,
            key="Cart_Abandonment_Rate",
        )
        wishlist_items = st.number_input(
            "Wishlist Items", 0, 28, value=2, key="Wishlist_Items"
        )
        email_open_pct = st.slider(
            "Email Open Rate (%)", 0.0, 100.0, value=8.5, key="Email_Open_Rate"
        )
        mobile_app_usage = st.number_input(
            "Mobile App Usage Score", 0.0, 100.0, value=12.0, key="Mobile_App_Usage"
        )
        social_engagement_pct = st.number_input(
            "Social Media Engagement Score",
            0.0,
            100.0,
            value=15.0,
            key="Social_Media_Engagement_Score",
        )

    with st.container():
        st.subheader("Purchase History")
        total_purchases = st.number_input(
            "Total Purchases", 0, 500, value=4, key="Total_Purchases"
        )
        avg_order_val = st.number_input(
            "Average Order Value ($)",
            0.0,
            9666.38,
            value=65.00,
            key="Average_Order_Value",
        )
        days_since_last = st.number_input(
            "Days Since Last Purchase",
            0,
            200,
            value=55,
            key="Days_Since_Last_Purchase",
        )

        discount_usage_pct = st.slider(
            "Discount Usage Rate (%)",
            0.0,
            100.0,
            value=42.0,
            key="Discount_Usage_Rate",
        )
        return_rate_pct = st.slider(
            "Returns Rate (%)", 0.0, 100.0, value=22.0, key="Returns_Rate"
        )
        payment_method_diversity = st.slider(
            "Payment Method Diversity",
            1,
            5,
            value=1,
            key="Payment_Method_Diversity",
        )

    with st.container():
        st.subheader("Support & Financials")
        cs_calls = st.number_input(
            "Customer Service Calls",
            0,
            21,
            value=7,
            key="Customer_Service_Calls",
        )
        reviews_written = st.number_input(
            "Product Reviews Written",
            0,
            21,
            value=1,
            key="Product_Reviews_Written",
        )
        lifetime_value = st.number_input(
            "Lifetime Value ($)",
            0.0,
            8987.24,
            value=260.00,
            key="Lifetime_Value",
        )
        credit_balance = st.number_input(
            "Credit Balance ($)",
            0.0,
            7197.00,
            value=15.00,
            key="Credit_Balance",
        )

    reset_button = st.button("🔄 Reset Default Values", width="stretch")
    if reset_button:
        for key in [
            "Age",
            "Gender",
            "Country",
            "Membership_Years",
            "Login_Frequency",
            "Session_Duration_Avg",
            "Pages_Per_Session",
            "Cart_Abandonment_Rate",
            "Wishlist_Items",
            "Email_Open_Rate",
            "Mobile_App_Usage",
            "Social_Media_Engagement_Score",
            "Total_Purchases",
            "Average_Order_Value",
            "Days_Since_Last_Purchase",
            "Discount_Usage_Rate",
            "Returns_Rate",
            "Payment_Method_Diversity",
            "Customer_Service_Calls",
            "Product_Reviews_Written",
            "Lifetime_Value",
            "Credit_Balance",
        ]:
            if key in st.session_state:
                st.session_state.pop(key, None)
        st.rerun()

with st.form("prediction_form"):
    submitted = st.form_submit_button("Evaluate Churn Risk", width="stretch")

if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Membership_Years": membership_years,
        "Login_Frequency": login_freq,
        "Session_Duration_Avg": session_dur,
        "Pages_Per_Session": pages_per_session,
        "Cart_Abandonment_Rate": cart_abandonment_pct,
        "Wishlist_Items": wishlist_items,
        "Total_Purchases": total_purchases,
        "Average_Order_Value": avg_order_val,
        "Days_Since_Last_Purchase": days_since_last,
        "Discount_Usage_Rate": discount_usage_pct,
        "Returns_Rate": return_rate_pct,
        "Email_Open_Rate": email_open_pct,
        "Customer_Service_Calls": cs_calls,
        "Product_Reviews_Written": reviews_written,
        "Social_Media_Engagement_Score": social_engagement_pct,
        "Mobile_App_Usage": mobile_app_usage,
        "Payment_Method_Diversity": payment_method_diversity,
        "Lifetime_Value": lifetime_value,
        "Credit_Balance": credit_balance,
    }

    try:
        input_df = pd.DataFrame([input_dict])
        cat_cols = input_df.select_dtypes(include=["object"]).columns
        num_cols = input_df.select_dtypes(exclude=["object"]).columns

        encoded_cats = ohe.transform(input_df[cat_cols])
        encoded_cats_df = pd.DataFrame(
            encoded_cats, columns=ohe.get_feature_names_out(cat_cols)
        )
        processed_df = pd.concat(
            [input_df[num_cols].reset_index(drop=True), encoded_cats_df], axis=1
        )
        processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
        scaled_input = scaler.transform(processed_df)

        churn_prob = model.predict_proba(scaled_input)[0, 1]

        if churn_prob >= 0.60:
            segment = "High Risk"
            badge_class = "high-risk"
            action = "Priority Outreach: Initiate direct relationship management, schedule feedback call, and send an exclusive retention incentive package."
        elif churn_prob >= threshold:
            segment = "Medium Risk"
            badge_class = "medium-risk"
            action = "Re-engagement Campaign: Dispatch personalized discount offerings, feature highlights, and tailored product recommendations."
        else:
            segment = "Low Risk"
            badge_class = "low-risk"
            action = "Standard Strategy: Maintain routine engagement workflows, loyalty rewards, and automated newsletter communications."

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Churn Probability", f"{churn_prob:.1%}")
        kpi2.metric("Decision Threshold", f"{threshold:.2f}")
        kpi3.metric("Customer Segment", segment)

        st.markdown(
            f'<div class="risk-badge {badge_class}">Status: {segment}</div>',
            unsafe_allow_html=True,
        )

        chart_df = pd.DataFrame(
            {
                "Category": ["Current Churn Risk", "Decision Threshold"],
                "Value": [churn_prob, threshold],
            }
        )

        fig = px.bar(
            chart_df,
            x="Category",
            y="Value",
            color="Category",
            color_discrete_map={
                "Current Churn Risk": (
                    "#2C5E3B" if churn_prob < threshold else "#B91C1C"
                ),
                "Decision Threshold": "#E5A93B",
            },
            text_auto=".1%",
            labels={"Category": "", "Value": "Probability Score"},
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#362417", family="Inter"),
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            yaxis=dict(
                range=[0, 1], gridcolor="#e8e0d0", zerolinecolor="#dcd3c1"
            ),
        )
        fig.update_traces(
            textfont_size=13, textposition="outside", cliponaxis=False
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("💡 Action Strategy")
        st.info(action)

        st.subheader("📋 Profile Overview")

        snapshot = [
            {"Attribute": "Age", "Value": str(age)},
            {"Attribute": "Gender", "Value": str(gender)},
            {"Attribute": "Country", "Value": str(country)},
            {
                "Attribute": "Membership Tenure",
                "Value": f"{membership_years} Years",
            },
            {
                "Attribute": "Lifetime Value",
                "Value": f"${lifetime_value:,.2f}",
            },
            {"Attribute": "Total Purchases", "Value": str(total_purchases)},
            {"Attribute": "Recency", "Value": f"{days_since_last} days ago"},
        ]

        snapshot_df = pd.DataFrame(snapshot)
        st.dataframe(snapshot_df, hide_index=True, width="stretch")

    except Exception as err:
        st.error(f"Prediction error: {err}")
else:
    st.info("Fill in the customer details and click the button to generate the churn risk analysis.")