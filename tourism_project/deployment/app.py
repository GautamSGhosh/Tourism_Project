from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Visit with Us | Wellness Package Propensity",
    page_icon="✈️",
    layout="wide"
)

MODEL_PATH = Path(__file__).resolve().parent / "best_model.joblib"

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_model_bundle()

st.title("✈️ Visit with Us: Wellness Tourism Package Predictor")
st.markdown("""
This intelligent decision-support system predicts whether a prospective customer will purchase the
**Wellness Tourism Package** before sales outreach, optimizing sales agent productivity and maximizing campaign conversion.
""")

if bundle is None:
    st.error("⚠️ Model artifact (`best_model.joblib`) not found. Please trigger the GitHub Actions pipeline first.")
    st.stop()

st.sidebar.header("ℹ️ Operating Configuration")
st.sidebar.markdown(f"**Model Type:** `{bundle.get('model_type', 'BaggingClassifier')}`")
st.sidebar.markdown(f"**Optimal Decision Threshold:** `{bundle['threshold']:.2%}`")
st.sidebar.info("Scores at or above this threshold represent high-probability conversions recommended for prioritized outreach.")

with st.form("customer_profile_form"):
    st.subheader("📋 Customer Profile & Sales Interaction Attributes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### **Demographic & Travel Profile**")
        age = st.number_input("Customer Age", min_value=18, max_value=80, value=36, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Gross Monthly Income ($)", min_value=1000.0, max_value=150000.0, value=22500.0, step=500.0)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0, help="Tier 1 > Tier 2 > Tier 3")
        own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        passport = st.selectbox("Holds Valid Passport?", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.markdown("##### **Travel History & Pitch Details**")
        trips = st.number_input("Average Annual Trips", min_value=1.0, max_value=30.0, value=3.0, step=1.0)
        persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
        children = st.number_input("Children Under Age 5", min_value=0.0, max_value=5.0, value=1.0, step=1.0)
        property_star = st.selectbox("Preferred Hotel Rating", [3.0, 4.0, 5.0], index=0)
        product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        duration = st.number_input("Pitch Duration (Minutes)", min_value=5.0, max_value=120.0, value=16.0, step=1.0)
        pitch_score = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
        followups = st.number_input("Follow-up Calls Scheduled", min_value=1.0, max_value=10.0, value=4.0, step=1.0)

    submitted = st.form_submit_button("🔮 Predict Purchase Propensity", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
    }])[bundle["feature_columns"]]

    probability = float(bundle["model"].predict_proba(input_data)[0, 1])
    is_recommended = probability >= bundle["threshold"]

    st.markdown("---")
    st.subheader("🎯 Prediction & Sales Recommendation")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            label="Estimated Purchase Probability",
            value=f"{probability:.1%}",
            delta=f"{(probability - bundle['threshold']):.1%} vs threshold"
        )

    with res_col2:
        if is_recommended:
            st.success(f"### ✅ HIGH PROPENSITY: PRIORITIZE OUTREACH\n"
                       f"Customer score exceeds the calibrated operating threshold of **{bundle['threshold']:.1%}**. "
                       f"Assign immediately to a Senior Travel Consultant for personalized package presentation.")
        else:
            st.info(f"### ℹ️ LOW PROPENSITY: STANDARD NURTURING\n"
                    f"Customer score is below the operational cutoff of **{bundle['threshold']:.1%}**. "
                    f"Route to automated digital email drip campaigns to preserve direct sales capacity.")

    with st.expander("🔍 View Processed Input Dataframe"):
        st.dataframe(input_data, use_container_width=True)