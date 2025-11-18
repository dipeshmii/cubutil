import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# LOAD MODELS + PREPROCESSORS
# ============================
with open("random_forest_regressor.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("random_forest_classifier.pkl", "rb") as f:
    cls_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ============================
# PAGE TITLE
# ============================
st.title("🚛 Cube Utilisation Prediction App")
st.write("Predict utilisation % and efficiency class using packing + truck parameters.")

# ============================
# FEATURES FROM YOUR DATASET
# ============================
numeric_features = [
    "truck_length_m",
    "truck_width_m",
    "truck_height_m",
    "truck_volume_m3",
    "total_boxes",
    "total_box_volume_m3",
    "avg_box_volume_m3",
    "small_box_count",
    "large_box_count",
    "irregular_parts_count",
    "weight_limit_kg",
    "total_weight_kg"
]

categorical_features = [
    "pallet_pattern",
    "packing_orientation"
]

# For dropdown options, define unique classes
pallet_options = ["pinwheel", "block", "mixed", "herringbone"]
orientation_options = ["LWH", "WHL", "HLW", "mixed"]

# ============================
# USER INPUT FORM
# ============================
st.subheader("📦 Enter Input Parameters")

inputs = {}

# Numeric inputs
for col in numeric_features:
    inputs[col] = st.number_input(col, value=0.0)

# Dropdowns for categorical inputs
inputs["pallet_pattern"] = st.selectbox("pallet_pattern", pallet_options)
inputs["packing_orientation"] = st.selectbox("packing_orientation", orientation_options)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# ============================
# PREPROCESS FUNCTION
# ============================
def preprocess(df):
    df = df.copy()
    
    # Encode categorical
    df[categorical_features] = encoder.transform(df[categorical_features])
    
    # Scale numeric
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    return df

# ============================
# PREDICT BUTTON
# ============================
if st.button("Predict"):
    X = preprocess(input_df)

    # Regression prediction
    util = reg_model.predict(X)[0]

    # Classification prediction
    util_class = cls_model.predict(X)[0]

    # ============================
    # DISPLAY RESULTS
    # ============================
    st.subheader("📊 Prediction Results")
    st.metric("Cube Utilisation (%)", f"{util:.2f}")

    if util >= 85:
        st.success("🟢 Excellent Utilisation")
    elif util >= 60:
        st.warning("🟡 Moderate Utilisation")
    else:
        st.error("🔴 Poor Utilisation")

    st.write(f"**Efficiency Class:** {util_class}")

    st.subheader("📌 Suggested Improvements")
    if util < 60:
        st.write("- Change pallet pattern (block/mixed recommended)")
        st.write("- Rotate boxes to improve density")
        st.write("- Reduce gaps between large & small boxes")
        st.write("- Avoid too many irregular parts in one layer")
    elif util < 85:
        st.write("- Minor adjustment in box orientation could help")
        st.write("- Mix small + large boxes for tighter fit")
    else:
        st.write("- Great loading efficiency! Continue same strategy.")

st.write("---")
st.caption("Built using RandomForest models | Dataset-specific UI | Designed for Royal Enfield Logistics Demo")
