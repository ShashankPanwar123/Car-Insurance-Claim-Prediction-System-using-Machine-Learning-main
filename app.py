# ===============================
# 🚗 CAR INSURANCE AI SYSTEM (FINAL FIXED)
# ===============================

import streamlit as st
import joblib
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Car Insurance AI", layout="wide")

# -----------------------
# LOAD NUMERIC MODEL
# -----------------------
@st.cache_resource
def load_numeric_model():
    return joblib.load("car_claim_numeric_model.pkl")

numeric_model = load_numeric_model()

# -----------------------
# LOAD IMAGE MODEL
# -----------------------
model = None

@st.cache_resource
def load_image_model():
    device = torch.device("cpu")
    model = torchvision.models.alexnet(weights="IMAGENET1K_V1")  # pretrained
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(torch.load("best_image_model.pth", map_location=device))
    model.eval()
    return model

# -----------------------
# LOAD DATA
# -----------------------
train_df = pd.read_csv("train.csv")

# 🔥 CLEAN DATA (FIX ERROR)
train_df["max_power"] = pd.to_numeric(train_df["max_power"], errors="coerce")
train_df["gross_weight"] = pd.to_numeric(train_df["gross_weight"], errors="coerce")

train_df["max_power"].fillna(train_df["max_power"].mean(), inplace=True)
train_df["gross_weight"].fillna(train_df["gross_weight"].mean(), inplace=True)

template = train_df.drop(columns=["is_claim"]).iloc[0:1].copy()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

classes = ["Normal", "Suspicious"]

# -----------------------
# UI
# -----------------------
st.title("🚗 Car Insurance Smart Assistant")

left, right = st.columns(2)

# =======================
# LEFT SIDE
# =======================
with left:
    st.subheader("👤 Customer Details")

    age = st.slider("Age", 18, 80, 30)
    car_age = st.slider("Car Age", 0, 20, 2)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    airbags = st.slider("Airbags", 0, 10, 2)

    uploaded_file = st.file_uploader("Upload Car Image", type=["jpg","png","jpeg"])

    predict_btn = st.button("🚀 Check Claim Status")

# =======================
# RIGHT SIDE
# =======================
with right:
    if predict_btn:

        with st.spinner("Analyzing..."):

            # -----------------------
            # PREPARE INPUT
            # -----------------------
            input_data = template.copy()

            fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}

            input_data["age_of_policyholder"] = age
            input_data["age_of_car"] = car_age
            input_data["fuel_type"] = fuel_map[fuel]
            input_data["airbags"] = airbags

            # 🔥 USE CLEANED DATA
            input_data["max_power"] = train_df["max_power"].mean() + age
            input_data["gross_weight"] = train_df["gross_weight"].mean() + car_age * 10

            # Feature engineering
            input_data["engine_per_weight"] = input_data["max_power"] / (input_data["gross_weight"] + 1)
            input_data["age_ratio"] = car_age / (age + 1)

            input_data.fillna(0, inplace=True)

            # -----------------------
            # NUMERIC PREDICTION
            # -----------------------
            claim_prob = numeric_model.predict_proba(input_data)[:,1][0]
            claim_prob = round(claim_prob, 4)

            st.subheader("📊 Claim Analysis")

            st.metric("Chance of Claim", f"{claim_prob*100:.2f}%")
            st.metric("Chance of No Claim", f"{(1-claim_prob)*100:.2f}%")

            st.progress(int(claim_prob * 100))

            # -----------------------
            # IMAGE MODEL
            # -----------------------
            confidence = 0
            image_pred = "Not Provided"

            if uploaded_file:

                if model is None:
                    model = load_image_model()

                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, use_container_width=True)

                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, 1).item()
                    confidence = probs[0][pred].item()

                confidence = round(confidence, 4)

                if confidence < 0.6:
                    image_pred = "Uncertain"
                    st.warning("⚠️ Image unclear — result may vary")
                else:
                    image_pred = classes[pred]

                st.subheader("🧠 Fraud Check Result")
                st.write(f"Result: {image_pred}")
                st.write(f"Confidence Level: {confidence*100:.2f}%")

            # -----------------------
            # FINAL DECISION
            # -----------------------
            if confidence < 0.6:
                approval_score = (1 - claim_prob)
            else:
                approval_score = (1 - claim_prob) * 0.7 + confidence * 0.3

            approval_score = round(approval_score, 4)

            st.subheader("🎯 Final Decision")

            st.metric("Chance of Claim Approval", f"{approval_score*100:.2f}%")

            if approval_score > 0.7:
                st.success("✅ High chance of approval")
            elif approval_score > 0.4:
                st.warning("⚠️ Needs review")
            else:
                st.error("❌ Low chance of approval")

            st.info("💡 Results may vary slightly due to AI prediction behavior.")