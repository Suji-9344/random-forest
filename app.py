import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Random Forest App")

st.title("🌳 Random Forest Prediction App")

@st.cache_data
def load_data():
    return pd.read_pickle("predictions_df.pkl")

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())
st.write("Dataset Shape:", df.shape)

# 🚑 AUTO-FIX: if dataset has only ONE column
if df.shape[1] == 1:
    st.warning("⚠ Only one column found. Adding a dummy feature automatically.")
    df["dummy_feature"] = np.arange(len(df))

# Features & target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode feature columns safely
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target if needed
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y.astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

st.success("✅ Random Forest model trained successfully!")

# Sidebar inputs
st.sidebar.header("Enter Input Values")

input_data = {}

for col in X.columns:
    mean_val = X[col].mean()

    # 🔒 ABSOLUTE SAFETY
    if pd.isna(mean_val) or np.isinf(mean_val):
        mean_val = 0.0

    input_data[str(col)] = st.sidebar.number_input(
        label=str(col),      # ✅ FORCE STRING
        value=float(mean_val)
    )

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write("Predicted Class:", prediction)
    st.write("Prediction Probab
