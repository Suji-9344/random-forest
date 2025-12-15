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

# ===============================
# CASE: ONLY ONE COLUMN DATASET
# ===============================
if df.shape[1] == 1:

    y = df.iloc[:, 0]

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    # Dummy feature
    X = np.arange(len(y)).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    st.success("✅ Random Forest model trained successfully!")

    # USER INPUT
    st.sidebar.header("Enter Input Value")

    user_value = st.sidebar.number_input(
        "Dummy Feature Value",
        step=1.0
    )

    if st.button("Predict"):
        user_input = np.array([[user_value]])
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)

        st.subheader("Prediction Result")
        st.write("Predicted Class:", prediction)
        st.write("Prediction Probability:")
        st.write(probability)

    st.stop()

# ===============================
# NORMAL CASE (2+ COLUMNS)
# ===============================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

st.success("✅ Random Forest model trained successfully!")

st.sidebar.header("Enter Input Values")

input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(col)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write("Predicted Class:", prediction)
    st.write("Prediction Probability:")
    st.write(probability)
