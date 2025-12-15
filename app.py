import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Random Forest Prediction App")

st.title("🌳 Random Forest Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_pickle("predictions_df.pkl")

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Separate features and target
target_column = df.columns[-1]   # last column as target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
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
    input_data[col] = st.sidebar.number_input(
        f"{col}", value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {prediction}")
    st.write("**Prediction Probability:**")
    st.write(prediction_proba)

