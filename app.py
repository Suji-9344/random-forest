import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Random Forest App")

st.title("🌳 Random Forest Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_pickle("predictions_df.pkl")

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Target column (last column)
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features
label_encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Encode target if needed
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y.astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
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

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {prediction}")
    st.write("**Prediction Probability:**")
    st.write(probability)
