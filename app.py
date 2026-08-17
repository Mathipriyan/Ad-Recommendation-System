
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Ad Recommendation System", page_icon="📢", layout="wide")

# --- Load saved artifacts ---
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('ctr_model.keras')
    encoders = joblib.load('encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, encoders, scaler, feature_columns

model, encoders, scaler, feature_columns = load_artifacts()

st.title("📢 Ad Recommendation System using Deep Learning")

# --- Sidebar input options ---
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ["Upload CSV", "Manual User ID"])

def preprocess_input(df):
    """Apply the same encoding used in training."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    # Ensure column order matches training
    df = df[[c for c in feature_columns if c in df.columns]]
    return df

def predict_ctr(df_processed):
    df_aligned = df_processed.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_aligned)
    preds = model.predict(X_scaled, verbose=0).ravel()
    return preds

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload user data CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", raw_df.head())

        processed_df = preprocess_input(raw_df)
        predictions = predict_ctr(processed_df)

        result_df = raw_df[['user_id', 'ad_id']].copy() if 'user_id' in raw_df.columns else raw_df.copy()
        result_df['predicted_ctr'] = predictions

        best_row = result_df.loc[result_df['predicted_ctr'].idxmax()]

        st.subheader("🎯 Best Ad Recommendation")
        st.markdown(f"- **User ID:** `{best_row.get('user_id', 'N/A')}`")
        st.markdown(f"- **Ad ID:** `{best_row.get('ad_id', 'N/A')}`")
        st.markdown(f"- **Predicted CTR:** `{best_row['predicted_ctr']:.4f}`")

        st.subheader("📋 All Predicted Ads")
        st.dataframe(result_df.sort_values('predicted_ctr', ascending=False).reset_index(drop=True))

else:
    manual_id = st.sidebar.text_input("Enter User ID")
    if st.sidebar.button("Get Recommendation") and manual_id:
        sample_df = pd.read_csv('user_sample.csv')
        user_rows = sample_df[sample_df['user_id'].astype(str) == manual_id]

        if user_rows.empty:
            st.warning("User ID not found in sample data. Try one from user_sample.csv.")
        else:
            processed_df = preprocess_input(user_rows)
            predictions = predict_ctr(processed_df)

            result_df = user_rows[['user_id', 'ad_id']].copy()
            result_df['predicted_ctr'] = predictions

            best_row = result_df.loc[result_df['predicted_ctr'].idxmax()]

            st.subheader("🎯 Best Ad Recommendation")
            st.markdown(f"- **User ID:** `{best_row['user_id']}`")
            st.markdown(f"- **Ad ID:** `{best_row['ad_id']}`")
            st.markdown(f"- **Predicted CTR:** `{best_row['predicted_ctr']:.4f}`")

            st.subheader("📋 All Predicted Ads for This User")
            st.dataframe(result_df.sort_values('predicted_ctr', ascending=False).reset_index(drop=True))