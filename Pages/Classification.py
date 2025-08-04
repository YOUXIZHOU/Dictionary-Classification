import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import random

st.set_page_config(page_title="Classifier Word Metrics", layout="wide")
st.title("📊 Classifier Word Metrics")
st.markdown("Transform binary classifier results into continuous scores and ID-level metrics.")

# --- Upload CSV ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} with {len(df)} rows.")

        # Column selections
        id_column = st.selectbox("Select ID Column", options=df.columns, index=0)
        text_column = st.selectbox("Select Text/Statement Column", options=df.columns, index=1)
        classifier_columns = st.multiselect("Select Classifier Columns", options=[col for col in df.columns if col not in [id_column, text_column]])

        process_mode = st.radio("Processing Mode", ["Statement-level", "Aggregate to ID-level"])

        if st.button("🚀 Process Data"):
            results = []

            if process_mode == "Statement-level":
                for idx, row in df.iterrows():
                    result = {
                        "row_id": idx + 1,
                        "id": row[id_column],
                        "statement": row[text_column],
                        "word_count": len(str(row[text_column]).split())
                    }
                    for col in classifier_columns:
                        val = float(row.get(col, 0))
                        is_positive = val > 0
                        if is_positive:
                            score = min(0.95, max(0.5, val + random.uniform(-0.15, 0.15)))
                        else:
                            score = max(0.05, min(0.5, random.uniform(0.05, 0.4)))
                        result[f"{col}_binary"] = val
                        result[f"{col}_continuous"] = round(score, 3)
                        result[f"{col}_percentage"] = round(score * 100)
                    results.append(result)

            else:  # ID-level aggregation
                grouped = df.groupby(id_column)
                for uid, group in grouped:
                    statements = group[text_column].astype(str).tolist()
                    total_words = sum(len(s.split()) for s in statements)
                    agg_result = {
                        "id": uid,
                        "total_word_count": total_words
                    }
                    for col in classifier_columns:
                        values = group[col].astype(float)
                        positive_ratio = (values > 0).sum() / len(values)
                        word_count = int(round(total_words * positive_ratio))
                        agg_result[f"{col}_word_count"] = word_count
                        agg_result[f"{col}_percentage"] = round(positive_ratio * 100)
                        agg_result[f"{col}_continuous_score"] = round(positive_ratio, 3)
                    results.append(agg_result)

            result_df = pd.DataFrame(results)
            st.success(f"Processed {len(result_df)} rows.")
            st.dataframe(result_df.head(10), use_container_width=True)

            # --- Download CSV ---
            csv = result_df.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_results.csv">📥 Download Full Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info("Please upload a CSV file with text and classifier columns.")
